#include <lr/present/ImageSequencePresenter.h>
#include <lr/present/Ppm.h>

#include <lexec/execution.hpp>
#include <lexec/scopes/counting_scope.hpp>
#include <stackfull/exec/spawn.hpp>
#include <stackfull/runtime/Runtime.h>
#include <stackfull/sync/WaitGroup.h>

#include <cstddef>
#include <mutex>
#include <stdexcept>
#include <utility>
#include <vector>

namespace lr::present {

namespace {

runtime::IoScheduler ioOf(ImageSequenceConfig const &config) {
    if (config.runtime == nullptr) {
        throw std::invalid_argument{"lr::present: an image sequence needs a runtime to write the files on"};
    }
    return config.runtime->io();
}

struct ImageSequencePresenterImpl final : ImageSequencePresenter {
    explicit ImageSequencePresenterImpl(ImageSequenceConfig const &config)
        : directory(config.directory), prefix(config.prefix), io(ioOf(config)) {}

    ~ImageSequencePresenterImpl() override { static_cast<void>(lexec::sync_wait(scope.join())); }

    void present(render::PresentFrame const &frame) override {
        auto const &pixels = frame.pixels;
        auto const count = static_cast<std::size_t>(pixels.extent.x) * static_cast<std::size_t>(pixels.extent.y);
        auto image = render::ImageRgba8{pixels.extent, std::vector<std::uint32_t>(pixels.pixels, pixels.pixels + count)};
        auto path = directory / (prefix + std::to_string(frame.frameIndex) + ".ppm");
        writing.add();
        lexec::spawn(stackfull::exec::spawn(io,
                                            [image = std::move(image), path = std::move(path)] {
                                                return stackfull::blocking([&image, &path] {
                                                    return writePpm(render::imageView(image), path);
                                                });
                                            })
                         | lexec::then([this](bool const written) noexcept { finished(written); })
                         | lexec::upon_error([this](auto const &) noexcept { finished(false); })
                         | lexec::upon_stopped([this]() noexcept { finished(false); }),
                     scope.get_token());
    }

    ImageSequenceProgress flush() override {
        writing.wait();
        auto const lock = std::lock_guard<std::mutex>{mutex};
        return progress;
    }

private:
    void finished(bool const written) noexcept {
        {
            auto const lock = std::lock_guard<std::mutex>{mutex};
            (written ? progress.written : progress.failed) += 1;
        }
        writing.done();
    }

    std::filesystem::path directory;
    std::string prefix;
    runtime::IoScheduler io;
    std::mutex mutex;
    ImageSequenceProgress progress;
    stackfull::sync::WaitGroup writing;
    lexec::simple_counting_scope scope;
};

} // namespace

std::shared_ptr<ImageSequencePresenter> makeImageSequencePresenter(ImageSequenceConfig const &config) {
    return std::make_shared<ImageSequencePresenterImpl>(config);
}

} // namespace lr::present
