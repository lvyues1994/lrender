#include <lr/present/SdlPresenter.h>

#include <SDL.h>

#include <array>
#include <atomic>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace lr::present {

namespace {

struct MailboxSlot {
    std::uint64_t frameIndex = 0;
    core::ivec2 extent{0};
    std::vector<std::uint32_t> pixels;
};

// Three slots: the producer fills its own and trades it for the shared one, marked fresh;
// the consumer trades its own for the shared one while that is fresh.
struct Mailbox {
    MailboxSlot &back() noexcept { return slots[backIndex]; }

    void publish() noexcept {
        auto const previous = shared.exchange(backIndex | kFresh, std::memory_order_acq_rel);
        backIndex = previous & kIndexMask;
    }

    // The newest slot published since the last call, or null.
    MailboxSlot const *takeLatest() noexcept {
        if ((shared.load(std::memory_order_acquire) & kFresh) == 0) {
            return nullptr;
        }
        auto const previous = shared.exchange(frontIndex, std::memory_order_acq_rel);
        frontIndex = previous & kIndexMask;
        return &slots[frontIndex];
    }

private:
    static constexpr std::uint32_t kIndexMask = 3u;
    static constexpr std::uint32_t kFresh = 4u;

    std::array<MailboxSlot, 3> slots;
    std::uint32_t backIndex = 0;  // the producer's
    std::uint32_t frontIndex = 1; // the consumer's
    std::atomic<std::uint32_t> shared{2u};
};

struct TextureDeleter {
    void operator()(SDL_Texture *const texture) const noexcept { SDL_DestroyTexture(texture); }
};

[[noreturn]] void fail(char const *call) { throw std::runtime_error{std::string{call} + ": " + SDL_GetError()}; }

SDL_Renderer *rendererOf(SdlPresenterConfig const &config) {
    if (config.renderer == nullptr) {
        throw std::invalid_argument{"lr::present: an SDL presenter needs an SDL_Renderer"};
    }
    return config.renderer;
}

struct SdlPresenterImpl final : SdlPresenter {
    explicit SdlPresenterImpl(SdlPresenterConfig const &config) : renderer(rendererOf(config)) {}

    void present(render::PresentFrame const &frame) override {
        auto const &pixels = frame.pixels;
        auto const count = static_cast<std::size_t>(pixels.extent.x) * static_cast<std::size_t>(pixels.extent.y);
        auto &slot = mailbox.back();
        slot.frameIndex = frame.frameIndex;
        slot.extent = pixels.extent;
        slot.pixels.assign(pixels.pixels, pixels.pixels + count);
        mailbox.publish();
    }

    std::optional<std::uint64_t> blitLatest() override {
        auto const *slot = mailbox.takeLatest();
        if (slot == nullptr) {
            return std::nullopt;
        }
        upload(*slot);
        if (SDL_RenderClear(renderer) != 0) {
            fail("SDL_RenderClear");
        }
        if (SDL_RenderCopy(renderer, texture.get(), nullptr, nullptr) != 0) {
            fail("SDL_RenderCopy");
        }
        SDL_RenderPresent(renderer);
        return slot->frameIndex;
    }

private:
    // Packed RGBA8 with red in the low byte is SDL's ABGR8888.
    void upload(MailboxSlot const &slot) {
        if (texture == nullptr or slot.extent != textureExtent) {
            texture.reset(SDL_CreateTexture(renderer, SDL_PIXELFORMAT_ABGR8888, SDL_TEXTUREACCESS_STREAMING,
                                            slot.extent.x, slot.extent.y));
            if (texture == nullptr) {
                fail("SDL_CreateTexture");
            }
            textureExtent = slot.extent;
        }
        if (SDL_UpdateTexture(texture.get(), nullptr, slot.pixels.data(), slot.extent.x * 4) != 0) {
            fail("SDL_UpdateTexture");
        }
    }

    SDL_Renderer *renderer;
    Mailbox mailbox;
    std::unique_ptr<SDL_Texture, TextureDeleter> texture;
    core::ivec2 textureExtent{0};
};

} // namespace

std::shared_ptr<SdlPresenter> makeSdlPresenter(SdlPresenterConfig const &config) {
    return std::make_shared<SdlPresenterImpl>(config);
}

} // namespace lr::present
