#include <lr/asset/AssetSource.h>

#include <stackfull/runtime/Runtime.h>

#include <cerrno>
#include <filesystem>
#include <system_error>

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

namespace lr::asset {

namespace {

struct FileDescriptor {
    explicit FileDescriptor(int const fd_) noexcept : fd(fd_) {}
    FileDescriptor(FileDescriptor const &) = delete;
    FileDescriptor &operator=(FileDescriptor const &) = delete;
    ~FileDescriptor() {
        if (fd >= 0) {
            ::close(fd);
        }
    }

    int fd;
};

tl::unexpected<AssetError> failure(AssetErrorKind const kind, std::string const &file, int const error) {
    return assetError(kind, file + ": " + std::error_code{error, std::generic_category()}.message());
}

AssetResult<Bytes> readWholeFile(std::string const &file) {
    auto const descriptor = FileDescriptor{::open(file.c_str(), O_RDONLY | O_CLOEXEC)};
    if (descriptor.fd < 0) {
        auto const error = errno;
        auto const missing = error == ENOENT or error == ENOTDIR;
        return failure(missing ? AssetErrorKind::NotFound : AssetErrorKind::ReadFailed, file, error);
    }
    struct stat status{};
    if (::fstat(descriptor.fd, &status) != 0) {
        return failure(AssetErrorKind::ReadFailed, file, errno);
    }
    if (not S_ISREG(status.st_mode)) {
        return failure(AssetErrorKind::ReadFailed, file, EISDIR);
    }
    auto bytes = Bytes(static_cast<std::size_t>(status.st_size));
    auto done = std::size_t{0};
    while (done < bytes.size()) {
        auto const got = ::read(descriptor.fd, bytes.data() + done, bytes.size() - done);
        if (got < 0 and errno == EINTR) {
            continue;
        }
        if (got < 0) {
            return failure(AssetErrorKind::ReadFailed, file, errno);
        }
        if (got == 0) {
            break;
        }
        done += static_cast<std::size_t>(got);
    }
    bytes.resize(done);
    return bytes;
}

struct FileSourceImpl final : AssetSource {
    explicit FileSourceImpl(FileSourceConfig const &config) : root(config.root) {}

    AssetResult<Bytes> read(AssetPath const &path) override {
        auto const file = fileOf(path);
        return stackfull::blocking([&file] { return readWholeFile(file); });
    }

    std::optional<std::string> localFile(AssetPath const &path) const override {
        auto error = std::error_code{};
        auto const absolute = std::filesystem::absolute(fileOf(path), error);
        if (error) {
            return std::nullopt;
        }
        return absolute.lexically_normal().string();
    }

private:
    std::string fileOf(AssetPath const &path) const {
        if (root.empty() or (not path.empty() and path.front() == '/')) {
            return path;
        }
        return (std::filesystem::path{root} / path).string();
    }

    std::string root;
};

} // namespace

std::unique_ptr<AssetSource> makeFileSource(FileSourceConfig const &config) {
    return std::make_unique<FileSourceImpl>(config);
}

} // namespace lr::asset
