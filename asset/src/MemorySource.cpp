#include <lr/asset/AssetSource.h>

#include <utility>

namespace lr::asset {

namespace {

struct MemorySourceImpl final : AssetSource {
    explicit MemorySourceImpl(MemorySourceConfig config) : files(std::move(config.files)) {}

    AssetResult<Bytes> read(AssetPath const &path) override {
        auto const found = files.find(path);
        if (found == files.end()) {
            return assetError(AssetErrorKind::NotFound, path + ": no such file in memory");
        }
        return found->second;
    }

    std::optional<std::string> localFile(AssetPath const &) const override { return std::nullopt; }

private:
    std::unordered_map<AssetPath, Bytes> files;
};

} // namespace

std::unique_ptr<AssetSource> makeMemorySource(MemorySourceConfig config) {
    return std::make_unique<MemorySourceImpl>(std::move(config));
}

} // namespace lr::asset
