#include <lr/asset/AndroidAssetSource.h>

#include <stackfull/runtime/Runtime.h>

#include <android/asset_manager.h>

#include <memory>

namespace lr::asset {

namespace {

struct AssetClose {
    void operator()(AAsset *const asset) const noexcept { AAsset_close(asset); }
};

AssetResult<Bytes> readAsset(AAssetManager *const manager, AssetPath const &path) {
    auto const asset = std::unique_ptr<AAsset, AssetClose>{AAssetManager_open(manager, path.c_str(), AASSET_MODE_BUFFER)};
    if (asset == nullptr) {
        return assetError(AssetErrorKind::NotFound, path + ": no such asset in the APK");
    }
    auto bytes = Bytes(static_cast<std::size_t>(AAsset_getLength64(asset.get())));
    auto done = std::size_t{0};
    while (done < bytes.size()) {
        auto const got = AAsset_read(asset.get(), bytes.data() + done, bytes.size() - done);
        if (got <= 0) {
            return assetError(AssetErrorKind::ReadFailed, path + ": reading the APK asset failed");
        }
        done += static_cast<std::size_t>(got);
    }
    return bytes;
}

struct AndroidAssetSourceImpl final : AssetSource {
    explicit AndroidAssetSourceImpl(AAssetManager *const manager_) : manager(manager_) {}

    AssetResult<Bytes> read(AssetPath const &path) override {
        return stackfull::blocking([this, &path] { return readAsset(manager, path); });
    }

    std::optional<std::string> localFile(AssetPath const &) const override { return std::nullopt; }

private:
    AAssetManager *manager;
};

} // namespace

std::unique_ptr<AssetSource> makeAndroidAssetSource(AAssetManager *const manager) {
    return std::make_unique<AndroidAssetSourceImpl>(manager);
}

} // namespace lr::asset
