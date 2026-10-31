#include "TexturePool.h"

#include "EnvironmentBrdf.h"
#include "kernels/Textures.h"

#include <lr/render/Bind.h>

#include <algorithm>
#include <cstring>
#include <memory>
#include <string>
#include <utility>

namespace lr::render {

namespace {

constexpr auto kStagingAlignment = std::size_t{16};

std::size_t alignUp(std::size_t const value, std::size_t const alignment) noexcept {
    return (value + alignment - 1) / alignment * alignment;
}

tc::uint groups8(std::uint32_t const size) { return groupsFor(size, 8); }

// A compressed format the device has arrays of, for the slots without a pool.
tc::CompressedFormat someArrayFormat(tc::Device const &device) {
    auto const formats = device.info().limits.compressedArrayFormats;
    for (auto bit = 0u; bit < tc::kCompressedFormatCount; ++bit) {
        if ((formats & (1u << bit)) != 0) {
            return static_cast<tc::CompressedFormat>(bit);
        }
    }
    throw tc::DeviceError{"lr::render: the device has no compressed array textures, which material kernels bind"};
}

std::uint32_t levelsOf(std::uint32_t const size) noexcept {
    auto levels = std::uint32_t{1};
    for (auto side = size; side > 1; side >>= 1) {
        ++levels;
    }
    return levels;
}

} // namespace

void TexturePool::prepare(tc::Device &device, RetireList &retired) {
    if (prepared) {
        return;
    }
    emptyColor = device.createImage<ColorPoolFormat, tc::Dim::D2Array, ColorPoolPixel>(tc::ivec3{1, 1, 1}, 1);
    auto const format = someArrayFormat(device);
    auto const block = tc::compressedFormatInfo(format).block;
    emptyBlocks = device.createImage<tc::Dim::D2Array>(format, tc::ivec3{block.x, block.y, 1}, 1);

    auto const size = static_cast<int>(environmentBrdfSize());
    environmentBrdf = device.createImage<EnvironmentBrdfFormat, tc::Dim::D2, EnvironmentBrdfPixel>(tc::ivec2{size, size}, 1);
    auto const &table = environmentBrdfTable();
    auto texels = std::make_shared<tc::BufferResource<tc::uint>>(device.createBuffer<tc::uint>(size * size));
    for (auto index = std::size_t{0}; index < table.size(); ++index) {
        (*texels)[static_cast<std::int32_t>(index)] = tc::packHalf2x16(table[index]);
    }
    device.uploadBuffer(*texels);
    device.copyBufferToImage(*texels, 0, environmentBrdf, 0);
    retired.push_back(std::move(texels));
    staging = std::vector<std::unique_ptr<tc::BufferResource<tc::uint>>>(config.framesInFlight);
    prepared = true;
}

tc::BufferResource<tc::uint> &TexturePool::stagingFor(tc::Device &device, std::size_t const bytes, RetireList &retired) {
    auto &buffer = staging[streamedFrames % staging.size()];
    if (buffer == nullptr or buffer->bytes().size() < bytes) {
        if (buffer != nullptr) {
            retired.push_back(std::shared_ptr<void>{std::move(buffer)});
        }
        auto const words = alignUp(std::max(bytes, config.streamingBytes), 4) / 4;
        buffer = std::make_unique<tc::BufferResource<tc::uint>>(
            device.createBuffer<tc::uint>(static_cast<std::int32_t>(words)));
    }
    return *buffer;
}

void TexturePool::change(tc::Device &device, PoolChange const &request, RetireList &retired) {
    auto &pool = pools[request.slot];
    auto const side = static_cast<int>(request.size);
    auto const extent = tc::ivec3{side, side, static_cast<int>(request.layers)};
    if (request.compressed) {
        auto blocks = device.createImage<tc::Dim::D2Array>(*request.compressed, extent, 0);
        pool = Pool{request.compressed, request.size, request.layers, {}, std::move(blocks)};
        return;
    }
    auto grown = device.createImage<ColorPoolFormat, tc::Dim::D2Array, ColorPoolPixel>(extent, 0);
    if (pool.size == request.size and pool.color) {
        auto kernel = kernels::CopyTextureLayer{};
        for (auto layer = 0u; layer < pool.layers; ++layer) {
            for (auto level = 0u; level < grown.levels(); ++level) {
                auto const levelSize = std::max(request.size >> level, 1u);
                device.useKernel(kernel);
                kernel.target.attach(grown, level);
                device.bindImage(kernel.target);
                bind(device, kernel.source, pool.color);
                set(device, kernel.level, static_cast<int>(level));
                set(device, kernel.layer, static_cast<int>(layer));
                set(device, kernel.size, static_cast<int>(levelSize));
                device.execute(kernel, tc::uvec3{groups8(levelSize), groups8(levelSize), 1});
            }
        }
        retired.push_back(std::make_shared<ColorPoolImage>(std::move(pool.color)));
    }
    pool = Pool{std::nullopt, request.size, request.layers, std::move(grown), {}};
}

void TexturePool::forget(core::TextureId const id) {
    if (resident.erase(id) != 0) {
        residencyChanged = true;
        ++residency;
    }
    queue.erase(std::remove_if(queue.begin(), queue.end(), [id](Pending const &pending) { return pending.upload.id == id; }),
                queue.end());
    if (sent) {
        auto &completed = sent->completed;
        completed.erase(std::remove_if(completed.begin(), completed.end(),
                                       [id](TextureUpload const &upload) { return upload.id == id; }),
                        completed.end());
    }
}

void TexturePool::apply(tc::Device &device, SceneUpdate const &update, RetireList &retired) {
    prepare(device, retired);
    for (auto const &request : update.pools) {
        change(device, request, retired);
        poolsChanged = true;
    }
    for (auto const id : update.removedTextures) {
        forget(id);
    }
    for (auto const &upload : update.textureUploads) {
        forget(upload.id);
        queue.push_back(Pending{upload, 0});
    }
}

void TexturePool::copyLevel(tc::TransferQueue &transfers, tc::BufferResource<tc::uint> const &buffer,
                            std::size_t const offset, Pending const &pending) {
    auto const &pool = pools[textureSlot(pending.upload.ref)];
    auto const layer = textureLayer(pending.upload.ref);
    if (pool.compressed) {
        transfers.copyBufferToImage(buffer, offset, pool.blocks, pending.nextLevel, layer);
    } else {
        transfers.copyBufferToImage(buffer, offset, pool.color, pending.nextLevel, layer);
    }
}

void TexturePool::generateMips(tc::Device &device, tc::uint const ref) {
    auto const &pool = pools[textureSlot(ref)];
    auto kernel = kernels::DownsampleTexture{};
    for (auto level = 1u; level < levelsOf(pool.size); ++level) {
        auto const levelSize = std::max(pool.size >> level, 1u);
        device.useKernel(kernel);
        kernel.target.attach(pool.color, level);
        device.bindImage(kernel.target);
        bind(device, kernel.source, pool.color);
        set(device, kernel.sourceLevel, static_cast<int>(level - 1));
        set(device, kernel.layer, static_cast<int>(textureLayer(ref)));
        set(device, kernel.size, static_cast<int>(levelSize));
        set(device, kernel.srgb, textureIsSrgb(ref) ? 1u : 0u);
        device.execute(kernel, tc::uvec3{groups8(levelSize), groups8(levelSize), 1});
    }
}

void TexturePool::arrive(tc::Device &device) {
    if (not sent) {
        return;
    }
    device.waitFor(sent->copies);
    for (auto const &upload : sent->completed) {
        if (not pools[textureSlot(upload.ref)].compressed) {
            generateMips(device, upload.ref);
        }
        resident[upload.id] = upload.ref;
        residencyChanged = true;
        ++residency;
    }
    sent.reset();
}

void TexturePool::send(tc::Device &device, tc::Submission const &previous, RetireList &retired) {
    if (queue.empty()) {
        return;
    }
    auto &buffer = stagingFor(device, queue.front().upload.image->levels[queue.front().nextLevel].size(), retired);
    ++streamedFrames;
    auto &transfers = device.transfers();
    transfers.waitFor(std::exchange(poolsChanged, false) ? device.submit() : previous);
    auto completed = std::vector<TextureUpload>{};
    auto used = std::size_t{0};
    while (not queue.empty()) {
        auto &pending = queue.front();
        auto const &level = pending.upload.image->levels[pending.nextLevel];
        auto const offset = alignUp(used, kStagingAlignment);
        if (used > 0 and offset + level.size() > config.streamingBytes) {
            break;
        }
        std::memcpy(buffer.data() + offset / 4, level.data(), level.size());
        device.uploadBuffer(buffer, offset / 4, alignUp(level.size(), 4) / 4);
        copyLevel(transfers, buffer, offset, pending);
        used = offset + level.size();
        ++pending.nextLevel;
        if (pending.nextLevel == pending.upload.levels) {
            completed.push_back(std::move(pending.upload));
            queue.pop_front();
        }
    }
    sent = Sent{transfers.submit(), std::move(completed)};
}

bool TexturePool::takeResidencyChange() noexcept {
    auto const changed = residencyChanged;
    residencyChanged = false;
    return changed;
}

tc::uint TexturePool::residentRef(core::TextureId const id) const {
    auto const found = resident.find(id);
    return found == resident.end() ? noTexture() : found->second;
}

MaterialTextures TexturePool::textures() const noexcept {
    auto result = MaterialTextures{};
    for (auto slot = 0u; slot < colorPoolSlots(); ++slot) {
        result.colorPools[slot] = pools[slot].color ? &pools[slot].color : &emptyColor;
    }
    for (auto slot = 0u; slot < compressedPoolSlots(); ++slot) {
        auto const &pool = pools[colorPoolSlots() + slot];
        result.compressedPools[slot] = pool.blocks ? &pool.blocks : &emptyBlocks;
    }
    result.environmentBrdf = &environmentBrdf;
    result.residency = residency;
    return result;
}

void TexturePool::clear() noexcept {
    prepared = false;
    pools = {};
    emptyColor = {};
    emptyBlocks = {};
    environmentBrdf = {};
    staging.clear();
    streamedFrames = 0;
    queue.clear();
    sent.reset();
    poolsChanged = false;
    resident.clear();
    residencyChanged = false;
    ++residency;
}

} // namespace lr::render
