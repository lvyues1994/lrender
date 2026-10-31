#include "Gltf.h"

#include "Paths.h"

#include <lr/core/Math.h>

#define CGLTF_IMPLEMENTATION
#include <cgltf.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <map>
#include <numeric>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

namespace lr::asset {

namespace {

using core::mat4;
using core::vec2;
using core::vec3;
using core::vec4;

constexpr auto kDefaultFarPlane = 1000.0f;

struct CgltfFree {
    void operator()(cgltf_data *const data) const noexcept { cgltf_free(data); }
};

using CgltfData = std::unique_ptr<cgltf_data, CgltfFree>;

std::string describe(cgltf_result const result) {
    switch (result) {
    case cgltf_result_data_too_short:
        return "data too short";
    case cgltf_result_unknown_format:
        return "unknown format";
    case cgltf_result_invalid_json:
        return "invalid JSON";
    case cgltf_result_invalid_gltf:
        return "invalid glTF";
    case cgltf_result_out_of_memory:
        return "out of memory";
    case cgltf_result_legacy_gltf:
        return "glTF 1.0";
    default:
        return "cgltf error " + std::to_string(static_cast<int>(result));
    }
}

tl::unexpected<AssetError> corrupt(std::string message) {
    return assetError(AssetErrorKind::Corrupt, std::move(message));
}

std::string nameOf(char const *const kind, std::size_t const index, char const *const name) {
    auto text = std::string{kind} + " " + std::to_string(index);
    if (name != nullptr and *name != '\0') {
        text += " '" + std::string{name} + "'";
    }
    return text;
}

bool isDataUri(char const *const uri) noexcept { return std::strncmp(uri, "data:", 5) == 0; }

std::string decodedUri(char const *const uri) {
    auto text = std::string{uri};
    text.resize(cgltf_decode_uri(text.data()));
    return text;
}

int base64Value(char const c) noexcept {
    if (c >= 'A' and c <= 'Z') {
        return c - 'A';
    }
    if (c >= 'a' and c <= 'z') {
        return c - 'a' + 26;
    }
    if (c >= '0' and c <= '9') {
        return c - '0' + 52;
    }
    if (c == '+' or c == '-') {
        return 62;
    }
    return c == '/' or c == '_' ? 63 : -1;
}

// The bytes of a "data:<type>;base64,<data>" URI; nullopt for any other.
std::optional<Bytes> dataUriBytes(char const *const uri) {
    auto const text = std::string_view{uri};
    auto const comma = text.find(',');
    if (comma == std::string_view::npos or comma < 7 or text.substr(comma - 7, 7) != ";base64") {
        return std::nullopt;
    }
    auto bytes = Bytes{};
    bytes.reserve((text.size() - comma) / 4 * 3);
    auto bits = std::uint32_t{0};
    auto pending = 0;
    for (auto const c : text.substr(comma + 1)) {
        if (c == '=') {
            break;
        }
        auto const value = base64Value(c);
        if (value < 0) {
            return std::nullopt;
        }
        bits = ((bits << 6u) | static_cast<std::uint32_t>(value)) & 0xFFFFFFu;
        pending += 6;
        if (pending >= 8) {
            pending -= 8;
            bytes.push_back(static_cast<std::uint8_t>((bits >> static_cast<std::uint32_t>(pending)) & 0xFFu));
        }
    }
    return bytes;
}

mat4 matrixOf(cgltf_float const *const m) {
    return mat4{vec4{m[0], m[1], m[2], m[3]}, vec4{m[4], m[5], m[6], m[7]}, vec4{m[8], m[9], m[10], m[11]},
                vec4{m[12], m[13], m[14], m[15]}};
}

vec3 pointOf(mat4 const &matrix, vec3 const &point) { return (matrix * vec4{point, 1.0f}).xyz(); }

vec3 unitOr(vec3 const &direction, vec3 const &fallback) {
    auto const length = tc::length(direction);
    return length > 0.0f ? direction / length : fallback;
}

vec3 directionOf(mat4 const &matrix, vec3 const &direction) {
    return unitOr((matrix * vec4{direction, 0.0f}).xyz(), direction);
}

core::TextureWrap wrapOf(cgltf_wrap_mode const mode) noexcept {
    if (mode == cgltf_wrap_mode_clamp_to_edge) {
        return core::TextureWrap::ClampToEdge;
    }
    return mode == cgltf_wrap_mode_mirrored_repeat ? core::TextureWrap::MirroredRepeat : core::TextureWrap::Repeat;
}

core::AlphaMode alphaModeOf(cgltf_alpha_mode const mode) noexcept {
    if (mode == cgltf_alpha_mode_mask) {
        return core::AlphaMode::Mask;
    }
    return mode == cgltf_alpha_mode_blend ? core::AlphaMode::Blend : core::AlphaMode::Opaque;
}

struct ExternalFile {
    std::size_t index; // of the buffer or image in the document
    AssetPath path;
    Bytes bytes;
};

std::vector<AssetPath> pathsOf(std::vector<ExternalFile> const &files) {
    auto paths = std::vector<AssetPath>{};
    for (auto const &file : files) {
        paths.push_back(file.path);
    }
    return paths;
}

// What building one scene keeps track of; cgltf_data's arrays are indexed by pointer
// differences.
struct SceneBuild {
    cgltf_data const &data;
    ImagePolicy const &policy;
    GltfParts parts;
    std::vector<ExternalFile> const &externalImages;
    SceneData scene;
    std::vector<std::optional<std::uint32_t>> materials;
    std::optional<std::uint32_t> defaultMaterial;
    std::map<std::pair<std::size_t, std::size_t>, std::optional<std::uint32_t>> primitives;
    std::map<std::pair<std::size_t, core::ColorEncoding>, std::uint32_t> textures;
    std::map<std::pair<std::size_t, core::ColorEncoding>, core::ImageHandle> images;
};

cgltf_accessor const *attributeOf(cgltf_primitive const &primitive, cgltf_attribute_type const type) {
    for (auto index = cgltf_size{0}; index < primitive.attributes_count; ++index) {
        auto const &attribute = primitive.attributes[index];
        if (attribute.type == type and attribute.index == 0) {
            return attribute.data;
        }
    }
    return nullptr;
}

AssetResult<std::vector<float>> floatsOf(cgltf_accessor const &accessor, std::size_t const components,
                                         std::string const &label) {
    if (cgltf_num_components(accessor.type) != components) {
        return corrupt(label + " has " + std::to_string(cgltf_num_components(accessor.type)) + " components, not "
                       + std::to_string(components));
    }
    auto floats = std::vector<float>(accessor.count * components);
    if (cgltf_accessor_unpack_floats(&accessor, floats.data(), floats.size()) != floats.size()) {
        return corrupt(label + " cannot be read");
    }
    return floats;
}

std::vector<vec3> vec3sOf(std::vector<float> const &floats) {
    auto result = std::vector<vec3>{};
    for (auto index = std::size_t{0}; index + 2 < floats.size(); index += 3) {
        result.push_back(vec3{floats[index], floats[index + 1], floats[index + 2]});
    }
    return result;
}

std::vector<vec2> vec2sOf(std::vector<float> const &floats) {
    auto result = std::vector<vec2>{};
    for (auto index = std::size_t{0}; index + 1 < floats.size(); index += 2) {
        result.push_back(vec2{floats[index], floats[index + 1]});
    }
    return result;
}

AssetResult<std::vector<std::uint32_t>> cornersOf(cgltf_primitive const &primitive, std::size_t const vertices,
                                                  std::string const &label) {
    if (vertices > std::numeric_limits<std::uint32_t>::max()) {
        return assetError(AssetErrorKind::Unsupported, label + " has more than 2^32 vertices");
    }
    auto corners = std::vector<std::uint32_t>{};
    if (primitive.indices == nullptr) {
        corners.resize(vertices);
        std::iota(corners.begin(), corners.end(), 0u);
        return corners;
    }
    corners.reserve(primitive.indices->count);
    for (auto index = cgltf_size{0}; index < primitive.indices->count; ++index) {
        auto const vertex = cgltf_accessor_read_index(primitive.indices, index);
        if (vertex >= vertices) {
            return corrupt(label + " refers to vertex " + std::to_string(vertex) + " of "
                           + std::to_string(vertices));
        }
        corners.push_back(static_cast<std::uint32_t>(vertex));
    }
    return corners;
}

// Strips and fans as triangle lists, in glTF's winding.
std::vector<std::uint32_t> triangleList(std::vector<std::uint32_t> const &corners, cgltf_primitive_type const type) {
    auto list = std::vector<std::uint32_t>{};
    if (type == cgltf_primitive_type_triangles) {
        list.assign(corners.begin(), corners.begin() + static_cast<std::ptrdiff_t>(corners.size() / 3 * 3));
    } else if (type == cgltf_primitive_type_triangle_strip) {
        for (auto index = std::size_t{0}; index + 2 < corners.size(); ++index) {
            auto const odd = index % 2;
            list.insert(list.end(), {corners[index], corners[index + 1 + odd], corners[index + 2 - odd]});
        }
    } else if (type == cgltf_primitive_type_triangle_fan) {
        for (auto index = std::size_t{1}; index + 1 < corners.size(); ++index) {
            list.insert(list.end(), {corners[index], corners[index + 1], corners[0]});
        }
    }
    return list;
}

// glTF's rule for primitives without normals: every triangle gets its own corners.
core::MeshData flatShaded(std::vector<vec3> const &positions, std::vector<vec2> const &uvs,
                          std::vector<std::uint32_t> const &triangles) {
    auto mesh = core::MeshData{};
    for (auto first = std::size_t{0}; first + 2 < triangles.size(); first += 3) {
        auto const &a = positions[triangles[first]];
        auto const normal =
            unitOr(tc::cross(positions[triangles[first + 1]] - a, positions[triangles[first + 2]] - a),
                   vec3{0.0f, 0.0f, 1.0f});
        for (auto corner = first; corner < first + 3; ++corner) {
            mesh.indices.push_back(static_cast<std::uint32_t>(mesh.positions.size()));
            mesh.positions.push_back(positions[triangles[corner]]);
            mesh.normals.push_back(normal);
            if (not uvs.empty()) {
                mesh.uvs.push_back(uvs[triangles[corner]]);
            }
        }
    }
    return mesh;
}

// The attribute's vectors, as many as positions; empty where it is absent.
template <std::size_t N>
AssetResult<std::vector<float>> optionalAttribute(cgltf_primitive const &primitive, cgltf_attribute_type const type,
                                                  std::size_t const vertices, std::string const &label) {
    auto const *const accessor = attributeOf(primitive, type);
    if (accessor == nullptr) {
        return std::vector<float>{};
    }
    if (accessor->count != vertices) {
        return corrupt(label + " has " + std::to_string(accessor->count) + " values for "
                       + std::to_string(vertices) + " positions");
    }
    return floatsOf(*accessor, N, label);
}

// nullopt (with a warning) for primitives that are not triangles.
AssetResult<std::optional<core::MeshData>> meshOf(SceneBuild &build, cgltf_primitive const &primitive,
                                                  std::string const &label) {
    auto const *const positionAccessor = attributeOf(primitive, cgltf_attribute_type_position);
    if (positionAccessor == nullptr) {
        build.scene.warnings.push_back(label + " has no POSITION; left out");
        return std::optional<core::MeshData>{};
    }
    auto const vertices = positionAccessor->count;
    auto const corners = cornersOf(primitive, vertices, label);
    if (not corners) {
        return tl::make_unexpected(corners.error());
    }
    auto const triangles = triangleList(*corners, primitive.type);
    if (triangles.empty()) {
        build.scene.warnings.push_back(label + " has no triangles (points, lines or too few indices); left out");
        return std::optional<core::MeshData>{};
    }
    auto const positions = floatsOf(*positionAccessor, 3, label + " POSITION");
    auto const normals = optionalAttribute<3>(primitive, cgltf_attribute_type_normal, vertices, label + " NORMAL");
    auto const uvs = optionalAttribute<2>(primitive, cgltf_attribute_type_texcoord, vertices, label + " TEXCOORD_0");
    for (auto const *const read : {&positions, &normals, &uvs}) {
        if (not *read) {
            return tl::make_unexpected(read->error());
        }
    }
    if (normals->empty()) {
        return std::optional<core::MeshData>{flatShaded(vec3sOf(*positions), vec2sOf(*uvs), triangles)};
    }
    auto mesh = core::MeshData{};
    mesh.positions = vec3sOf(*positions);
    mesh.normals = vec3sOf(*normals);
    mesh.uvs = vec2sOf(*uvs);
    mesh.indices = triangles;
    return std::optional<core::MeshData>{std::move(mesh)};
}

core::TextureSampler samplerOf(cgltf_sampler const *const sampler) {
    auto result = core::TextureSampler{};
    if (sampler != nullptr) {
        result.wrapU = wrapOf(sampler->wrap_s);
        result.wrapV = wrapOf(sampler->wrap_t);
    }
    return result;
}

AssetResult<ByteView> imageBytes(SceneBuild const &build, cgltf_image const &image, std::size_t const index,
                                 Bytes &uriBytes, std::string const &label) {
    if (image.buffer_view != nullptr) {
        auto const &view = *image.buffer_view;
        if (view.buffer->data == nullptr) {
            return corrupt(label + " is in a buffer without data");
        }
        return ByteView{static_cast<std::uint8_t const *>(view.buffer->data) + view.offset, view.size};
    }
    if (image.uri == nullptr) {
        return corrupt(label + " has neither a URI nor a buffer view");
    }
    if (isDataUri(image.uri)) {
        auto bytes = dataUriBytes(image.uri);
        if (not bytes) {
            return corrupt(label + " has a data URI that is not base64");
        }
        uriBytes = std::move(*bytes);
        return ByteView::of(uriBytes);
    }
    auto const found = std::find_if(build.externalImages.begin(), build.externalImages.end(),
                                    [index](ExternalFile const &file) { return file.index == index; });
    if (found == build.externalImages.end()) {
        return corrupt(label + " was not read");
    }
    return ByteView::of(found->bytes);
}

AssetResult<core::ImageHandle> imageOf(SceneBuild &build, cgltf_image const &image, core::ColorEncoding const encoding) {
    auto const index = static_cast<std::size_t>(&image - build.data.images);
    auto const key = std::pair{index, encoding};
    if (auto const found = build.images.find(key); found != build.images.end()) {
        return found->second;
    }
    auto const label = nameOf("image", index, image.uri != nullptr and not isDataUri(image.uri) ? image.uri : image.name);
    auto uriBytes = Bytes{};
    auto const bytes = imageBytes(build, image, index, uriBytes, label);
    if (not bytes) {
        return tl::make_unexpected(bytes.error());
    }
    auto decoded = decodeImage(*bytes, encoding, build.policy);
    if (not decoded) {
        return assetError(decoded.error().kind, label + ": " + decoded.error().message);
    }
    auto handle = std::make_shared<core::ImageData const>(std::move(*decoded));
    build.images.emplace(key, handle);
    return handle;
}

AssetResult<std::optional<core::TextureId>> textureOf(SceneBuild &build, cgltf_texture_view const &view,
                                                      core::ColorEncoding const encoding, std::string const &label) {
    if (view.texture == nullptr) {
        return std::optional<core::TextureId>{};
    }
    if (view.texcoord != 0 or (view.has_transform and view.transform.has_texcoord and view.transform.texcoord != 0)) {
        build.scene.warnings.push_back(label + " samples a texture with TEXCOORD_1 or later; it gets TEXCOORD_0");
    }
    auto const index = static_cast<std::size_t>(view.texture - build.data.textures);
    auto const key = std::pair{index, encoding};
    if (auto const found = build.textures.find(key); found != build.textures.end()) {
        return std::optional<core::TextureId>{core::TextureId{found->second}};
    }
    if (view.texture->image == nullptr) {
        build.scene.warnings.push_back(label + ": " + nameOf("texture", index, view.texture->name)
                                       + " has no PNG, JPEG or KTX2 image (only KHR_texture_basisu or "
                                         "EXT_texture_webp); left out");
        return std::optional<core::TextureId>{};
    }
    auto const image = imageOf(build, *view.texture->image, encoding);
    if (not image) {
        return tl::make_unexpected(image.error());
    }
    auto const id = static_cast<std::uint32_t>(build.scene.textures.size());
    build.scene.textures.push_back(SceneTexture{*image, samplerOf(view.texture->sampler)});
    build.textures.emplace(key, id);
    return std::optional<core::TextureId>{core::TextureId{id}};
}

core::UvTransform uvTransformOf(cgltf_texture_transform const &transform) {
    auto result = core::UvTransform{};
    result.offset = vec2{transform.offset[0], transform.offset[1]};
    result.rotation = transform.rotation;
    result.scale = vec2{transform.scale[0], transform.scale[1]};
    return result;
}

bool sameTransform(cgltf_texture_transform const &a, cgltf_texture_transform const &b) noexcept {
    return a.offset[0] == b.offset[0] and a.offset[1] == b.offset[1] and a.rotation == b.rotation
       and a.scale[0] == b.scale[0] and a.scale[1] == b.scale[1];
}

// One transform for every texture: the first view's that has one.
void applyUvTransform(SceneBuild &build, std::vector<cgltf_texture_view const *> const &views,
                      core::MaterialDesc &material, std::string const &label) {
    auto chosen = static_cast<cgltf_texture_view const *>(nullptr);
    for (auto const *const view : views) {
        if (view->texture == nullptr or not view->has_transform) {
            continue;
        }
        if (chosen == nullptr) {
            chosen = view;
            material.uvTransform = uvTransformOf(view->transform);
        } else if (not sameTransform(chosen->transform, view->transform)) {
            build.scene.warnings.push_back(label + " transforms its textures' coordinates differently; all get the "
                                                   "first texture's KHR_texture_transform");
            return;
        }
    }
}

AssetResult<core::MaterialDesc> materialOf(SceneBuild &build, cgltf_material const &material, std::string const &label) {
    auto const &pbr = material.pbr_metallic_roughness;
    auto desc = core::MaterialDesc{};
    desc.baseColor = vec4{pbr.base_color_factor[0], pbr.base_color_factor[1], pbr.base_color_factor[2],
                          pbr.base_color_factor[3]};
    desc.metallic = pbr.metallic_factor;
    desc.roughness = pbr.roughness_factor;
    auto const strength = material.has_emissive_strength ? material.emissive_strength.emissive_strength : 1.0f;
    desc.emissive = vec3{material.emissive_factor[0], material.emissive_factor[1], material.emissive_factor[2]} * strength;
    desc.alphaMode = alphaModeOf(material.alpha_mode);
    desc.alphaCutoff = material.alpha_cutoff;
    desc.doubleSided = material.double_sided != 0;
    if (material.normal_texture.texture != nullptr) {
        desc.normalScale = material.normal_texture.scale;
    }
    if (material.unlit) {
        build.scene.warnings.push_back(label + " is KHR_materials_unlit; shaded as metallic-roughness");
    }
    if (build.parts == GltfParts::Geometry) {
        return desc;
    }
    struct Slot {
        cgltf_texture_view const *view;
        core::ColorEncoding encoding;
        std::optional<core::TextureId> *texture;
    };
    auto const slots = {
        Slot{&pbr.base_color_texture, core::ColorEncoding::Srgb, &desc.baseColorTexture},
        Slot{&pbr.metallic_roughness_texture, core::ColorEncoding::Linear, &desc.metallicRoughnessTexture},
        Slot{&material.emissive_texture, core::ColorEncoding::Srgb, &desc.emissiveTexture},
        Slot{&material.normal_texture, core::ColorEncoding::Linear, &desc.normalTexture},
    };
    auto views = std::vector<cgltf_texture_view const *>{};
    for (auto const &slot : slots) {
        auto const texture = textureOf(build, *slot.view, slot.encoding, label);
        if (not texture) {
            return tl::make_unexpected(texture.error());
        }
        *slot.texture = *texture;
        views.push_back(slot.view);
    }
    applyUvTransform(build, views, desc, label);
    return desc;
}

// glTF's default material for primitives without one.
core::MaterialDesc defaultMaterial() {
    auto desc = core::MaterialDesc{};
    desc.baseColor = vec4{1.0f};
    desc.metallic = 1.0f;
    desc.roughness = 1.0f;
    return desc;
}

AssetResult<std::uint32_t> materialIndex(SceneBuild &build, cgltf_material const *const material) {
    auto &materials = build.scene.materials;
    if (material == nullptr) {
        if (not build.defaultMaterial) {
            build.defaultMaterial = static_cast<std::uint32_t>(materials.size());
            materials.push_back(defaultMaterial());
        }
        return *build.defaultMaterial;
    }
    auto const index = static_cast<std::size_t>(material - build.data.materials);
    auto &slot = build.materials[index];
    if (slot) {
        return *slot;
    }
    auto desc = materialOf(build, *material, nameOf("material", index, material->name));
    if (not desc) {
        return tl::make_unexpected(desc.error());
    }
    slot = static_cast<std::uint32_t>(materials.size());
    materials.push_back(std::move(*desc));
    return *slot;
}

// The scene mesh of a primitive, converted the first time a node draws it.
AssetResult<std::optional<std::uint32_t>> primitiveMesh(SceneBuild &build, cgltf_mesh const &mesh,
                                                        std::size_t const primitive) {
    auto const meshIndex = static_cast<std::size_t>(&mesh - build.data.meshes);
    auto const key = std::pair{meshIndex, primitive};
    if (auto const found = build.primitives.find(key); found != build.primitives.end()) {
        return found->second;
    }
    auto const label = nameOf("mesh", meshIndex, mesh.name) + " primitive " + std::to_string(primitive);
    auto const &source = mesh.primitives[primitive];
    auto converted = meshOf(build, source, label);
    if (not converted) {
        return tl::make_unexpected(converted.error());
    }
    auto result = std::optional<std::uint32_t>{};
    if (*converted) {
        auto const material = materialIndex(build, source.material);
        if (not material) {
            return tl::make_unexpected(material.error());
        }
        result = static_cast<std::uint32_t>(build.scene.meshes.size());
        build.scene.meshes.push_back(SceneMesh{std::make_shared<core::MeshData const>(std::move(**converted)), *material});
    }
    build.primitives.emplace(key, result);
    return result;
}

void addLight(SceneBuild &build, cgltf_light const &light, mat4 const &world, std::string const &label) {
    auto const color = vec3{light.color[0], light.color[1], light.color[2]};
    if (light.type == cgltf_light_type_directional) {
        auto sun = core::DirectionalLight{};
        sun.direction = directionOf(world, vec3{0.0f, 0.0f, -1.0f});
        sun.color = color;
        sun.illuminance = light.intensity;
        sun.castsShadows = true;
        build.scene.lights.emplace_back(sun);
        return;
    }
    if (light.type == cgltf_light_type_spot) {
        build.scene.warnings.push_back(label + " is a spot light; lit as a point light without its cone");
    }
    auto lamp = core::PointLight{};
    lamp.position = pointOf(world, vec3{0.0f});
    lamp.color = color;
    lamp.intensity = light.intensity;
    lamp.range = light.range;
    build.scene.lights.emplace_back(lamp);
}

void addCamera(SceneBuild &build, cgltf_camera const &camera, mat4 const &world, std::string const &label) {
    if (camera.type != cgltf_camera_type_perspective) {
        build.scene.warnings.push_back(label + " is not a perspective camera; left out");
        return;
    }
    auto const &perspective = camera.data.perspective;
    auto result = SceneCamera{};
    result.camera.position = pointOf(world, vec3{0.0f});
    result.camera.target = result.camera.position + directionOf(world, vec3{0.0f, 0.0f, -1.0f});
    result.camera.up = directionOf(world, vec3{0.0f, 1.0f, 0.0f});
    result.camera.verticalFov = perspective.yfov;
    result.camera.nearPlane = perspective.znear;
    result.camera.farPlane = perspective.has_zfar ? perspective.zfar : kDefaultFarPlane;
    if (perspective.has_aspect_ratio) {
        result.aspectRatio = perspective.aspect_ratio;
    }
    build.scene.cameras.push_back(result);
}

AssetResult<void> visit(SceneBuild &build, cgltf_node const &node, mat4 const &parentToWorld) {
    auto local = std::array<cgltf_float, 16>{};
    cgltf_node_transform_local(&node, local.data());
    auto const world = parentToWorld * matrixOf(local.data());
    auto const label = nameOf("node", static_cast<std::size_t>(&node - build.data.nodes), node.name);
    if (node.mesh != nullptr) {
        for (auto primitive = cgltf_size{0}; primitive < node.mesh->primitives_count; ++primitive) {
            auto const mesh = primitiveMesh(build, *node.mesh, primitive);
            if (not mesh) {
                return tl::make_unexpected(mesh.error());
            }
            if (*mesh) {
                build.scene.instances.push_back(SceneInstance{**mesh, world});
            }
        }
    }
    if (build.parts == GltfParts::Everything and node.light != nullptr) {
        addLight(build, *node.light, world, label);
    }
    if (build.parts == GltfParts::Everything and node.camera != nullptr) {
        addCamera(build, *node.camera, world, label);
    }
    for (auto child = cgltf_size{0}; child < node.children_count; ++child) {
        auto const visited = visit(build, *node.children[child], world);
        if (not visited) {
            return visited;
        }
    }
    return {};
}

// The default scene's nodes, else the first scene's, else every node without a parent.
std::vector<cgltf_node const *> rootsOf(cgltf_data const &data) {
    auto const *const scene = data.scene != nullptr ? data.scene : (data.scenes_count != 0 ? data.scenes : nullptr);
    if (scene != nullptr) {
        return std::vector<cgltf_node const *>(scene->nodes, scene->nodes + scene->nodes_count);
    }
    auto roots = std::vector<cgltf_node const *>{};
    for (auto index = cgltf_size{0}; index < data.nodes_count; ++index) {
        if (data.nodes[index].parent == nullptr) {
            roots.push_back(&data.nodes[index]);
        }
    }
    return roots;
}

core::Aabb boundsOf(SceneData const &scene) {
    if (scene.instances.empty()) {
        return core::Aabb{};
    }
    auto bounds = core::Aabb{vec3{std::numeric_limits<float>::infinity()}, vec3{-std::numeric_limits<float>::infinity()}};
    for (auto const &instance : scene.instances) {
        auto const box = core::transformed(core::boundsOf(*scene.meshes[instance.mesh].mesh), instance.objectToWorld);
        bounds.min = tc::min(bounds.min, box.min);
        bounds.max = tc::max(bounds.max, box.max);
    }
    return bounds;
}

struct GltfDocumentImpl final : GltfDocument {
    GltfDocumentImpl(Bytes source_, AssetPath const &path, CgltfData data_)
        : source(std::move(source_)), data(std::move(data_)) {
        for (auto index = cgltf_size{0}; index < data->buffers_count; ++index) {
            auto const *const uri = data->buffers[index].uri;
            if (uri != nullptr and not isDataUri(uri)) {
                buffers.push_back(ExternalFile{index, resolvedPath(path, decodedUri(uri)), {}});
            }
        }
        for (auto index = cgltf_size{0}; index < data->images_count; ++index) {
            auto const *const uri = data->images[index].uri;
            if (uri != nullptr and not isDataUri(uri) and usedByTexture(index)) {
                images.push_back(ExternalFile{index, resolvedPath(path, decodedUri(uri)), {}});
            }
        }
    }

    std::vector<AssetPath> externalBuffers() const override { return pathsOf(buffers); }
    std::vector<AssetPath> externalImages() const override { return pathsOf(images); }
    void provideBuffer(std::size_t const index, Bytes bytes) override { buffers.at(index).bytes = std::move(bytes); }
    void provideImage(std::size_t const index, Bytes bytes) override { images.at(index).bytes = std::move(bytes); }

    AssetResult<SceneData> buildScene(ImagePolicy const &policy, GltfParts const parts) override {
        auto const loaded = loadBuffers();
        if (not loaded) {
            return tl::make_unexpected(loaded.error());
        }
        auto build = SceneBuild{*data, policy, parts, images, {}, {}, {}, {}, {}, {}};
        build.materials.resize(data->materials_count);
        for (auto const *const root : rootsOf(*data)) {
            auto const visited = visit(build, *root, mat4{1.0f});
            if (not visited) {
                return tl::make_unexpected(visited.error());
            }
        }
        build.scene.bounds = boundsOf(build.scene);
        return std::move(build.scene);
    }

private:
    bool usedByTexture(std::size_t const image) const noexcept {
        for (auto index = cgltf_size{0}; index < data->textures_count; ++index) {
            if (data->textures[index].image == data->images + image) {
                return true;
            }
        }
        return false;
    }

    // The external buffers in place, then the embedded ones decoded and the whole file
    // checked: accessors, views and indices within their buffers.
    AssetResult<void> loadBuffers() {
        for (auto &file : buffers) {
            auto &buffer = data->buffers[file.index];
            if (file.bytes.size() < buffer.size) {
                return corrupt(file.path + " is shorter than the buffer's byteLength");
            }
            buffer.data = file.bytes.data();
            buffer.data_free_method = cgltf_data_free_method_none;
        }
        auto const options = cgltf_options{};
        if (auto const result = cgltf_load_buffers(&options, data.get(), nullptr); result != cgltf_result_success) {
            return corrupt("cannot load the glTF buffers: " + describe(result));
        }
        if (auto const result = cgltf_validate(data.get()); result != cgltf_result_success) {
            return corrupt("invalid glTF: " + describe(result));
        }
        return {};
    }

    Bytes source; // the JSON and a .glb file's binary chunk, which `data` points into
    CgltfData data;
    std::vector<ExternalFile> buffers;
    std::vector<ExternalFile> images; // those textures use
};

} // namespace

AssetResult<std::unique_ptr<GltfDocument>> parseGltf(Bytes bytes, AssetPath const &path) {
    auto const options = cgltf_options{};
    auto *parsed = static_cast<cgltf_data *>(nullptr);
    auto const result = cgltf_parse(&options, bytes.data(), bytes.size(), &parsed);
    auto data = CgltfData{parsed};
    if (result == cgltf_result_legacy_gltf) {
        return assetError(AssetErrorKind::Unsupported, path + ": glTF 1.0; lrender loads glTF 2.0");
    }
    if (result != cgltf_result_success) {
        return corrupt(path + ": not a glTF 2.0 file (" + describe(result) + ")");
    }
    return std::unique_ptr<GltfDocument>{std::make_unique<GltfDocumentImpl>(std::move(bytes), path, std::move(data))};
}

core::MeshData mergedMesh(SceneData const &scene) {
    auto merged = core::MeshData{};
    auto const keepUvs = std::all_of(scene.instances.begin(), scene.instances.end(), [&scene](SceneInstance const &instance) {
        return not scene.meshes[instance.mesh].mesh->uvs.empty();
    });
    for (auto const &instance : scene.instances) {
        auto const &mesh = *scene.meshes[instance.mesh].mesh;
        auto const &world = instance.objectToWorld;
        auto const normals = tc::transpose(tc::inverse(world));
        auto const first = static_cast<std::uint32_t>(merged.positions.size());
        for (auto vertex = std::size_t{0}; vertex < mesh.positions.size(); ++vertex) {
            merged.positions.push_back(pointOf(world, mesh.positions[vertex]));
            merged.normals.push_back(directionOf(normals, mesh.normals[vertex]));
            if (keepUvs) {
                merged.uvs.push_back(mesh.uvs[vertex]);
            }
        }
        auto const mirrored = tc::determinant(world) < 0.0f;
        for (auto corner = std::size_t{0}; corner + 2 < mesh.indices.size(); corner += 3) {
            auto const second = std::size_t{mirrored ? 2u : 1u};
            merged.indices.insert(merged.indices.end(),
                                  {first + mesh.indices[corner], first + mesh.indices[corner + second],
                                   first + mesh.indices[corner + 3 - second]});
        }
    }
    return merged;
}

} // namespace lr::asset
