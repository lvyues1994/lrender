file(GLOB lr_asset_sources CONFIGURE_DEPENDS src/*.cpp src/*.h include/lr/asset/*.h)

add_library(lr_asset STATIC ${lr_asset_sources})
add_library(lr::asset ALIAS lr_asset)

target_include_directories(lr_asset PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>")
target_link_libraries(lr_asset
    PUBLIC lr::core lr::runtime tl::expected
    PRIVATE stackfull::runtime lr_third_party_cgltf lr_third_party_stb lr_build_options)

if(ANDROID)
    target_sources(lr_asset PRIVATE src/android/AndroidAssetSource.cpp)
    target_link_libraries(lr_asset PRIVATE android)
endif()
