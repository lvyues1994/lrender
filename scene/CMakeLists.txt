file(GLOB lr_scene_sources CONFIGURE_DEPENDS src/*.cpp src/*.h include/lr/scene/*.h)

add_library(lr_scene STATIC ${lr_scene_sources})
add_library(lr::scene ALIAS lr_scene)

target_include_directories(lr_scene PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>")
# Scripts are co2 coroutines; lexec's co2 bridge comes with lexec through lr::runtime.
target_link_libraries(lr_scene
    PUBLIC lr::core lr::runtime co2::co2
    PRIVATE lr_build_options)
