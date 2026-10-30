file(GLOB_RECURSE lr_core_sources CONFIGURE_DEPENDS src/*.cpp include/*.h)

add_library(lr_core STATIC ${lr_core_sources})
add_library(lr::core ALIAS lr_core)

target_include_directories(lr_core PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>")
# The math types are compute_shaders' GLSL types.
target_link_libraries(lr_core PUBLIC tc PRIVATE lr_build_options)
