file(GLOB_RECURSE lr_runtime_sources CONFIGURE_DEPENDS src/*.cpp include/*.h)

add_library(lr_runtime STATIC ${lr_runtime_sources})
add_library(lr::runtime ALIAS lr_runtime)

target_include_directories(lr_runtime PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>")
target_link_libraries(lr_runtime
    PUBLIC lexec::runtime stackfull::exec
    PRIVATE lr_build_options)
