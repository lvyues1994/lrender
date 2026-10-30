file(GLOB_RECURSE lr_gpu_sources CONFIGURE_DEPENDS src/*.cpp include/*.h)

add_library(lr_gpu STATIC ${lr_gpu_sources})
add_library(lr::gpu ALIAS lr_gpu)

target_include_directories(lr_gpu PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>")
target_link_libraries(lr_gpu
    PUBLIC lr::runtime tc
    PRIVATE tc_cpu tc_devices stackfull::runtime lr_build_options)
