file(GLOB_RECURSE lr_present_vulkan_sources CONFIGURE_DEPENDS src/*.cpp src/*.h include/*.h)

add_library(lr_present_vulkan STATIC ${lr_present_vulkan_sources})
add_library(lr::present_vulkan ALIAS lr_present_vulkan)

target_include_directories(lr_present_vulkan PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>")
target_link_libraries(lr_present_vulkan
    PUBLIC lr::render Vulkan::Vulkan
    PRIVATE lr_build_options)
