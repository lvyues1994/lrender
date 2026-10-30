file(GLOB_RECURSE lr_present_image_sources CONFIGURE_DEPENDS src/*.cpp include/*.h)

add_library(lr_present_image STATIC ${lr_present_image_sources})
add_library(lr::present_image ALIAS lr_present_image)

target_include_directories(lr_present_image PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>")
target_link_libraries(lr_present_image
    PUBLIC lr::render lr::runtime
    PRIVATE stackfull::runtime lr_build_options)
