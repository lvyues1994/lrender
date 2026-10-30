file(GLOB_RECURSE lr_present_sdl_sources CONFIGURE_DEPENDS src/*.cpp include/*.h)

add_library(lr_present_sdl STATIC ${lr_present_sdl_sources})
add_library(lr::present_sdl ALIAS lr_present_sdl)

target_include_directories(lr_present_sdl PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>")
target_link_libraries(lr_present_sdl
    PUBLIC lr::render SDL2::SDL2
    PRIVATE lr_build_options)
