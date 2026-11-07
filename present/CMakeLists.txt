# One target per presenter; hosts link the ones they use.

add_subdirectory(image_sequence)

find_package(Vulkan QUIET)
if(TARGET Vulkan::Vulkan)
    add_subdirectory(vulkan)
else()
    message(STATUS "lrender: Vulkan not found, so no lr::present_vulkan")
endif()

find_package(SDL2 CONFIG QUIET)
if(TARGET SDL2::SDL2)
    add_subdirectory(sdl)
else()
    message(STATUS "lrender: SDL2 not found, so no lr::present_sdl and no lrender_view")
endif()
