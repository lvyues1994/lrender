# One target per presenter; hosts link the ones they use.

add_subdirectory(image_sequence)

find_package(SDL2 CONFIG QUIET)
if(TARGET SDL2::SDL2)
    add_subdirectory(sdl)
else()
    message(STATUS "lrender: SDL2 not found, so no lr::present_sdl and no lrender_view")
endif()
