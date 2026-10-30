cmake_minimum_required(VERSION 3.25)

project(lrender
    VERSION 0.1.0
    DESCRIPTION "A C++17 rendering engine library whose GPU work is compute kernels written in C++"
    LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(LRENDER_BUILD_TESTS "Build lrender's tests" ${PROJECT_IS_TOP_LEVEL})
option(LRENDER_BUILD_EXAMPLES "Build lrender_cli and the other examples" ${PROJECT_IS_TOP_LEVEL})
option(LRENDER_BUILD_PROBES "Build the probes that measure lrender on real devices" ${PROJECT_IS_TOP_LEVEL})
option(LRENDER_WARNINGS_AS_ERRORS "Treat warnings in lrender's own code as errors" ${PROJECT_IS_TOP_LEVEL})
option(LRENDER_KERNEL_RELOAD
    "Let a running program reload edited kernels from disk (compute_shaders' TC_PROGRAM_RELOAD); off for releases" ON)
set(LRENDER_SANITIZE "" CACHE STRING "Sanitizer for lrender's own code: '' | address | thread | undefined")
set_property(CACHE LRENDER_SANITIZE PROPERTY STRINGS "" address thread undefined)

include(cmake/LrBuildOptions.cmake)
include(cmake/LrDependencies.cmake)

add_subdirectory(runtime)
add_subdirectory(core)
add_subdirectory(gpu)
add_subdirectory(render)
add_subdirectory(present)

if(LRENDER_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

if(LRENDER_BUILD_PROBES)
    add_subdirectory(probes)
endif()

if(LRENDER_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
