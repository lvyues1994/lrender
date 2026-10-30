# The four libraries lrender is built on, each pinned to one commit. A local checkout
# replaces a download with -DFETCHCONTENT_SOURCE_DIR_<NAME>=<path> (for example in an
# ignored CMakeUserPresets.json).
#
# lexec is made available first: stackfull::exec declares lexec as well, and FetchContent
# keeps the first declaration, so the whole build shares one lexec.

include(FetchContent)

FetchContent_Declare(lexec
    GIT_REPOSITORY https://github.com/lvyues1994/lexec.git
    GIT_TAG a2568cc5bf14f69addc7d0e04b1e5ed5c0b2173b
    SYSTEM)

FetchContent_Declare(co2
    GIT_REPOSITORY https://github.com/lvyues1994/coro.git
    GIT_TAG a265e5771b0e3ca68d6d473b9bb410fadddf4c23
    SYSTEM)

FetchContent_Declare(stackfull
    GIT_REPOSITORY https://github.com/lvyues1994/stackfull.git
    GIT_TAG 67d087c00c00b904a25cffa1420cd284175b3a19
    SYSTEM)

FetchContent_Declare(compute_shaders
    GIT_REPOSITORY https://github.com/lvyues1994/compute_shaders.git
    GIT_TAG 915657a5d625fefb608b51befc5871f9ebd069db
    SYSTEM)

set(STACKFULL_BUILD_EXEC ON)
if(LRENDER_SANITIZE)
    # stackfull annotates its stack switches for the sanitizer only when it is built with it.
    set(STACKFULL_SANITIZE ${LRENDER_SANITIZE})
endif()
set(TC_PROGRAM_RELOAD ${LRENDER_KERNEL_RELOAD})

FetchContent_MakeAvailable(lexec co2 stackfull compute_shaders)
