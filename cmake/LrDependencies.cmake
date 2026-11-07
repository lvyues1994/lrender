# The four libraries lrender is built on, each pinned to one commit, and the decoders of
# the asset module. A local checkout
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
    GIT_TAG acb88611e1de8fbe7ed6e195ba76ad9cbb04a9dc
    SYSTEM)

FetchContent_Declare(compute_shaders
    GIT_REPOSITORY https://github.com/lvyues1994/compute_shaders.git
    GIT_TAG fb599b65d427ee8a2dd8dc4c7994fa8ed5c5bf7a
    SYSTEM)

# The asset module's: tl::expected for AssetResult, cgltf for glTF, stb_image for PNG and
# JPEG (the tests also write images with stb_image_write). Archives of pinned commits.
FetchContent_Declare(tl_expected
    URL https://github.com/TartanLlama/expected/archive/1770e3559f2f6ea4a5fb4f577ad22aeb30fbd8e4.tar.gz # v1.3.1
    URL_HASH SHA256=8e6655be615ab8dd3a52745d54c8e88258e58a5c0efe4c4133072abe82414885
    SYSTEM)

FetchContent_Declare(cgltf
    URL https://github.com/jkuhlmann/cgltf/archive/360db1a95480fe102ae9c69b27c5d101167ff5ba.tar.gz # v1.15
    URL_HASH SHA256=445d135cf793232ae6a585ca1404e4ff28d4f4dbca070689034fe780370ac84a
    SYSTEM)

FetchContent_Declare(stb
    URL https://github.com/nothings/stb/archive/003501f6d7a8668ce8b6d875e34dd97ad0d5d5b5.tar.gz
    URL_HASH SHA256=a6c31950b3bdf05eb92258768ba2a11bd44449490574b1aac67028452288c40d
    SYSTEM)

set(STACKFULL_BUILD_EXEC ON)
if(LRENDER_SANITIZE)
    # stackfull annotates its stack switches for the sanitizer only when it is built with it.
    set(STACKFULL_SANITIZE ${LRENDER_SANITIZE})
endif()
set(TC_PROGRAM_RELOAD ${LRENDER_KERNEL_RELOAD})
# tl-expected's tests would download an unpinned Catch2.
set(EXPECTED_BUILD_TESTS OFF)
set(EXPECTED_BUILD_PACKAGE OFF)

FetchContent_MakeAvailable(lexec co2 stackfull compute_shaders tl_expected cgltf stb)

# cgltf and stb are headers without a CMake project of their own.
add_library(lr_third_party_cgltf INTERFACE)
target_include_directories(lr_third_party_cgltf SYSTEM INTERFACE "${cgltf_SOURCE_DIR}")
add_library(lr_third_party_stb INTERFACE)
target_include_directories(lr_third_party_stb SYSTEM INTERFACE "${stb_SOURCE_DIR}")
