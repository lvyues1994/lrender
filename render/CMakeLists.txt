file(GLOB_RECURSE lr_render_sources CONFIGURE_DEPENDS src/*.cpp src/*.h include/*.h kernels/*.h)

add_library(lr_render STATIC ${lr_render_sources})
target_include_directories(lr_render
    PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(lr_render
    PUBLIC lr::core lr::gpu lr::runtime tc
    PRIVATE lr_build_options)

tc_add_kernels(lr_render HEADERS
    kernels/Visibility.h
    kernels/Lights.h
    kernels/Shading.h
    kernels/Temporal.h
    kernels/PathTrace.h
    kernels/Post.h
    kernels/Textures.h)

# tc_add_kernels writes the GLSL and SPIR-V here; Renderer::reloadKernels reads them back.
if(LRENDER_KERNEL_RELOAD)
    target_compile_definitions(lr_render PRIVATE
        LR_BUILT_IN_KERNEL_DIRECTORY="${CMAKE_CURRENT_BINARY_DIR}/tc_generated/lr_render/glsl")
endif()

# Each transpiled kernel registers itself from a static initializer that nothing refers
# to, so consumers link the whole archive or the linker drops the registrations.
add_library(lr_render_whole INTERFACE)
target_link_libraries(lr_render_whole INTERFACE "$<LINK_LIBRARY:WHOLE_ARCHIVE,lr_render>")
add_library(lr::render ALIAS lr_render_whole)
