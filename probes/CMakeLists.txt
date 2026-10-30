add_executable(layout_probe layout_probe/main.cpp layout_probe/LayoutKernels.h)
tc_add_kernels(layout_probe HEADERS layout_probe/LayoutKernels.h)
# The probe's kernels reuse the render module's kernel functions.
target_include_directories(layout_probe PRIVATE "${PROJECT_SOURCE_DIR}/render")
target_link_libraries(layout_probe PRIVATE lr::render tc_devices lr_build_options)
