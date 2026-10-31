include(FetchContent)

FetchContent_Declare(doctest
    URL https://github.com/doctest/doctest/archive/refs/tags/v2.5.3.tar.gz
    URL_HASH SHA256=174ebc4e769928959614789c5b4e9c3d0a0f81a62bb608756b127bfebfb21331
    SYSTEM)
set(DOCTEST_NO_INSTALL ON)
FetchContent_MakeAvailable(doctest)

add_library(lr_test_main OBJECT main.cpp)
target_link_libraries(lr_test_main PUBLIC doctest::doctest PRIVATE lr_build_options)

# lr_add_test(<name> SOURCES <files...> [KERNELS <kernel headers...>] [LIBRARIES <targets...>])
function(lr_add_test name)
    cmake_parse_arguments(PARSE_ARGV 1 arg "" "" "SOURCES;KERNELS;LIBRARIES")
    add_executable(${name} ${arg_SOURCES} ${arg_KERNELS})
    target_link_libraries(${name} PRIVATE lr_test_main ${arg_LIBRARIES} lr_build_options)
    if(arg_KERNELS)
        tc_add_kernels(${name} HEADERS ${arg_KERNELS})
    endif()
    add_test(NAME ${name} COMMAND ${name})
    if(LRENDER_SANITIZE MATCHES "address")
        set_tests_properties(${name} PROPERTIES ENVIRONMENT
            "LSAN_OPTIONS=suppressions=${CMAKE_CURRENT_SOURCE_DIR}/lsan.supp:print_suppressions=0")
    endif()
endfunction()

lr_add_test(lr_runtime_tests
    SOURCES runtime/RuntimeTest.cpp
    LIBRARIES lr::runtime stackfull::runtime)

lr_add_test(lr_dependencies_tests
    SOURCES dependencies/DependenciesTest.cpp
    KERNELS dependencies/SmokeKernels.h
    LIBRARIES lr::runtime co2::co2 tc_devices)

lr_add_test(lr_core_tests
    SOURCES core/CoreTest.cpp
    LIBRARIES lr::core)

lr_add_test(lr_gpu_tests
    SOURCES gpu/GpuQueueTest.cpp
    LIBRARIES lr::gpu)

lr_add_test(lr_render_tests
    SOURCES render/RenderWorldTest.cpp render/FrameGraphTest.cpp render/BvhTest.cpp
            render/VisibilityPipelineTest.cpp render/PathTracePipelineTest.cpp render/RendererTest.cpp
            render/TextureTest.cpp render/ShadowTest.cpp render/TransparencyTest.cpp
            render/VariantTest.cpp render/MaterialModelTest.cpp render/TemporalTest.cpp
    KERNELS render/TestKernels.h render/TestMaterialModels.h
    LIBRARIES lr::render tc_devices)
# White-box tests of the render module's internals, and of its kernels' functions on the host.
target_include_directories(lr_render_tests PRIVATE "${PROJECT_SOURCE_DIR}/render/src" "${PROJECT_SOURCE_DIR}/render")

set(lr_present_test_sources present/ImageSequencePresenterTest.cpp)
set(lr_present_test_libraries lr::present_image tc_devices)
if(TARGET lr::present_sdl)
    list(APPEND lr_present_test_sources present/SdlPresenterTest.cpp)
    list(APPEND lr_present_test_libraries lr::present_sdl)
endif()
lr_add_test(lr_present_tests
    SOURCES ${lr_present_test_sources}
    KERNELS render/TestKernels.h
    LIBRARIES ${lr_present_test_libraries})
target_include_directories(lr_present_tests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
# Files the tests write go under the build tree.
target_compile_definitions(lr_present_tests PRIVATE LR_TEST_OUTPUT_DIR="${CMAKE_CURRENT_BINARY_DIR}/output")
