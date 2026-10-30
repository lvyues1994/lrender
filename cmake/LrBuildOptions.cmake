# Compile and link options for lrender's own targets, linked PRIVATE. Dependencies are
# fetched as SYSTEM projects and keep their own flags.

add_library(lr_build_options INTERFACE)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(lr_build_options INTERFACE
        -Wall -Wextra -Wpedantic
        -Wshadow
        -Wnon-virtual-dtor
        -Wold-style-cast
        -Woverloaded-virtual
        -Wsuggest-override
        -Wzero-as-null-pointer-constant
        -Wconversion -Wsign-compare
        -Werror=return-type
        -Werror=vla)
    if(LRENDER_WARNINGS_AS_ERRORS)
        target_compile_options(lr_build_options INTERFACE -Werror)
    endif()
    if(LRENDER_SANITIZE)
        target_compile_options(lr_build_options INTERFACE -fsanitize=${LRENDER_SANITIZE} -fno-omit-frame-pointer)
        target_link_options(lr_build_options INTERFACE -fsanitize=${LRENDER_SANITIZE})
    endif()
endif()
