# ChatSession (request/chat_request.cpp) driven through a fake AutoModel subclass.
#
# Links the real AutoModel base (automodel.cpp), the backend registry (model_backend.cpp),
# the chat library and xgrammar exactly as phi4_core's frontend test does, with the
# builtin backend set emptied in the test file. No prebuilt engine library is linked,
# so no NPU DLL is loaded for the engine and none is opened.
#
# Needs, as cache variables (pass them from the command line):
#   -DXRT_INCLUDE_DIR=<.../runtime_src/core/include>     headers only; automodel.hpp includes the NPU runtime
#   -DXRT_GEN_INCLUDE_DIR=<dir with xrt/detail/version.h>
#   -DFLM_VCPKG_INCLUDE_DIR=<vcpkg installed x64-windows include>   Boost
#   -DFLM_XRT_LIB_DIR=<dir with xrt_coreutil.lib>
#   -DFLM_TOKENIZERS_LIB_DIR=<dir with tokenizers_cpp.lib and tokenizers_c.lib>;<dir with sentencepiece.lib>
cmake_minimum_required(VERSION 3.22)
project(chat_session_tests LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(FLM_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../..")

set(XRT_INCLUDE_DIR "C:/dev/XRT/src/runtime_src/core/include" CACHE PATH "Where XRT headers live")
set(XRT_GEN_INCLUDE_DIR "" CACHE PATH "Extra include dir for generated XRT headers (xrt/detail/version.h)")
set(FLM_VCPKG_INCLUDE_DIR "" CACHE PATH "vcpkg installed include dir (Boost for XRT)")
set(FLM_XRT_LIB_DIR "" CACHE PATH "Directory with xrt_coreutil.lib")
set(FLM_TOKENIZERS_LIB_DIR "" CACHE PATH "Directories with tokenizers_cpp.lib, tokenizers_c.lib, sentencepiece.lib (;-separated)")
set(FLM_TOKENIZERS_INCLUDE_DIR "${FLM_SOURCE_DIR}/../third_party/tokenizers-cpp/include" CACHE PATH "tokenizers-cpp headers")

# The chat library and the xgrammar static library (flm_xgrammar) it needs.
include("${FLM_SOURCE_DIR}/common/chat/chat_sources.cmake")

add_executable(test_chat_session
    test_chat_session.cpp
    "${FLM_SOURCE_DIR}/common/request/chat_request.cpp"
    "${FLM_SOURCE_DIR}/common/AutoModel/automodel.cpp"
    "${FLM_SOURCE_DIR}/common/AutoModel/model_backend.cpp"
    "${FLM_SOURCE_DIR}/common/modules/sampler.cpp"
    "${FLM_SOURCE_DIR}/common/tokenizer/tokenizer.cpp"
    "${FLM_SOURCE_DIR}/common/utils.cpp"
    ${FLM_CHAT_SOURCES})

target_include_directories(test_chat_session PRIVATE
    "${CMAKE_CURRENT_LIST_DIR}"
    "${FLM_SOURCE_DIR}/include"
    "${FLM_SOURCE_DIR}/common"
    "${FLM_SOURCE_DIR}/../include"
    "${FLM_TOKENIZERS_INCLUDE_DIR}"
    "${XRT_INCLUDE_DIR}")
if(XRT_GEN_INCLUDE_DIR)
    target_include_directories(test_chat_session PRIVATE "${XRT_GEN_INCLUDE_DIR}")
endif()
if(FLM_VCPKG_INCLUDE_DIR)
    target_include_directories(test_chat_session PRIVATE "${FLM_VCPKG_INCLUDE_DIR}")
endif()

# The definitions flm_lib gives every source (src/CMakeLists.txt).
target_compile_definitions(test_chat_session PRIVATE
    DISABLE_ABI_CHECK=1
    __FLM_VERSION__="test"
    __NPU_VERSION__="0.0.0.0"
    CMAKE_INSTALL_PREFIX="${CMAKE_CURRENT_BINARY_DIR}/install-prefix"
    CMAKE_XCLBIN_PREFIX="${CMAKE_CURRENT_BINARY_DIR}/xclbin-prefix"
    WIN32_LEAN_AND_MEAN
    NOMINMAX
    _CRT_SECURE_NO_WARNINGS
    _CRT_NONSTDC_NO_DEPRECATE
    __WINDOWS__)
if(MSVC)
    target_compile_options(test_chat_session PRIVATE /utf-8 /wd4005 /wd4244)
endif()

if(FLM_TOKENIZERS_LIB_DIR)
    target_link_directories(test_chat_session PRIVATE ${FLM_TOKENIZERS_LIB_DIR})
endif()
if(FLM_XRT_LIB_DIR)
    target_link_directories(test_chat_session PRIVATE "${FLM_XRT_LIB_DIR}")
endif()
target_link_libraries(test_chat_session PRIVATE
    flm_xgrammar
    tokenizers_cpp
    tokenizers_c
    sentencepiece
    xrt_coreutil
    ntdll ws2_32 Bcrypt userenv psapi iphlpapi crypt32 secur32 advapi32 normaliz wldap32 shell32)

enable_testing()
add_test(NAME test_chat_session COMMAND test_chat_session)
