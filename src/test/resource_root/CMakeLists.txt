# Resource-root lookups in utils (set_resource_root / set_models_directory).
# Links utils.cpp only: no NPU runtime, no XRT and no engine library, so this builds
# and runs on any Windows host with MSVC.
cmake_minimum_required(VERSION 3.22)
project(resource_root_tests LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(FLM_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../..")

# utils/utils.hpp includes device_runtime.hpp, which includes the XRT headers even
# though nothing here touches the device. Pass the same directories build-flm.ps1 uses:
#   -DXRT_INCLUDE_DIR=<.../runtime_src/core/include> -DXRT_GEN_INCLUDE_DIR=<dir holding xrt/detail/version.h>
#   -DFLM_VCPKG_INCLUDE_DIR=<vcpkg installed x64-windows include>  (Boost headers that xrt/detail/any.h needs)
set(XRT_INCLUDE_DIR "C:/dev/XRT/src/runtime_src/core/include" CACHE PATH "Where XRT headers live")
set(XRT_GEN_INCLUDE_DIR "" CACHE PATH "Extra include dir for generated XRT headers (xrt/detail/version.h)")
set(FLM_VCPKG_INCLUDE_DIR "" CACHE PATH "vcpkg installed include dir (Boost for XRT)")

add_executable(test_resource_root
    test_resource_root.cpp
    "${FLM_SOURCE_DIR}/common/utils.cpp")
target_include_directories(test_resource_root PRIVATE
    "${FLM_SOURCE_DIR}/include"
    "${XRT_INCLUDE_DIR}")
if(XRT_GEN_INCLUDE_DIR)
    target_include_directories(test_resource_root PRIVATE "${XRT_GEN_INCLUDE_DIR}")
endif()
if(FLM_VCPKG_INCLUDE_DIR)
    target_include_directories(test_resource_root PRIVATE "${FLM_VCPKG_INCLUDE_DIR}")
endif()
# The same definitions flm_lib gives utils.cpp (src/CMakeLists.txt), minus the
# NPU-related ones that utils.cpp does not read.
target_compile_definitions(test_resource_root PRIVATE
    WIN32_LEAN_AND_MEAN
    NOMINMAX
    _CRT_SECURE_NO_WARNINGS
    _CRT_NONSTDC_NO_DEPRECATE
    __WINDOWS__
    DISABLE_ABI_CHECK=1
    CMAKE_INSTALL_PREFIX="${CMAKE_CURRENT_BINARY_DIR}/install-prefix"
    CMAKE_XCLBIN_PREFIX="${CMAKE_CURRENT_BINARY_DIR}/xclbin-prefix")
if(WIN32)
    # SHGetFolderPathA in utils.cpp
    target_link_libraries(test_resource_root PRIVATE shell32)
endif()

enable_testing()
add_test(NAME test_resource_root COMMAND test_resource_root)
