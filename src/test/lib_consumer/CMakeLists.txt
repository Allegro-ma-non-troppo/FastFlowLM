# A CMake consumer that embeds FastFlowLM as a library: add_subdirectory() of the src
# directory with the flm command-line target switched off, then a plain executable
# linked against flm_lib. The program runs CPU-only code (no NPU, no model load).
#
# Configure with the same FLM_VERSION, NPU_VERSION, vcpkg toolchain and XRT settings
# as the main build. FLM_VERSION and NPU_VERSION are required by src/CMakeLists.txt
# and have no default here.
cmake_minimum_required(VERSION 3.22)
project(flm_lib_consumer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# This directory is src/test/lib_consumer, so FastFlowLM's src/ is two levels up.
set(FLM_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../..")

# A parent project that embeds FastFlowLM does its own packaging, so the CLI target
# (flm.exe) and its install and packaging rules must not be created. This must be a
# normal variable (or a cache entry) set before add_subdirectory().
set(FLM_BUILD_CLI OFF)
add_subdirectory("${FLM_SOURCE_DIR}" flm_build)

add_executable(flm_lib_consumer consumer.cpp)
target_link_libraries(flm_lib_consumer PRIVATE flm_lib)
target_compile_definitions(flm_lib_consumer PRIVATE
    FLM_CONSUMER_MODEL_LIST_PATH="${FLM_SOURCE_DIR}/model_list.json")

enable_testing()
add_test(NAME flm_lib_consumer COMMAND flm_lib_consumer)
