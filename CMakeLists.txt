cmake_minimum_required(VERSION 3.17...3.22)

project(iree-wrapper VERSION 1.0 LANGUAGES CXX C)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

list(APPEND CMAKE_MODULE_PATH "${PROJECT_BINARY_DIR}/lib/cmake/mlir")

add_subdirectory(thirdparty/iree EXCLUDE_FROM_ALL)
add_subdirectory(dSHARK)