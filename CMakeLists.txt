cmake_minimum_required(VERSION 3.13.4)

project(iree-wrapper VERSION 1.0 LANGUAGES CXX C)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

list(APPEND CMAKE_MODULE_PATH "${PROJECT_BINARY_DIR}/lib/cmake/mlir")

add_library(asio INTERFACE)
target_compile_options(asio INTERFACE ASIO_STANDALONE)
target_include_directories(asio INTERFACE thirdparty/asio/asio/include)
target_link_libraries(asio INTERFACE pthread)

add_subdirectory(thirdparty/iree EXCLUDE_FROM_ALL)
add_subdirectory(dSHARK)
