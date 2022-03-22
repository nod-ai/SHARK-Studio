cmake_minimum_required(VERSION 3.13.4)

project(dshark_server CXX C)

add_executable(dshark_server dshark_server.cpp)

target_include_directories(dshark_server PRIVATE ../dshark_network_lib ../dshark_run_module)
target_link_directories(dshark_server PRIVATE ../dshark_network_lib ../dshark_run_module)
target_link_libraries(dshark_server PRIVATE dshark_network_lib pthread run_module)

set_target_properties(dshark_server PROPERTIES LINKER_LANGUAGE CXX)
