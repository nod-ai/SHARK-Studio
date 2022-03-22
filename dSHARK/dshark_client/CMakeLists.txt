cmake_minimum_required(VERSION 3.13.4)

project(dshark_client)

add_executable(dshark_client dshark_client.cpp)

target_include_directories(dshark_client PRIVATE ../dshark_network_lib)
target_link_directories(dshark_client PRIVATE ../dshark_network_lib)
target_link_libraries(dshark_client dshark_network_lib pthread)
