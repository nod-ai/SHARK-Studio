cmake_minimum_required(VERSION 3.13.4)

add_library(dshark_network_lib dshark_common.h dshark_queue.h dshark_message.h dshark_connection.h dshark_client.h dshark_server.h)

set_target_properties(dshark_network_lib PROPERTIES LINKER_LANGUAGE CXX)
