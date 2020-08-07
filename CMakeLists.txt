cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(mnist-cpp)

find_package(httplib REQUIRED)
find_package(nlohmann_json 3.2.0 REQUIRED)
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(server server.cpp)
target_link_libraries(server "${TORCH_LIBRARIES}" nlohmann_json::nlohmann_json)
set_property(TARGET server PROPERTY CXX_STANDARD 14)
