cmake_minimum_required(VERSION 3.5)

project(SVM_prepare)

set(CMAKE_CXX_STANDARD 11)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(SVM_prepare  main.cpp)

target_link_libraries(SVM_prepare ${OpenCV_LIBS})