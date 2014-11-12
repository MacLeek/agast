cmake_minimum_required(VERSION 2.8.3)
project(agastpp)

set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-Wall -std=c++11 -O3 -fPIC")

find_package(OpenCV)

include_directories(include
    ${OpenCV_INCLUDE_DIRS}
    )


file(GLOB CC_FILES src/${PROJECT_NAME}/*.cc)
add_library(${PROJECT_NAME} ${CC_FILES})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBRARIES})

add_executable(agast_demo src/demo/agast_demo.cc)
target_link_libraries(agast_demo ${PROJECT_NAME})
