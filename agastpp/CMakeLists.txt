cmake_minimum_required(VERSION 2.8.3)
project(agastpp)

add_definitions(-Wall -std=c++11 -O3 -fPIC)

find_package(OpenCV)

catkin_package(
    INCLUDE_DIRS include
    LIBRARIES ${PROJECT_NAME}
    )

include_directories(
    include
    ${OpenCV_INCLUDE_DIRS}
    )

## Declare a cpp library
file(GLOB CC_FILES src/${PROJECT_NAME}/*.cc)
add_library(${PROJECT_NAME}
    ${CC_FILES}
    )
target_link_libraries(${PROJECT_NAME}
    ${OpenCV_LIBRARIES}
    )

#add_executable(demo
    #src/demo/demo.cc
    #)
#target_link_libraries(demo
    #${PROJECT_NAME}
    #)
