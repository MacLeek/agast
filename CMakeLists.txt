cmake_minimum_required(VERSION 2.8.3)
project(agastpp)

set(CMAKE_CXX_FLAGS "-Wall -std=c++11 -fPIC")
set(CMAKE_BUILD_TYPE Release)
## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
    )
find_package(OpenCV)

catkin_package(
    INCLUDE_DIRS include
    LIBRARIES agastpp
    )

include_directories(include)
include_directories(
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
add_executable(demo
    src/demo/demo.cc
    )
target_link_libraries(demo
    ${PROJECT_NAME}
    )

#############
## Testing ##
#############

## Add gtest based cpp test target and link libraries
# catkin_add_gtest(${PROJECT_NAME}-test test/test_agastpp.cpp)
# if(TARGET ${PROJECT_NAME}-test)
#   target_link_libraries(${PROJECT_NAME}-test ${PROJECT_NAME})
# endif()

## Add folders to be run by python nosetests
# catkin_add_nosetests(test)
