cmake_minimum_required(VERSION 2.8)
project(ORB_GPU)

SET(CMAKE_BUILD_TYPE Release)

MESSAGE("Build type: " ${CMAKE_BUILD_TYPE})

set(CMAKE_CXX_STANDARD 11)

find_package(OpenCV 3.4 REQUIRED)

include_directories(
${PROJECT_SOURCE_DIR}/include
${OPENCV_INCLUDE_DIRS}
)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR})

add_library(${PROJECT_NAME} STATIC
src/CPU_ORB.cc
src/GPU_ORB.cc
)

target_link_libraries(${PROJECT_NAME}
${OpenCV_LIBS}
${OpenCV_LIBRARY_DIRS}
)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR})

add_executable(testmain testmain.cc)

target_link_libraries(testmain 
${PROJECT_NAME}
)


