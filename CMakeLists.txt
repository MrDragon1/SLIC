cmake_minimum_required(VERSION 3.17)
project(DIPlab3)

find_package(OpenCV REQUIRED)
set(CMAKE_CXX_STANDARD 14)

add_executable(DIPlab3 main.cpp)

set(OpenCV_LIBS opencv_core opencv_imgproc opencv_highgui opencv_imgcodecs)
target_link_libraries(DIPlab3 ${OpenCV_LIBS} )