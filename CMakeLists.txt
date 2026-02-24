cmake_minimum_required(VERSION 3.14)
project(ransac_cpp CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(OpenCV REQUIRED)
find_package(HDF5 REQUIRED COMPONENTS CXX)

include(FetchContent)
FetchContent_Declare(
    bshoshany_thread_pool
    GIT_REPOSITORY https://github.com/bshoshany/thread-pool.git
    GIT_TAG        master
)
FetchContent_MakeAvailable(bshoshany_thread_pool)

add_executable(ransac_benchmark
    src/main.cpp
    src/plane.cpp
    src/occu.cpp
)

target_include_directories(ransac_benchmark PRIVATE
    include
    ${HDF5_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${bshoshany_thread_pool_SOURCE_DIR}/include
)

target_link_libraries(ransac_benchmark PRIVATE
    Eigen3::Eigen
    ${OpenCV_LIBS}
    ${HDF5_CXX_LIBRARIES}
)

target_compile_options(ransac_benchmark PRIVATE
    -O3 -march=native -ffast-math
)
