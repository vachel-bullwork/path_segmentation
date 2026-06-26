cmake_minimum_required(VERSION 3.18)

# NOTE: project() declares CXX only. CUDA is enabled conditionally below so the
# package configures and builds on hosts with no CUDA toolchain (simulation).
project(path_segmentor CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

# ── Optional CUDA ─────────────────────────────────────────────────────────
# USE_CUDA=ON (default) builds the GPU backend IF a CUDA compiler is present.
# Pass -DUSE_CUDA=OFF to force a CPU-only build even on a CUDA host.
option(USE_CUDA "Build the CUDA GPU backend if a CUDA toolchain is available" ON)

include(CheckLanguage)
check_language(CUDA)

set(HAVE_CUDA FALSE)
if(USE_CUDA AND CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_ARCHITECTURES 87)               # Jetson Orin Nano = SM 8.7
  set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -DNDEBUG --use_fast_math")
  find_package(CUDAToolkit REQUIRED)
  set(HAVE_CUDA TRUE)
  message(STATUS "path_segmentor: CUDA backend ENABLED (SM 8.7)")
else()
  message(STATUS "path_segmentor: CUDA backend DISABLED — CPU-only build")
endif()

# ── ROS / ament dependencies ──────────────────────────────────────────────
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(image_geometry REQUIRED)
find_package(cv_bridge REQUIRED)

# ── OpenCV resolution ──────────────────────────────────────────────────────
# On the Jetson the custom CUDA-enabled 4.10 build lives at /opt/opencv4.10 and
# MUST be linked by full .so path (see CONTEXT.txt — avoids cmake target
# overwriting by ament/cv_bridge's internal find_package(OpenCV)).
# On a simulation host we just use whatever system OpenCV is installed.
if(HAVE_CUDA AND EXISTS "/opt/opencv4.10/lib/libopencv_core.so.4.10.0")
  set(OPENCV410_ROOT "/opt/opencv4.10")
  set(OCV_INCLUDE "${OPENCV410_ROOT}/include/opencv4")
  set(OCV_LIBS
    "${OPENCV410_ROOT}/lib/libopencv_core.so.4.10.0"
    "${OPENCV410_ROOT}/lib/libopencv_imgproc.so.4.10.0")
  set(OCV_RPATH "${OPENCV410_ROOT}/lib")
  message(STATUS "path_segmentor: OpenCV = hard-wired 4.10 CUDA at ${OPENCV410_ROOT}")
else()
  find_package(OpenCV REQUIRED COMPONENTS core imgproc)
  set(OCV_INCLUDE ${OpenCV_INCLUDE_DIRS})
  set(OCV_LIBS ${OpenCV_LIBS})
  set(OCV_RPATH "")
  message(STATUS "path_segmentor: OpenCV = system ${OpenCV_VERSION}")
endif()

# ── CUDA terrain kernels (static lib, only when CUDA is enabled) ───────────
if(HAVE_CUDA)
  add_library(terrain_kernels STATIC src/terrain_kernels.cu)
  set_target_properties(terrain_kernels PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE  ON
  )
  target_include_directories(terrain_kernels PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    ${CUDAToolkit_INCLUDE_DIRS}
  )
  target_link_libraries(terrain_kernels CUDA::cudart)
endif()

# ── Node sources (CPU backend always; CUDA backend only when enabled) ──────
set(NODE_SOURCES
  src/path_corridor_segmenter.cpp
  src/terrain_backend_cpu.cpp
)
if(HAVE_CUDA)
  list(APPEND NODE_SOURCES src/terrain_backend_cuda.cpp)
endif()

add_executable(path_corridor_segmenter ${NODE_SOURCES})

# Link OpenCV FIRST so the correct SONAME lands in DT_NEEDED ahead of
# cv_bridge's transitive OpenCV (matters only on the Jetson 4.10 setup).
target_link_libraries(path_corridor_segmenter ${OCV_LIBS})

if(HAVE_CUDA)
  target_compile_definitions(path_corridor_segmenter PRIVATE HAVE_CUDA)
  target_link_libraries(path_corridor_segmenter terrain_kernels CUDA::cudart)
  target_include_directories(path_corridor_segmenter PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
endif()

ament_target_dependencies(path_corridor_segmenter
  rclcpp
  nav_msgs
  sensor_msgs
  std_msgs
  tf2
  tf2_ros
  tf2_geometry_msgs
  image_geometry
  cv_bridge
)

target_include_directories(path_corridor_segmenter PRIVATE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  ${OCV_INCLUDE}
)

# RPATH: at runtime prefer the 4.10 CUDA build (Jetson only; empty on sim).
if(OCV_RPATH)
  set_target_properties(path_corridor_segmenter PROPERTIES
    BUILD_RPATH              "${OCV_RPATH}"
    INSTALL_RPATH            "${OCV_RPATH}"
    BUILD_WITH_INSTALL_RPATH TRUE
  )
endif()

# ── Install ─────────────────────────────────────────────────────────────────
install(TARGETS path_corridor_segmenter
  DESTINATION lib/${PROJECT_NAME}
)
install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
)

ament_package()
