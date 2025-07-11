cmake_minimum_required(VERSION 3.10)
project(edge-impulse-sdk)

# Set the required macros for C linkage
add_definitions(-DEIDSP_SIGNAL_C_FN_POINTER=0)
add_definitions(-DEI_C_LINKAGE=1)

# Include Edge Impulse SDK utilities
include(edge-impulse-sdk/cmake/utils.cmake)

# Find all model and SDK source files
RECURSIVE_FIND_FILE_APPEND(MODEL_SOURCE "tflite-model" "*.cpp")
RECURSIVE_FIND_FILE_APPEND(MODEL_SOURCE "model-parameters" "*.cpp")
RECURSIVE_FIND_FILE_APPEND(MODEL_SOURCE "edge-impulse-sdk" "*.cpp")
RECURSIVE_FIND_FILE_APPEND(MODEL_SOURCE "edge-impulse-sdk/third_party" "*.cpp")
# TensorFlow Lite Micro uses .cc extension
RECURSIVE_FIND_FILE_APPEND(MODEL_SOURCE "edge-impulse-sdk/tensorflow" "*.cc")
list(APPEND MODEL_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/edge_impulse_c_api.cpp")
list(APPEND MODEL_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/tflite_detection_postprocess_wrapper.cc")
# Explicitly add detection_postprocess.cc to ensure the symbol is included
list(APPEND MODEL_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/edge-impulse-sdk/tensorflow/lite/micro/kernels/detection_postprocess.cc")

# Create a static library instead of an executable
add_library(edge-impulse-sdk STATIC ${MODEL_SOURCE})

# Ensure Release builds use high optimization
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -DNDEBUG -flto -ffast-math -funroll-loops")
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -O3 -DNDEBUG -flto -ffast-math -funroll-loops")

# Enable Link Time Optimization for Release builds
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)

# Set include directories for the library
target_include_directories(edge-impulse-sdk PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/edge-impulse-sdk
    ${CMAKE_CURRENT_SOURCE_DIR}/model-parameters
    ${CMAKE_CURRENT_SOURCE_DIR}/tflite-model
    ${CMAKE_CURRENT_SOURCE_DIR}/edge-impulse-sdk/tensorflow
    ${CMAKE_CURRENT_SOURCE_DIR}/edge-impulse-sdk/tensorflow/lite
    ${CMAKE_CURRENT_SOURCE_DIR}/edge-impulse-sdk/tensorflow/lite/micro
    ${CMAKE_CURRENT_SOURCE_DIR}/edge-impulse-sdk/tensorflow/lite/micro/kernels
)

# Set C++ standard
set_target_properties(edge-impulse-sdk PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

# Add compiler flags for better compatibility
target_compile_options(edge-impulse-sdk PRIVATE
    $<$<CONFIG:Release>:-O3 -DNDEBUG -flto -ffast-math -funroll-loops>
    -fPIC
    -Wall
    -Wextra
)
