cmake_minimum_required(VERSION 3.10)
project(edge-impulse-sdk)

# Set the required macros for C linkage
add_definitions(-DEIDSP_SIGNAL_C_FN_POINTER=0)
add_definitions(-DEI_C_LINKAGE=1)

# Enable XNNPACK for better performance when using full TensorFlow Lite
if(EI_CLASSIFIER_USE_FULL_TFLITE)
    add_definitions(-DEI_CLASSIFIER_USE_FULL_TFLITE=1)
    add_definitions(-DTFLITE_KERNEL_USE_XNNPACK=1)
    add_definitions(-DTFLITE_USE_XNNPACK=1)
    add_definitions(-DEI_CLASSIFIER_ENABLE_DETECTION_POSTPROCESS_OP=1)
    # Add the same defines as the official Makefile
    add_definitions(-DDISABLEFLOAT16)
endif()

# Include Edge Impulse SDK utilities
include(edge-impulse-sdk/cmake/utils.cmake)

# Find all model and SDK source files
RECURSIVE_FIND_FILE_APPEND(MODEL_SOURCE "tflite-model" "*.cpp")
RECURSIVE_FIND_FILE_APPEND(MODEL_SOURCE "model-parameters" "*.cpp")
RECURSIVE_FIND_FILE_APPEND(MODEL_SOURCE "edge-impulse-sdk" "*.cpp")
RECURSIVE_FIND_FILE_APPEND(MODEL_SOURCE "edge-impulse-sdk/third_party" "*.cpp")

# Conditionally include TensorFlow Lite source files
if(EI_CLASSIFIER_USE_FULL_TFLITE)
    # When using full TensorFlow Lite, we link against prebuilt libraries
    # Only include Edge Impulse specific TensorFlow files
    RECURSIVE_FIND_FILE_APPEND(MODEL_SOURCE "edge-impulse-sdk/tensorflow/lite/micro/kernels" "detection_postprocess.cc")
    # Include custom kernels as per the official Makefile
    RECURSIVE_FIND_FILE_APPEND(MODEL_SOURCE "edge-impulse-sdk/tensorflow/lite/kernels/custom" "*.cc")
else()
    # When using TensorFlow Lite Micro, include all TensorFlow source files
    RECURSIVE_FIND_FILE_APPEND(MODEL_SOURCE "edge-impulse-sdk/tensorflow" "*.cc")
    # Explicitly add detection_postprocess.cc to ensure the symbol is included
    list(APPEND MODEL_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/edge-impulse-sdk/tensorflow/lite/micro/kernels/detection_postprocess.cc")
endif()

list(APPEND MODEL_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/edge_impulse_c_api.cpp")
list(APPEND MODEL_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/tflite_detection_postprocess_wrapper.cc")

# Add tflite_full implementation when using full TensorFlow Lite
if(EI_CLASSIFIER_USE_FULL_TFLITE)
    set(TFLITE_FULL_IMPL_FILE "${CMAKE_CURRENT_SOURCE_DIR}/../ffi_glue/tflite_full_impl.cpp")
    if(EXISTS ${TFLITE_FULL_IMPL_FILE})
        list(APPEND MODEL_SOURCE ${TFLITE_FULL_IMPL_FILE})
    endif()
endif()

# Create a static library instead of an executable
add_library(edge-impulse-sdk STATIC ${MODEL_SOURCE})

# Ensure Release builds use high optimization
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -DNDEBUG -fno-lto -ffast-math -funroll-loops")
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -O3 -DNDEBUG -fno-lto -ffast-math -funroll-loops")

# Disable Link Time Optimization for Release builds
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE FALSE)

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
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_BINARY_DIR}
    .
    ${CMAKE_CURRENT_BINARY_DIR}/tflite-model
    # Add tensorflow-lite include directory for full TFLite
    ${CMAKE_CURRENT_SOURCE_DIR}/../tensorflow-lite
)
target_compile_definitions(edge-impulse-sdk PUBLIC INCBIN_SILENCE_BITCODE_WARNING)

# Set C++ standard
set_target_properties(edge-impulse-sdk PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

# Add compiler flags for better compatibility
target_compile_options(edge-impulse-sdk PRIVATE
    $<$<CONFIG:Release>:-O3 -DNDEBUG -fno-lto -ffast-math -funroll-loops>
    -fPIC
    -Wall
    -Wextra
    -fvisibility=hidden
)

# Link against prebuilt libraries when using full TensorFlow Lite
if(EI_CLASSIFIER_USE_FULL_TFLITE)
    # Determine the platform-specific library directory
    if(DEFINED TARGET_PLATFORM)
        set(TFLITE_LIB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../tflite/${TARGET_PLATFORM}")
    else()
        # Fallback to auto-detection
        if(APPLE)
            if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64")
                set(TFLITE_LIB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../tflite/mac-arm64")
            else()
                set(TFLITE_LIB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../tflite/mac-x86_64")
            endif()
        else()
            if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
                set(TFLITE_LIB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../tflite/linux-aarch64")
            else()
                set(TFLITE_LIB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../tflite/linux-x86")
            endif()
        endif()
    endif()

    message(STATUS "Linking against TensorFlow Lite libraries in: ${TFLITE_LIB_DIR}")

    # Add linker flags to match the official Makefile
    target_link_options(edge-impulse-sdk PRIVATE
        -Wl,--no-as-needed
        -ldl
    )

    # Link against prebuilt TensorFlow Lite and XNNPACK libraries in the correct order
    # This matches the official Makefile: -ltensorflow-lite -lcpuinfo -lfarmhash -lfft2d_fftsg -lfft2d_fftsg2d -lruy -lXNNPACK -lpthreadpool
    target_link_libraries(edge-impulse-sdk PRIVATE
        ${TFLITE_LIB_DIR}/libtensorflow-lite.a
        ${TFLITE_LIB_DIR}/libcpuinfo.a
        ${TFLITE_LIB_DIR}/libfarmhash.a
        ${TFLITE_LIB_DIR}/libfft2d_fftsg.a
        ${TFLITE_LIB_DIR}/libfft2d_fftsg2d.a
        ${TFLITE_LIB_DIR}/libruy.a
        ${TFLITE_LIB_DIR}/libXNNPACK.a
        ${TFLITE_LIB_DIR}/libpthreadpool.a
        ${TFLITE_LIB_DIR}/libflatbuffers.a
    )
endif()
