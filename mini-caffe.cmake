# mini-caffe.cmake

option(USE_CUDA "Use CUDA support" OFF)
option(USE_CUDNN "Use CUDNN support" OFF)

include(${CMAKE_CURRENT_LIST_DIR}/cmake/Cuda.cmake)

if(USE_JAVA)
  find_package(JNI)
endif()

# turn on C++11
if(CMAKE_COMPILER_IS_GNUCXX OR (CMAKE_CXX_COMPILER_ID MATCHES "Clang"))
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
endif()

# include and library
if(USE_CUDA)
    include_directories(${CMAKE_CURRENT_LIST_DIR}/include)
    list(APPEND Caffe_LINKER_LIBS protobuf)
else(USE_CUDA)
    include_directories(${CMAKE_CURRENT_LIST_DIR}/include
                      /opt/intel/mkl/include)
    link_directories(/opt/intel/mkl/lib/intel64 /opt/intel/lib/intel64)
    list(APPEND Caffe_LINKER_LIBS protobuf libmkl_core)
endif(USE_CUDA)

# source file structure
file(GLOB CAFFE_INCLUDE ${CMAKE_CURRENT_LIST_DIR}/include/caffe/*.h
                        ${CMAKE_CURRENT_LIST_DIR}/include/caffe/*.hpp)
file(GLOB CAFFE_SRC ${CMAKE_CURRENT_LIST_DIR}/src/*.hpp
                    ${CMAKE_CURRENT_LIST_DIR}/src/*.cpp)
file(GLOB CAFFE_SRC_LAYERS ${CMAKE_CURRENT_LIST_DIR}/src/layers/*.hpp
                           ${CMAKE_CURRENT_LIST_DIR}/src/layers/*.cpp)
file(GLOB CAFFE_SRC_UTIL ${CMAKE_CURRENT_LIST_DIR}/src/util/*.hpp
                         ${CMAKE_CURRENT_LIST_DIR}/src/util/*.cpp)
file(GLOB CAFFE_SRC_PROTO ${CMAKE_CURRENT_LIST_DIR}/src/proto/caffe.pb.h
                          ${CMAKE_CURRENT_LIST_DIR}/src/proto/caffe.pb.cc)

# cpp code
set(CAFFE_COMPILE_CODE ${CAFFE_INCLUDE}
                       ${CAFFE_SRC}
                       ${CAFFE_SRC_LAYERS}
                       ${CAFFE_SRC_UTIL}
                       ${CAFFE_SRC_PROTO})

# cuda support
if(HAVE_CUDA)
  message(STATUS "We have CUDA support")
  # cuda code
  file(GLOB CAFFE_SRC_UTIL_CU ${CMAKE_CURRENT_LIST_DIR}/src/util/*.cu)
  file(GLOB CAFFE_SRC_LAYERS_CU ${CMAKE_CURRENT_LIST_DIR}/src/layers/*.cu)
  set(CAFFE_CUDA_CODE ${CAFFE_SRC_UTIL_CU}
                      ${CAFFE_SRC_LAYERS_CU})
  list(APPEND CAFFE_SRC_UTIL ${CAFFE_SRC_UTIL_CU})
  list(APPEND CAFFE_SRC_LAYERS ${CAFFE_SRC_LAYERS_CU})
  # cudnn support
  if(HAVE_CUDNN)
    message(STATUS "We have CUDNN support")
    # source file structure
    file(GLOB CAFFE_SRC_LAYERS_CUDNN ${CMAKE_CURRENT_LIST_DIR}/src/layers/cudnn/*.hpp
                                     ${CMAKE_CURRENT_LIST_DIR}/src/layers/cudnn/*.cpp
                                     ${CMAKE_CURRENT_LIST_DIR}/src/layers/cudnn/*.cu)
    # cuda code
    file(GLOB CAFFE_CUDNN_CUDA_CODE ${CMAKE_CURRENT_LIST_DIR}/src/layers/cudnn/*.cu)
    list(APPEND CAFFE_CUDA_CODE ${CAFFE_CUDNN_CUDA_CODE})
    # cpp code
    file(GLOB CAFFR_CUDNN_CPP_CODE ${CMAKE_CURRENT_LIST_DIR}/src/layers/cudnn/*.hpp
                                   ${CMAKE_CURRENT_LIST_DIR}/src/layers/cudnn/*.cpp)
    list(APPEND CAFFE_COMPILE_CODE ${CAFFR_CUDNN_CPP_CODE})
    set(CAFFE_SRC_LAYERS_CUDNN ${CAFFR_CUDNN_CPP_CODE}
                               ${CAFFE_CUDNN_CUDA_CODE})
  endif()
  caffe_cuda_compile(CAFFE_CUDA_OBJS ${CAFFE_CUDA_CODE})
  list(APPEND CAFFE_COMPILE_CODE ${CAFFE_CUDA_OBJS})
endif()

# file structure
source_group(include FILES ${CAFFE_INCLUDE})
source_group(src FILES ${CAFFE_SRC})
source_group(src\\layers FILES ${CAFFE_SRC_LAYERS})
source_group(src\\util FILES ${CAFFE_SRC_UTIL})
source_group(src\\proto FILES ${CAFFE_SRC_PROTO})
source_group(src\\jni FILES ${CAFFE_SRC_JNI})
source_group(src\\layers\\cudnn FILES ${CAFFE_SRC_LAYERS_CUDNN})

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  set(Caffe_LINK -Wl,-force_load caffe)
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  set(Caffe_LINK -Wl,--whole-archive caffe -Wl,--no-whole-archive)
endif()

add_definitions(-DCAFFE_EXPORTS)
add_library(caffe STATIC ${CAFFE_COMPILE_CODE})
target_link_libraries(caffe ${Caffe_LINKER_LIBS})
