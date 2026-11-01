cmake_minimum_required(VERSION 3.24)
project(schorle LANGUAGES CXX)

if(NOT APPLE)
  message(FATAL_ERROR "schorle requires macOS (Metal)")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(METAL_CPP_URL "https://developer.apple.com/metal/cpp/files/metal-cpp_26.4.zip"
    CACHE STRING "metal-cpp archive URL")
set(METAL_CPP_SHA256 "4e7587c0f351d782f1f3e7c67c649b40d0ba780c051b2f6729f00ddc61ada28c"
    CACHE STRING "SHA256 of the metal-cpp archive")

include(FetchContent)
FetchContent_Declare(metal_cpp
  URL      ${METAL_CPP_URL}
  URL_HASH SHA256=${METAL_CPP_SHA256}
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
FetchContent_MakeAvailable(metal_cpp)

add_library(metal_cpp INTERFACE)
add_library(schorle::metal_cpp ALIAS metal_cpp)
target_include_directories(metal_cpp SYSTEM INTERFACE ${metal_cpp_SOURCE_DIR})
target_link_libraries(metal_cpp INTERFACE
  "-framework Foundation" "-framework Metal" "-framework QuartzCore")

set(METAL_STD "metal4.0" CACHE STRING "Metal shading language standard")
set(METAL_FLAGS -std=${METAL_STD} -O3 CACHE STRING "Extra flags for the metal compiler")

set(SCHORLE_KERNELS ${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/ozakiI.metal)
set(SCHORLE_METALLIB ${CMAKE_CURRENT_BINARY_DIR}/schorle.metallib)

set(_air_files)
foreach(src IN LISTS SCHORLE_KERNELS)
  get_filename_component(name ${src} NAME_WE)
  set(air ${CMAKE_CURRENT_BINARY_DIR}/kernels/${name}.air)
  add_custom_command(
    OUTPUT ${air}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/kernels
    COMMAND xcrun -sdk macosx metal ${METAL_FLAGS} -c ${src} -o ${air}
    DEPENDS ${src}
    COMMENT "Compiling Metal kernel ${name}.metal"
    VERBATIM)
  list(APPEND _air_files ${air})
endforeach()

add_custom_command(
  OUTPUT ${SCHORLE_METALLIB}
  COMMAND xcrun -sdk macosx metallib ${_air_files} -o ${SCHORLE_METALLIB}
  DEPENDS ${_air_files}
  COMMENT "Linking schorle.metallib"
  VERBATIM)
add_custom_target(schorle_kernels ALL DEPENDS ${SCHORLE_METALLIB})

add_library(schorle INTERFACE)
add_library(schorle::schorle ALIAS schorle)
target_include_directories(schorle INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/src/include)
target_link_libraries(schorle INTERFACE schorle::metal_cpp)
target_compile_definitions(schorle INTERFACE SCHORLE_METALLIB_PATH="${SCHORLE_METALLIB}")
add_dependencies(schorle schorle_kernels)

add_executable(bench scripts/bench.cpp)
target_link_libraries(bench PRIVATE schorle::schorle "-framework Accelerate")
add_dependencies(bench schorle_kernels)

add_executable(profile_kernels scripts/profile.cpp)
target_link_libraries(profile_kernels PRIVATE schorle::schorle)
