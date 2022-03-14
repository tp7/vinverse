cmake_minimum_required(VERSION 3.16)

project(libvinverse LANGUAGES CXX)

add_library(vinverse SHARED
    vinverse/vinverse.cpp
    vinverse/vinverse_sse2.cpp
    vinverse/vinverse_avx2.cpp
    vinverse/vinverse_avx512.cpp
)

target_include_directories(vinverse PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/vinverse
    /usr/local/include/avisynth
)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
endif()

string(TOLOWER ${CMAKE_BUILD_TYPE} build_type)
if (build_type STREQUAL debug)
    target_compile_definitions(vinverse PRIVATE DEBUG_BUILD)
else (build_type STREQUAL release)
    target_compile_definitions(vinverse PRIVATE RELEASE_BUILD)
endif ()

message(STATUS "Build type - ${CMAKE_BUILD_TYPE}")

target_compile_features(vinverse PRIVATE cxx_std_17)

set_source_files_properties(vinverse/vinverse_sse2.cpp PROPERTIES COMPILE_OPTIONS "-mfpmath=sse;-msse2")
set_source_files_properties(vinverse/vinverse_avx2.cpp PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma")
set_source_files_properties(vinverse/vinverse_avx512.cpp PROPERTIES COMPILE_OPTIONS "-mavx512f;-mavx512bw;-mavx512dq;-mavx512vl;-mfma")

find_package (Git)

if (GIT_FOUND)
    execute_process (COMMAND ${GIT_EXECUTABLE} describe --tags --abbrev=0
        OUTPUT_VARIABLE ver
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    set_target_properties(vinverse PROPERTIES OUTPUT_NAME "vinverse.${ver}")
else ()
    message (STATUS "GIT not found")
endif ()

include(GNUInstallDirs)

INSTALL(TARGETS vinverse LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}/avisynth")

# uninstall target
if(NOT TARGET uninstall)
  configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake_uninstall.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake"
    IMMEDIATE @ONLY)

  add_custom_target(uninstall
    COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake)
endif()
