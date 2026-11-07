cmake_minimum_required(VERSION 3.20)
project(hi3d_autoremesher LANGUAGES CXX)
if(WIN32)
  enable_language(RC)
endif()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT AUTO_REMESHER_SOURCE)
  message(FATAL_ERROR "AUTO_REMESHER_SOURCE must identify the verified pinned source")
endif()
set(TBB_BUILD_SHARED OFF CACHE BOOL "" FORCE)
set(TBB_BUILD_STATIC ON CACHE BOOL "" FORCE)
set(TBB_BUILD_TBBMALLOC OFF CACHE BOOL "" FORCE)
set(TBB_BUILD_TBBMALLOC_PROXY OFF CACHE BOOL "" FORCE)
set(TBB_BUILD_TESTS OFF CACHE BOOL "" FORCE)
add_subdirectory(${AUTO_REMESHER_SOURCE}/thirdparty/tbb tbb EXCLUDE_FROM_ALL)
# Vendored TBB predates std::byte; compile it in its original language mode.
set_target_properties(tbb_static PROPERTIES CXX_STANDARD 14 CXX_STANDARD_REQUIRED ON)
file(GLOB CORE CONFIGURE_DEPENDS ${AUTO_REMESHER_SOURCE}/src/AutoRemesher/*.cpp)
add_executable(autoremesher-core autoremesher_cli.cpp ${CORE}
  ${AUTO_REMESHER_SOURCE}/thirdparty/isotropicremesher/isotropicremesher.cpp
  ${AUTO_REMESHER_SOURCE}/thirdparty/isotropicremesher/isotropichalfedgemesh.cpp
  ${AUTO_REMESHER_SOURCE}/thirdparty/isotropicremesher/axisalignedboundingboxtree.cpp
  ${AUTO_REMESHER_SOURCE}/thirdparty/meshoptimizer/src/simplifier.cpp
  ${AUTO_REMESHER_SOURCE}/thirdparty/meshoptimizer/src/indexgenerator.cpp)
target_include_directories(autoremesher-core PRIVATE
  ${AUTO_REMESHER_SOURCE}/include ${AUTO_REMESHER_SOURCE}/thirdparty/eigen
  ${AUTO_REMESHER_SOURCE}/thirdparty/isotropicremesher
  ${AUTO_REMESHER_SOURCE}/thirdparty/meshoptimizer/src ${AUTO_REMESHER_SOURCE}/thirdparty/tbb/include)
target_compile_definitions(autoremesher-core PRIVATE NDEBUG NOMINMAX _USE_MATH_DEFINES
  AUTO_REMESHER_DEBUG=0 AUTO_REMESHER_DEV=0 __TBB_NO_IMPLICIT_LINKAGE=1)
target_link_libraries(autoremesher-core PRIVATE tbb_static)
if(MSVC)
  target_compile_options(autoremesher-core PRIVATE /O2 /bigobj /EHsc)
endif()
