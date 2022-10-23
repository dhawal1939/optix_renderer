cmake_minimum_required(VERSION 3.12)

if(CUDA_TOOLKIT_ROOT_DIR)
  # Somebody called find_cuda--they are trying to use the deprecated build.
  message(WARNING "CUDA_TOOLKIT_ROOT_DIR is set. "
    "This usually means that a parent project has called find_cuda, which "
    "has been deprecated in CMake since 2018 and is no longer supported by OWL. "
    "Instead, the parent project is required to use 'intermediate' or 'modern' "
    "CMake CUDA style with first-class CUDA language support (i.e., "
    "project(foo LANGUAGES C CXX CUDA)). "
  )
endif()

set(CMAKE_BUILD_TYPE_INIT "Release")

set(CMAKE_TOOLCHAIN_FILE $ENV{VCPKG_CMAKE_PATH})

message(STATUS "${CMAKE_TOOLCHAIN_FILE}")


project(optix_renderer VERSION 0.0.1 LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(owl_dir ${PROJECT_SOURCE_DIR}/libs/owl)
add_subdirectory(${owl_dir} EXCLUDE_FROM_ALL)

# Find Packages required
find_package(imgui CONFIG REQUIRED)
find_package(glfw3 CONFIG REQUIRED)

find_library( GLFW3_LIBRARY glfw3)


include_directories(include)

embed_ptx(
  OUTPUT_TARGET
    optix_renderer_ptx
  PTX_LINK_LIBRARIES
    owl::owl
  SOURCES
    src/deviceCode.cu
)

add_executable(optix_renderer src/hostCode.cpp)

target_link_libraries(optix_renderer
  PRIVATE
    optix_renderer_ptx
    glfw
    owl::owl
    imgui::imgui
    owl_viewer
)