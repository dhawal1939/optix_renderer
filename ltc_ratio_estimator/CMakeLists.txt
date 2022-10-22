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

project(ltc_ratio_estimator VERSION 0.0.1 LANGUAGES C CXX CUDA)

set(OptiX_ROOT_DIR "C:/ProgramData/NVIDIA Corporation/OptiX SDK 7.5.0/")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(owl_dir ${PROJECT_SOURCE_DIR}/owl)
add_subdirectory(${owl_dir} EXCLUDE_FROM_ALL)

# Find Packages required
find_package(imgui CONFIG REQUIRED)
find_package(glfw3 CONFIG REQUIRED)

# hardcoding it for quick fix
set( GLFW3_LIBRARY "C:/Users/dhawals/repos/vcpkg/installed/x64-windows/lib/glfw3dll.lib")

message(STATUS "${GLFW3_LIBRARY}")


embed_ptx(
  OUTPUT_TARGET
    ltc_ratio_estimator_ptx
  PTX_LINK_LIBRARIES
    owl::owl
  SOURCES
    deviceCode.cu
)

add_executable(ltc_ratio_estimator hostCode.cpp)

target_link_libraries(ltc_ratio_estimator
  PRIVATE
    ltc_ratio_estimator_ptx
    owl::owl
    imgui::imgui
    ${GLFW3_LIBRARY}
    owl_viewer
)