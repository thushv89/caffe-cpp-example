cmake_minimum_required( VERSION 2.8 )

# +-----------------------------------------------------------------------------
# | Basic settings
# +-----------------------------------------------------------------------------
project( "CafeExample" )

# Set output directories for libraries and executables
set( BASE_DIR ${CMAKE_SOURCE_DIR} )
set( CMAKE_LIBRARY_OUTPUT_DIRECTORY ${BASE_DIR}/lib )
set( CMAKE_RUNTIME_OUTPUT_DIRECTORY ${BASE_DIR}/bin )

# Cmake module search locations
list( APPEND CMAKE_MODULE_PATH ${BASE_DIR}/cmake )


# +-----------------------------------------------------------------------------
# | Library search and setup
# +-----------------------------------------------------------------------------

# Build shared libraries by default
set( BUILD_SHARED_LIBS True )

# find_package( OpenCV REQUIRED )
find_package( Caffe REQUIRED )


# Use local include and library directories
include_directories(
    ${BASE_DIR}/src
    ${Caffe_INCLUDE_DIRS}
    ${Caffe_LIBRARIES}


)
link_directories(
    ${BASE_DIR}/lib
)


# +-----------------------------------------------------------------------------
# | Compiler settings
# +-----------------------------------------------------------------------------

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    set( CMAKE_CXX_FLAGS "-std=c++11 -Wall")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    set( CMAKE_CXX_FLAGS "-std=c++11 -Wall" )
endif()


# +-----------------------------------------------------------------------------
# | Compile code
# +-----------------------------------------------------------------------------

add_subdirectory( src )
