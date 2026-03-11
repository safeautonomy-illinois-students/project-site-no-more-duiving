# generated from ament/cmake/core/templates/nameConfig.cmake.in

# prevent multiple inclusion
if(_gem_gazebo_CONFIG_INCLUDED)
  # ensure to keep the found flag the same
  if(NOT DEFINED gem_gazebo_FOUND)
    # explicitly set it to FALSE, otherwise CMake will set it to TRUE
    set(gem_gazebo_FOUND FALSE)
  elseif(NOT gem_gazebo_FOUND)
    # use separate condition to avoid uninitialized variable warning
    set(gem_gazebo_FOUND FALSE)
  endif()
  return()
endif()
set(_gem_gazebo_CONFIG_INCLUDED TRUE)

# output package information
if(NOT gem_gazebo_FIND_QUIETLY)
  message(STATUS "Found gem_gazebo: 1.0.0 (${gem_gazebo_DIR})")
endif()

# warn when using a deprecated package
if(NOT "" STREQUAL "")
  set(_msg "Package 'gem_gazebo' is deprecated")
  # append custom deprecation text if available
  if(NOT "" STREQUAL "TRUE")
    set(_msg "${_msg} ()")
  endif()
  # optionally quiet the deprecation message
  if(NOT ${gem_gazebo_DEPRECATED_QUIET})
    message(DEPRECATION "${_msg}")
  endif()
endif()

# flag package as ament-based to distinguish it after being find_package()-ed
set(gem_gazebo_FOUND_AMENT_PACKAGE TRUE)

# include all config extra files
set(_extras "")
foreach(_extra ${_extras})
  include("${gem_gazebo_DIR}/${_extra}")
endforeach()
