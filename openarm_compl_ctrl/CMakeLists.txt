cmake_minimum_required(VERSION 3.20)
project(openarm_compl_ctrl)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 23)

# Dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(rosidl_default_runtime REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(bdd_ros2_interfaces REQUIRED)
find_package(OpenArmCAN REQUIRED)
find_package(openarm_description REQUIRED)
find_package(kdl_parser REQUIRED)
find_package(orocos_kdl REQUIRED)

include_directories(include)

# ------------------------
# Messages
# ------------------------
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/PIDDebug.msg"
  "msg/CartesianPIDDebug.msg"
  DEPENDENCIES std_msgs
)

ament_export_dependencies(rosidl_default_runtime)

# ------------------------
# Library
# ------------------------
add_library(openarm_comm
  src/openarm_comm.cpp
)

target_include_directories(openarm_comm PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include/${PROJECT_NAME}>
)

ament_target_dependencies(openarm_comm
  OpenArmCAN
)

install(TARGETS openarm_comm
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

# ------------------------
# Executables
# ------------------------
file(GLOB_RECURSE NODE_SOURCES
  "src/*_node.cpp"
  "src/*_test.cpp"
)

foreach(SRC_FILE ${NODE_SOURCES})
  get_filename_component(EXE_NAME ${SRC_FILE} NAME_WE)

  add_executable(${EXE_NAME} ${SRC_FILE})

  target_include_directories(${EXE_NAME} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include/${PROJECT_NAME}>
  )

  ament_target_dependencies(${EXE_NAME}
    rclcpp
    ament_index_cpp
    std_msgs
    sensor_msgs
    geometry_msgs
    bdd_ros2_interfaces
    openarm_description
    kdl_parser
    orocos_kdl
    rosidl_default_runtime
  )

  target_link_libraries(${EXE_NAME}
    OpenArmCAN::openarm_can
    openarm_comm
  )

  # Link generated message typesupport automatically
  rosidl_get_typesupport_target(cpp_typesupport_target
    ${PROJECT_NAME} "rosidl_typesupport_cpp"
  )
  target_link_libraries(${EXE_NAME} "${cpp_typesupport_target}")

  install(TARGETS ${EXE_NAME}
    RUNTIME DESTINATION lib/${PROJECT_NAME}
  )
endforeach()

# ------------------------
# Install URDF
# ------------------------
install(
  DIRECTORY urdf/
  DESTINATION share/${PROJECT_NAME}/urdf
)

ament_package()

