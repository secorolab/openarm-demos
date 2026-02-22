cmake_minimum_required(VERSION 3.20)
project(openarm_compl_ctrl)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 23)

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)

find_package(bdd_ros2_interfaces REQUIRED)

find_package(OpenArmCAN REQUIRED)
find_package(openarm_description REQUIRED)

find_package(kdl_parser REQUIRED)
find_package(orocos_kdl REQUIRED)

include_directories(
  include
)


rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/PIDDebug.msg"
)

# Build openarm_comm library
add_library(openarm_comm
  src/openarm_comm.cpp)
target_include_directories(openarm_comm PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include/${PROJECT_NAME}>)
ament_target_dependencies(openarm_comm
  OpenArmCAN
)

# Node executable
# add_executable(dual_arm_node src/dual_arm_node.cpp)
# target_include_directories(dual_arm_node PUBLIC
#   $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
#   $<INSTALL_INTERFACE:include/${PROJECT_NAME}>)
# ament_target_dependencies(dual_arm_node
#   rclcpp
#   ament_index_cpp
#   std_msgs
#   sensor_msgs
#   geometry_msgs
#   bdd_ros2_interfaces
#   kdl_parser
#   orocos_kdl
#   openarm_description
# )
# target_link_libraries(openarm_comm OpenArmCAN::openarm_can)

add_executable(cart_movement_node src/cart_movement_node.cpp)
target_include_directories(cart_movement_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include/${PROJECT_NAME}>)
ament_target_dependencies(cart_movement_node
  rclcpp
  ament_index_cpp
  std_msgs
  sensor_msgs
  geometry_msgs
  bdd_ros2_interfaces
  kdl_parser
  orocos_kdl
  openarm_description
)
target_link_libraries(cart_movement_node 
  OpenArmCAN::openarm_can
  openarm_comm
)

add_executable(achd_test src/achd_test.cpp)
target_include_directories(achd_test PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include/${PROJECT_NAME}>)
ament_target_dependencies(achd_test
  rclcpp
  ament_index_cpp
  std_msgs
  sensor_msgs
  geometry_msgs
  bdd_ros2_interfaces
  kdl_parser
  orocos_kdl
  openarm_description
)
target_link_libraries(achd_test 
  OpenArmCAN::openarm_can
  openarm_comm
)

add_executable(achd_ns_test src/achd_ns_test.cpp)
target_include_directories(achd_test PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include/${PROJECT_NAME}>)
ament_target_dependencies(achd_ns_test
  rclcpp
  ament_index_cpp
  std_msgs
  sensor_msgs
  geometry_msgs
  bdd_ros2_interfaces
  kdl_parser
  orocos_kdl
  openarm_description
)
target_link_libraries(achd_ns_test 
  OpenArmCAN::openarm_can
  openarm_comm
)

install(TARGETS openarm_comm
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin)

# Link against the generated message typesupport
rosidl_get_typesupport_target(cpp_typesupport_target ${PROJECT_NAME} "rosidl_typesupport_cpp")
target_link_libraries(cart_movement_node "${cpp_typesupport_target}")

install(
  DIRECTORY urdf/
  DESTINATION share/${PROJECT_NAME}/urdf
)

install(TARGETS cart_movement_node achd_test achd_ns_test
  DESTINATION lib/${PROJECT_NAME})

ament_package()
