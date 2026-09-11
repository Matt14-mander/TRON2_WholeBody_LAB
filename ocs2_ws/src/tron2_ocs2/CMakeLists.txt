cmake_minimum_required(VERSION 3.16)
project(tron2_ocs2 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(ament_cmake REQUIRED)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(Boost REQUIRED COMPONENTS system filesystem log_setup log)
find_package(ocs2_core REQUIRED)
find_package(ocs2_ddp REQUIRED)
find_package(ocs2_mpc REQUIRED)
find_package(ocs2_oc REQUIRED)
find_package(ocs2_pinocchio_interface REQUIRED)
find_package(ocs2_mobile_manipulator REQUIRED)
find_package(ocs2_self_collision REQUIRED)
find_package(pinocchio REQUIRED)
find_package(hpp-fcl REQUIRED)

add_library(${PROJECT_NAME}
  src/Tron2Dynamics.cpp
  src/Tron2PinocchioMapping.cpp
  src/NominalCost.cpp
  src/WrenchEstimator.cpp
  src/SolverCore.cpp
)
target_include_directories(${PROJECT_NAME} PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${Boost_INCLUDE_DIRS}
)
target_link_libraries(${PROJECT_NAME} PUBLIC Eigen3::Eigen Boost::system Boost::filesystem Boost::log_setup Boost::log pinocchio::pinocchio hpp-fcl::hpp-fcl)
ament_target_dependencies(${PROJECT_NAME} PUBLIC
  ocs2_core ocs2_ddp ocs2_mpc ocs2_oc ocs2_pinocchio_interface
  ocs2_mobile_manipulator ocs2_self_collision pinocchio
)
target_compile_options(${PROJECT_NAME} PUBLIC ${OCS2_PINOCCHIO_FLAGS})

add_executable(tron2_ocs2_smoke src/smoke_main.cpp)
target_link_libraries(tron2_ocs2_smoke ${PROJECT_NAME})
ament_target_dependencies(tron2_ocs2_smoke ocs2_oc ocs2_core)

install(TARGETS ${PROJECT_NAME} tron2_ocs2_smoke
  EXPORT export_${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)
install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY config launch DESTINATION share/${PROJECT_NAME})
ament_export_targets(export_${PROJECT_NAME} HAS_LIBRARY_TARGET)
ament_export_dependencies(
  Eigen3 ocs2_core ocs2_ddp ocs2_mpc ocs2_oc
  ocs2_pinocchio_interface ocs2_mobile_manipulator ocs2_self_collision pinocchio hpp-fcl Boost
)

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_dynamics test/test_dynamics.cpp)
  target_link_libraries(test_dynamics ${PROJECT_NAME})
endif()

ament_package()
