#pragma once

#include <array>
#include <cstddef>
#include <stdexcept>
#include <string>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <ocs2_core/Types.h>

namespace tron2_ocs2 {

constexpr std::size_t kArmDof = 6;
constexpr std::size_t kStateDim = 18;
constexpr std::size_t kInputDim = 9;
constexpr std::size_t kBasePoseDim = 6;
constexpr std::size_t kArmPositionIndex = 6;
constexpr std::size_t kArmVelocityIndex = 12;
constexpr std::array<double, 5> kWrenchPredictionTimes{0.0, 0.2, 0.4, 0.6, 0.8};

// State: [p_WB xyz, yaw, pitch, roll, q_arm, dq_arm].
// Input: [v_Bx, v_By, yaw_rate, ddq_arm].
struct Observation {
  double time = 0.0;
  Eigen::Vector3d basePositionWorld = Eigen::Vector3d::Zero();
  Eigen::Quaterniond baseOrientationWorld = Eigen::Quaterniond::Identity();
  Eigen::Matrix<double, 6, 1> baseTwistBody = Eigen::Matrix<double, 6, 1>::Zero();
  Eigen::Matrix<double, 6, 1> armPosition = Eigen::Matrix<double, 6, 1>::Zero();
  Eigen::Matrix<double, 6, 1> armVelocity = Eigen::Matrix<double, 6, 1>::Zero();
};

struct EndEffectorTarget {
  Eigen::Vector3d positionWorld = Eigen::Vector3d::Zero();
  Eigen::Quaterniond orientationWorld = Eigen::Quaterniond::Identity();
};

struct Solution {
  double time = 0.0;
  bool valid = false;
  Eigen::Matrix<double, 6, 1> armPosition = Eigen::Matrix<double, 6, 1>::Zero();
  Eigen::Matrix<double, 6, 1> armVelocity = Eigen::Matrix<double, 6, 1>::Zero();
  Eigen::Matrix<double, 6, 1> armEffort = Eigen::Matrix<double, 6, 1>::Zero();
  Eigen::Vector3d baseVelocityCommand = Eigen::Vector3d::Zero();
  Eigen::Matrix<double, 5, 6, Eigen::RowMajor> baseWrenchPrediction =
      Eigen::Matrix<double, 5, 6, Eigen::RowMajor>::Zero();
};

struct ModelSettings {
  double desiredBaseHeight = 0.74;
  double heightResponseGain = 5.0;
  double pitchResponseGain = 5.0;
  double rollResponseGain = 5.0;
  double commandLeadTime = 0.02;
  bool recompileLibraries = false;
  bool verbose = false;
};

inline void requireSize(const ocs2::vector_t& value, std::size_t expected, const char* name) {
  if (static_cast<std::size_t>(value.size()) != expected) {
    throw std::invalid_argument(std::string(name) + " has size " + std::to_string(value.size()) +
                                ", expected " + std::to_string(expected));
  }
}

}  // namespace tron2_ocs2
