#pragma once

#include <cmath>
#include <utility>
#include <ocs2_core/initialization/Initializer.h>
#include "tron2_ocs2/Types.h"

namespace tron2_ocs2 {

// Seed the cold policy and newly extended horizon with a dynamically consistent
// trajectory. DefaultInitializer holds x and sets u=0, violating arm-only base
// equality and ignoring measured arm velocity and base relaxation dynamics.
class Tron2Initializer final : public ocs2::Initializer {
 public:
  Tron2Initializer(ModelSettings settings, Eigen::Vector3d command)
      : settings_(settings), command_(std::move(command)) {}
  Tron2Initializer* clone() const override { return new Tron2Initializer(*this); }
  void compute(ocs2::scalar_t time, const ocs2::vector_t& state,
      ocs2::scalar_t nextTime, ocs2::vector_t& input, ocs2::vector_t& nextState) override {
    const double dt = nextTime - time;
    input = ocs2::vector_t::Zero(kInputDim);
    input.head<3>() = command_;
    nextState = state;
    const double halfAngle = command_(2) * dt / 2.0;
    const double sinc = std::abs(halfAngle) < 1e-8 ?
        1.0 - halfAngle * halfAngle / 6.0 : std::sin(halfAngle) / halfAngle;
    const double yaw = state(3) + halfAngle;
    nextState(0) += dt * sinc * (std::cos(yaw) * command_(0) - std::sin(yaw) * command_(1));
    nextState(1) += dt * sinc * (std::sin(yaw) * command_(0) + std::cos(yaw) * command_(1));
    nextState(2) = settings_.desiredBaseHeight +
        (state(2) - settings_.desiredBaseHeight) * std::exp(-settings_.heightResponseGain * dt);
    nextState(3) += command_(2) * dt;
    nextState(4) *= std::exp(-settings_.pitchResponseGain * dt);
    nextState(5) *= std::exp(-settings_.rollResponseGain * dt);
    nextState.segment(kArmPositionIndex, kArmDof) += dt * state.segment(kArmVelocityIndex, kArmDof);
  }
 private:
  ModelSettings settings_;
  Eigen::Vector3d command_;
};
}  // namespace tron2_ocs2
