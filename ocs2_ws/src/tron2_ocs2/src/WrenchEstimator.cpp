#include "tron2_ocs2/WrenchEstimator.h"

#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/rnea.hpp>
#include <ocs2_core/misc/LinearInterpolation.h>

namespace tron2_ocs2 {

WrenchEstimator::WrenchEstimator(pinocchio::Model model, std::string baseFrame,
                                 std::string armRootJoint, ModelSettings settings)
    : model_(std::move(model)), data_(model_), settings_(settings), mapping_(settings) {
  if (!model_.existFrame(baseFrame)) {
    throw std::invalid_argument("Pinocchio frame not found: " + baseFrame);
  }
  if (!model_.existJointName(armRootJoint)) {
    throw std::invalid_argument("Pinocchio joint not found: " + armRootJoint);
  }
  baseFrameId_ = model_.getFrameId(baseFrame);
  armRootJointId_ = model_.getJointId(armRootJoint);
}

void WrenchEstimator::runRnea(const ocs2::vector_t& x, const ocs2::vector_t& u) {
  requireSize(x, kStateDim, "state");
  requireSize(u, kInputDim, "input");
  const ocs2::vector_t q = mapping_.getPinocchioJointPosition(x);
  const ocs2::vector_t v = mapping_.getPinocchioJointVelocity(x, u);
  ocs2::vector_t a = ocs2::vector_t::Zero(model_.nv);
  const double yaw = x(3);
  a(0) = (-std::sin(yaw) * u(0) - std::cos(yaw) * u(1)) * u(2);
  a(1) = ( std::cos(yaw) * u(0) - std::sin(yaw) * u(1)) * u(2);
  a(2) = settings_.heightResponseGain * settings_.heightResponseGain *
         (x(2) - settings_.desiredBaseHeight);
  a(4) = settings_.pitchResponseGain * settings_.pitchResponseGain * x(4);
  a(5) = settings_.rollResponseGain * settings_.rollResponseGain * x(5);
  a.tail(kArmDof) = u.tail(kArmDof);
  pinocchio::rnea(model_, data_, q, v, a);
  pinocchio::updateFramePlacements(model_, data_);
}

Eigen::Matrix<double, 6, 1> WrenchEstimator::armOnBaseWrench(const ocs2::vector_t& x,
                                                             const ocs2::vector_t& u) {
  runRnea(x, u);
  // data.f is the parent-on-child wrench. Negating it yields arm-on-parent.
  const pinocchio::Force armOnParentLocal = -data_.f[armRootJointId_];
  const pinocchio::Force armOnWorld = data_.oMi[armRootJointId_].act(armOnParentLocal);
  const pinocchio::Force armOnBase = data_.oMf[baseFrameId_].actInv(armOnWorld);
  Eigen::Matrix<double, 6, 1> out;
  out.head<3>() = armOnBase.linear();
  out.tail<3>() = armOnBase.angular();
  return out;
}

Eigen::Matrix<double, 6, 1> WrenchEstimator::armEffort(const ocs2::vector_t& x,
                                                       const ocs2::vector_t& u) {
  runRnea(x, u);
  return data_.tau.tail(kArmDof);
}

std::pair<ocs2::vector_t, ocs2::vector_t> WrenchEstimator::sample(
    double time, const ocs2::PrimalSolution& trajectory) const {
  if (trajectory.timeTrajectory_.empty() || trajectory.stateTrajectory_.empty() ||
      trajectory.inputTrajectory_.empty()) {
    throw std::runtime_error("Cannot sample an empty OCS2 solution.");
  }
  auto x = ocs2::LinearInterpolation::interpolate(
      time, trajectory.timeTrajectory_, trajectory.stateTrajectory_);
  auto u = ocs2::LinearInterpolation::interpolate(
      time, trajectory.timeTrajectory_, trajectory.inputTrajectory_);
  requireSize(x, kStateDim, "sampled state");
  requireSize(u, kInputDim, "sampled input");
  return {std::move(x), std::move(u)};
}

Eigen::Matrix<double, 5, 6, Eigen::RowMajor> WrenchEstimator::predict(
    double initialTime, const ocs2::PrimalSolution& trajectory) {
  Eigen::Matrix<double, 5, 6, Eigen::RowMajor> out;
  for (std::size_t i = 0; i < kWrenchPredictionTimes.size(); ++i) {
    const auto [x, u] = sample(initialTime + kWrenchPredictionTimes[i], trajectory);
    out.row(i) = armOnBaseWrench(x, u).transpose();
  }
  return out;
}

}  // namespace tron2_ocs2
