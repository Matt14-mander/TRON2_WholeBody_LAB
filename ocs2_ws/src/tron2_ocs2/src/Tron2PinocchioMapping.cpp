#include "tron2_ocs2/Tron2PinocchioMapping.h"

#include <cmath>

namespace tron2_ocs2 {

template <typename Scalar>
auto Tron2PinocchioMapping<Scalar>::getPinocchioJointPosition(const vector_t& state) const -> vector_t {
  return state.head(kBasePoseDim + kArmDof);
}

template <typename Scalar>
auto Tron2PinocchioMapping<Scalar>::getPinocchioJointVelocity(const vector_t& state,
                                                              const vector_t& input) const -> vector_t {
  vector_t velocity = vector_t::Zero(kBasePoseDim + kArmDof);
  const auto yaw = state(3);
  velocity(0) = cos(yaw) * input(0) - sin(yaw) * input(1);
  velocity(1) = sin(yaw) * input(0) + cos(yaw) * input(1);
  velocity(2) = -settings_.heightResponseGain * (state(2) - settings_.desiredBaseHeight);
  velocity(3) = input(2);
  velocity(4) = -settings_.pitchResponseGain * state(4);
  velocity(5) = -settings_.rollResponseGain * state(5);
  velocity.tail(kArmDof) = state.tail(kArmDof);
  return velocity;
}

template <typename Scalar>
auto Tron2PinocchioMapping<Scalar>::getOcs2Jacobian(const vector_t& state, const matrix_t& Jq,
                                                    const matrix_t& Jv) const
    -> std::pair<matrix_t, matrix_t> {
  matrix_t dfdx = matrix_t::Zero(Jq.rows(), kStateDim);
  matrix_t dfdu = matrix_t::Zero(Jq.rows(), kInputDim);
  dfdx.leftCols(kBasePoseDim + kArmDof) = Jq;

  const auto yaw = state(3);
  const auto bodyVx = Scalar(0);  // State Jacobians cannot depend on an unavailable input.
  const auto bodyVy = Scalar(0);
  dfdx.col(3) += Jv.col(0) * (-sin(yaw) * bodyVx - cos(yaw) * bodyVy) +
                   Jv.col(1) * ( cos(yaw) * bodyVx - sin(yaw) * bodyVy);
  dfdx.col(2) += Jv.col(2) * Scalar(-settings_.heightResponseGain);
  dfdx.col(4) += Jv.col(4) * Scalar(-settings_.pitchResponseGain);
  dfdx.col(5) += Jv.col(5) * Scalar(-settings_.rollResponseGain);
  dfdx.rightCols(kArmDof) += Jv.rightCols(kArmDof);

  dfdu.col(0) = Jv.col(0) * cos(yaw) + Jv.col(1) * sin(yaw);
  dfdu.col(1) = Jv.col(0) * -sin(yaw) + Jv.col(1) * cos(yaw);
  dfdu.col(2) = Jv.col(3);
  return {dfdx, dfdu};
}

template class Tron2PinocchioMapping<ocs2::scalar_t>;
template class Tron2PinocchioMapping<ocs2::ad_scalar_t>;

}  // namespace tron2_ocs2
