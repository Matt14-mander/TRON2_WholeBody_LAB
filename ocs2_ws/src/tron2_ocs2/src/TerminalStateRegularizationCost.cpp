#include "tron2_ocs2/TerminalStateRegularizationCost.h"

#include <cmath>
#include <stdexcept>
#include <utility>

namespace tron2_ocs2 {

TerminalStateRegularizationCost::TerminalStateRegularizationCost(
    ocs2::vector_t nominalState, double weight)
    : nominalState_(std::move(nominalState)), weight_(weight) {
  requireSize(nominalState_, kStateDim, "terminal nominalState");
  if (!std::isfinite(weight_) || weight_ <= 0.0) {
    throw std::invalid_argument("Terminal state regularization weight must be positive and finite.");
  }
}

ocs2::scalar_t TerminalStateRegularizationCost::getValue(
    ocs2::scalar_t, const ocs2::vector_t& state,
    const ocs2::TargetTrajectories&, const ocs2::PreComputation&) const {
  requireSize(state, kStateDim, "terminal state");
  return 0.5 * weight_ * (state - nominalState_).squaredNorm();
}

ocs2::ScalarFunctionQuadraticApproximation
TerminalStateRegularizationCost::getQuadraticApproximation(
    ocs2::scalar_t time, const ocs2::vector_t& state,
    const ocs2::TargetTrajectories& target,
    const ocs2::PreComputation& preComputation) const {
  ocs2::ScalarFunctionQuadraticApproximation out(kStateDim);
  const auto error = state - nominalState_;
  out.f = getValue(time, state, target, preComputation);
  out.dfdx = weight_ * error;
  out.dfdxx = weight_ * ocs2::matrix_t::Identity(kStateDim, kStateDim);
  return out;
}

}  // namespace tron2_ocs2
