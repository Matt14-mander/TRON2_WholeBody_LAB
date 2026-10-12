#pragma once

#include <cmath>
#include <memory>
#include <stdexcept>
#include <utility>

#include <ocs2_core/constraint/LinearStateConstraint.h>
#include <ocs2_core/penalties/Penalties.h>
#include <ocs2_core/soft_constraint/StateSoftConstraint.h>

#include "tron2_ocs2/Types.h"

namespace tron2_ocs2 {

// The final state has no input. Apply the same position/velocity bounds and
// barrier as the running cost; acceleration remains a running input bound.
inline std::unique_ptr<ocs2::StateSoftConstraint> makeArmStateLimitCost(
    const ocs2::vector_t& lower, const ocs2::vector_t& upper,
    double velocityLimit, double mu, double delta) {
  if (lower.size() != kArmDof || upper.size() != kArmDof ||
      !lower.allFinite() || !upper.allFinite() || (lower.array() >= upper.array()).any() ||
      !std::isfinite(velocityLimit) || velocityLimit <= 0.0 ||
      !std::isfinite(mu) || mu <= 0.0 || !std::isfinite(delta) || delta <= 0.0) {
    throw std::invalid_argument("Invalid terminal arm position/velocity limits or penalty.");
  }
  ocs2::vector_t h = ocs2::vector_t::Zero(4 * kArmDof);
  ocs2::matrix_t F = ocs2::matrix_t::Zero(4 * kArmDof, kStateDim);
  for (std::size_t i = 0; i < kArmDof; ++i) {
    h(i) = -lower(i);
    F(i, kArmPositionIndex + i) = 1.0;
    h(kArmDof + i) = upper(i);
    F(kArmDof + i, kArmPositionIndex + i) = -1.0;
    h(2 * kArmDof + i) = velocityLimit;
    F(2 * kArmDof + i, kArmVelocityIndex + i) = 1.0;
    h(3 * kArmDof + i) = velocityLimit;
    F(3 * kArmDof + i, kArmVelocityIndex + i) = -1.0;
  }
  return std::make_unique<ocs2::StateSoftConstraint>(
      std::make_unique<ocs2::LinearStateConstraint>(std::move(h), std::move(F)),
      std::make_unique<ocs2::RelaxedBarrierPenalty>(
          ocs2::RelaxedBarrierPenalty::Config{mu, delta}));
}

}  // namespace tron2_ocs2
