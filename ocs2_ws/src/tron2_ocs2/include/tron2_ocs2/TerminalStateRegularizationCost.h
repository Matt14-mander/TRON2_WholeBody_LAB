#pragma once

#include <ocs2_core/cost/StateCost.h>

#include "tron2_ocs2/Types.h"

namespace tron2_ocs2 {

/** Small terminal quadratic cost that makes the final state Hessian positive definite. */
class TerminalStateRegularizationCost final : public ocs2::StateCost {
 public:
  TerminalStateRegularizationCost(ocs2::vector_t nominalState, double weight);
  TerminalStateRegularizationCost* clone() const override {
    return new TerminalStateRegularizationCost(*this);
  }

  ocs2::scalar_t getValue(ocs2::scalar_t time, const ocs2::vector_t& state,
                          const ocs2::TargetTrajectories& target,
                          const ocs2::PreComputation& preComputation) const override;
  ocs2::ScalarFunctionQuadraticApproximation getQuadraticApproximation(
      ocs2::scalar_t time, const ocs2::vector_t& state,
      const ocs2::TargetTrajectories& target,
      const ocs2::PreComputation& preComputation) const override;

 private:
  ocs2::vector_t nominalState_;
  double weight_;
};

}  // namespace tron2_ocs2
