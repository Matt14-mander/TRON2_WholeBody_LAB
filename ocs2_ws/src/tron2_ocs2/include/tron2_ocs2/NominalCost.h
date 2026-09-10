#pragma once

#include <ocs2_core/cost/StateInputCost.h>

#include "tron2_ocs2/Types.h"

namespace tron2_ocs2 {

class NominalCost final : public ocs2::StateInputCost {
 public:
  NominalCost(ocs2::vector_t nominalState, ocs2::vector_t stateWeights,
              ocs2::vector_t inputWeights);
  NominalCost* clone() const override { return new NominalCost(*this); }

  ocs2::scalar_t getValue(ocs2::scalar_t time, const ocs2::vector_t& state,
                          const ocs2::vector_t& input,
                          const ocs2::TargetTrajectories& target,
                          const ocs2::PreComputation& preComputation) const override;
  ocs2::ScalarFunctionQuadraticApproximation getQuadraticApproximation(
      ocs2::scalar_t time, const ocs2::vector_t& state, const ocs2::vector_t& input,
      const ocs2::TargetTrajectories& target,
      const ocs2::PreComputation& preComputation) const override;

 private:
  ocs2::vector_t nominalState_;
  ocs2::vector_t stateWeights_;
  ocs2::vector_t inputWeights_;
};

}  // namespace tron2_ocs2
