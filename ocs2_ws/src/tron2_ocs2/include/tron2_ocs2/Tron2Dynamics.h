#pragma once

#include <ocs2_core/dynamics/SystemDynamicsBaseAD.h>
#include <ocs2_core/dynamics/SystemDynamicsBase.h>

#include "tron2_ocs2/Types.h"

namespace tron2_ocs2 {

class Tron2Dynamics final : public ocs2::SystemDynamicsBaseAD {
 public:
  Tron2Dynamics(ModelSettings settings, const std::string& libraryFolder,
                bool recompileLibraries, bool verbose);
  Tron2Dynamics* clone() const override { return new Tron2Dynamics(*this); }

 protected:
  Tron2Dynamics(const Tron2Dynamics& rhs) = default;
  ocs2::ad_vector_t systemFlowMap(ocs2::ad_scalar_t time, const ocs2::ad_vector_t& state,
                                  const ocs2::ad_vector_t& input,
                                  const ocs2::ad_vector_t& parameters) const override;

 private:
  ModelSettings settings_;
};

// Same paper model, with exact sparse derivatives instead of generated AD calls.
// Keep the AD implementation available for numerical comparison and rollback.
class AnalyticTron2Dynamics final : public ocs2::SystemDynamicsBase {
 public:
  explicit AnalyticTron2Dynamics(ModelSettings settings) : settings_(settings) {}
  AnalyticTron2Dynamics* clone() const override { return new AnalyticTron2Dynamics(*this); }
  using ocs2::SystemDynamicsBase::computeFlowMap;
  using ocs2::SystemDynamicsBase::linearApproximation;
  ocs2::vector_t computeFlowMap(ocs2::scalar_t time, const ocs2::vector_t& state,
      const ocs2::vector_t& input, const ocs2::PreComputation&) override;
  ocs2::VectorFunctionLinearApproximation linearApproximation(ocs2::scalar_t time,
      const ocs2::vector_t& state, const ocs2::vector_t& input,
      const ocs2::PreComputation&) override;
 private:
  ModelSettings settings_;
};

}  // namespace tron2_ocs2
