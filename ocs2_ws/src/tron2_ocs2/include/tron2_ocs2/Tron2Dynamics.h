#pragma once

#include <ocs2_core/dynamics/SystemDynamicsBaseAD.h>

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

}  // namespace tron2_ocs2
