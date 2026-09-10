#include "tron2_ocs2/Tron2Dynamics.h"

#include <cmath>

namespace tron2_ocs2 {

Tron2Dynamics::Tron2Dynamics(ModelSettings settings, const std::string& libraryFolder,
                             bool recompileLibraries, bool verbose)
    : settings_(settings) {
  initialize(kStateDim, kInputDim, "tron2_wholebody_dynamics", libraryFolder,
             recompileLibraries, verbose);
}

ocs2::ad_vector_t Tron2Dynamics::systemFlowMap(ocs2::ad_scalar_t, const ocs2::ad_vector_t& x,
                                               const ocs2::ad_vector_t& u,
                                               const ocs2::ad_vector_t&) const {
  ocs2::ad_vector_t dx = ocs2::ad_vector_t::Zero(kStateDim);
  const auto yaw = x(3);
  dx(0) = cos(yaw) * u(0) - sin(yaw) * u(1);
  dx(1) = sin(yaw) * u(0) + cos(yaw) * u(1);
  dx(2) = -settings_.heightResponseGain * (x(2) - settings_.desiredBaseHeight);
  dx(3) = u(2);
  dx(4) = -settings_.pitchResponseGain * x(4);
  dx(5) = -settings_.rollResponseGain * x(5);
  dx.segment(kArmPositionIndex, kArmDof) = x.segment(kArmVelocityIndex, kArmDof);
  dx.segment(kArmVelocityIndex, kArmDof) = u.tail(kArmDof);
  return dx;
}

}  // namespace tron2_ocs2
