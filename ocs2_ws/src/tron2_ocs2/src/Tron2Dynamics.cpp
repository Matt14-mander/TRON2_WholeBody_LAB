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

ocs2::vector_t AnalyticTron2Dynamics::computeFlowMap(ocs2::scalar_t,
    const ocs2::vector_t& x, const ocs2::vector_t& u, const ocs2::PreComputation&) {
  ocs2::vector_t dx = ocs2::vector_t::Zero(kStateDim);
  const double c = std::cos(x(3)), s = std::sin(x(3));
  dx(0) = c * u(0) - s * u(1);
  dx(1) = s * u(0) + c * u(1);
  dx(2) = -settings_.heightResponseGain * (x(2) - settings_.desiredBaseHeight);
  dx(3) = u(2);
  dx(4) = -settings_.pitchResponseGain * x(4);
  dx(5) = -settings_.rollResponseGain * x(5);
  dx.segment(kArmPositionIndex, kArmDof) = x.segment(kArmVelocityIndex, kArmDof);
  dx.segment(kArmVelocityIndex, kArmDof) = u.tail(kArmDof);
  return dx;
}

ocs2::VectorFunctionLinearApproximation AnalyticTron2Dynamics::linearApproximation(
    ocs2::scalar_t time, const ocs2::vector_t& x, const ocs2::vector_t& u,
    const ocs2::PreComputation& preComp) {
  ocs2::VectorFunctionLinearApproximation out;
  out.f = computeFlowMap(time, x, u, preComp);
  out.dfdx = ocs2::matrix_t::Zero(kStateDim, kStateDim);
  out.dfdu = ocs2::matrix_t::Zero(kStateDim, kInputDim);
  const double c = std::cos(x(3)), s = std::sin(x(3));
  out.dfdx(0, 3) = -s * u(0) - c * u(1);
  out.dfdx(1, 3) = c * u(0) - s * u(1);
  out.dfdx(2, 2) = -settings_.heightResponseGain;
  out.dfdx(4, 4) = -settings_.pitchResponseGain;
  out.dfdx(5, 5) = -settings_.rollResponseGain;
  out.dfdx.block(kArmPositionIndex, kArmVelocityIndex, kArmDof, kArmDof).setIdentity();
  out.dfdu(0, 0) = c; out.dfdu(0, 1) = -s;
  out.dfdu(1, 0) = s; out.dfdu(1, 1) = c;
  out.dfdu(3, 2) = 1.0;
  out.dfdu.block(kArmVelocityIndex, 3, kArmDof, kArmDof).setIdentity();
  return out;
}

}  // namespace tron2_ocs2
