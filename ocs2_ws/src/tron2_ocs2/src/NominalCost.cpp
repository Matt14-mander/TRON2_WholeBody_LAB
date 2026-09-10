#include "tron2_ocs2/NominalCost.h"

#include <utility>

namespace tron2_ocs2 {

NominalCost::NominalCost(ocs2::vector_t nominalState, ocs2::vector_t stateWeights,
                         ocs2::vector_t inputWeights)
    : nominalState_(std::move(nominalState)),
      stateWeights_(std::move(stateWeights)), inputWeights_(std::move(inputWeights)) {
  requireSize(nominalState_, kStateDim, "nominalState");
  requireSize(stateWeights_, kStateDim, "stateWeights");
  requireSize(inputWeights_, kInputDim, "inputWeights");
}

ocs2::scalar_t NominalCost::getValue(ocs2::scalar_t, const ocs2::vector_t& x,
                                     const ocs2::vector_t& u,
                                     const ocs2::TargetTrajectories&,
                                     const ocs2::PreComputation&) const {
  const auto dx = x - nominalState_;
  return 0.5 * dx.dot(stateWeights_.cwiseProduct(dx)) +
         0.5 * u.dot(inputWeights_.cwiseProduct(u));
}

ocs2::ScalarFunctionQuadraticApproximation NominalCost::getQuadraticApproximation(
    ocs2::scalar_t time, const ocs2::vector_t& x, const ocs2::vector_t& u,
    const ocs2::TargetTrajectories& target, const ocs2::PreComputation& preComp) const {
  ocs2::ScalarFunctionQuadraticApproximation out(kStateDim, kInputDim);
  const auto dx = x - nominalState_;
  out.f = getValue(time, x, u, target, preComp);
  out.dfdx = stateWeights_.cwiseProduct(dx);
  out.dfdu = inputWeights_.cwiseProduct(u);
  out.dfdxx = stateWeights_.asDiagonal();
  out.dfduu = inputWeights_.asDiagonal();
  out.dfdux.setZero();
  return out;
}

}  // namespace tron2_ocs2
