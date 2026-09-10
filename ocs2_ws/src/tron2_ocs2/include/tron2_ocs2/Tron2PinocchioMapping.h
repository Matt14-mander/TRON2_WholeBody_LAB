#pragma once

#include <ocs2_core/automatic_differentiation/Types.h>
#include <ocs2_pinocchio_interface/PinocchioStateInputMapping.h>

#include "tron2_ocs2/Types.h"

namespace tron2_ocs2 {

template <typename Scalar>
class Tron2PinocchioMapping final : public ocs2::PinocchioStateInputMapping<Scalar> {
 public:
  using Base = ocs2::PinocchioStateInputMapping<Scalar>;
  using typename Base::matrix_t;
  using typename Base::vector_t;

  explicit Tron2PinocchioMapping(ModelSettings settings) : settings_(settings) {}
  Tron2PinocchioMapping* clone() const override { return new Tron2PinocchioMapping(*this); }

  vector_t getPinocchioJointPosition(const vector_t& state) const override;
  vector_t getPinocchioJointVelocity(const vector_t& state, const vector_t& input) const override;
  std::pair<matrix_t, matrix_t> getOcs2Jacobian(const vector_t& state, const matrix_t& Jq,
                                                const matrix_t& Jv) const override;

 private:
  ModelSettings settings_;
};

using Tron2PinocchioMappingD = Tron2PinocchioMapping<ocs2::scalar_t>;
using Tron2PinocchioMappingAD = Tron2PinocchioMapping<ocs2::ad_scalar_t>;

}  // namespace tron2_ocs2
