#include <gtest/gtest.h>
#include "tron2_ocs2/ArmStateLimitCost.h"

TEST(ArmStateLimitCost, PenalizesTerminalViolationsAndDerivativesPointIntoBounds) {
  const ocs2::vector_t lower = ocs2::vector_t::Constant(tron2_ocs2::kArmDof, -1.0);
  const ocs2::vector_t upper = -lower;
  auto cost = tron2_ocs2::makeArmStateLimitCost(lower, upper, 5.0, 0.1, 0.001);
  ocs2::TargetTrajectories target;
  ocs2::PreComputation preComp;
  const ocs2::vector_t nominal = ocs2::vector_t::Zero(tron2_ocs2::kStateDim);
  const double reference = cost->getValue(3.0, nominal, target, preComp);
  for (std::size_t joint = 0; joint < tron2_ocs2::kArmDof; ++joint) {
    for (int sign : {-1, 1}) {
      for (const auto index : {tron2_ocs2::kArmPositionIndex + joint,
                              tron2_ocs2::kArmVelocityIndex + joint}) {
        auto x = nominal;
        x(index) = sign * (index < tron2_ocs2::kArmVelocityIndex ? 1.2 : 5.2);
        const auto approximation = cost->getQuadraticApproximation(3.0, x, target, preComp);
        EXPECT_GT(approximation.f, reference);
        EXPECT_GT(sign * approximation.dfdx(index), 0.0);
        EXPECT_GT(approximation.dfdxx(index, index), 0.0);
        const double step = 1e-6;
        auto plus = x, minus = x;
        plus(index) += step;
        minus(index) -= step;
        const double numerical = (cost->getValue(3.0, plus, target, preComp) -
                                  cost->getValue(3.0, minus, target, preComp)) / (2 * step);
        EXPECT_NEAR(approximation.dfdx(index), numerical, std::abs(numerical) * 1e-6);
        std::unique_ptr<ocs2::StateCost> clone(cost->clone());
        EXPECT_DOUBLE_EQ(clone->getValue(3.0, x, target, preComp), approximation.f);
      }
    }
  }
}

TEST(ArmStateLimitCost, RejectsInvalidBoundsAndPenalty) {
  const ocs2::vector_t lower = ocs2::vector_t::Constant(tron2_ocs2::kArmDof, -1.0);
  const ocs2::vector_t upper = -lower;
  EXPECT_THROW(tron2_ocs2::makeArmStateLimitCost(upper, lower, 5, 0.1, 0.001), std::invalid_argument);
  EXPECT_THROW(tron2_ocs2::makeArmStateLimitCost(lower, upper, 0, 0.1, 0.001), std::invalid_argument);
  EXPECT_THROW(tron2_ocs2::makeArmStateLimitCost(lower, upper, 5, -0.1, 0.001), std::invalid_argument);
}
