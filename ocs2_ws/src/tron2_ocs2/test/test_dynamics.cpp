#include <gtest/gtest.h>

#include <ocs2_core/PreComputation.h>

#include "tron2_ocs2/Tron2Dynamics.h"

TEST(Tron2Dynamics, PaperModelFlowMap) {
  tron2_ocs2::ModelSettings settings;
  settings.recompileLibraries = true;
  settings.verbose = false;
  tron2_ocs2::Tron2Dynamics dynamics(settings, "/tmp/tron2_ocs2_test", true, false);
  ocs2::vector_t x = ocs2::vector_t::Zero(tron2_ocs2::kStateDim);
  ocs2::vector_t u = ocs2::vector_t::Zero(tron2_ocs2::kInputDim);
  x(2) = settings.desiredBaseHeight + 0.1;
  x(3) = 1.5707963267948966;
  x(4) = 0.2;
  x(5) = -0.3;
  x.segment<6>(tron2_ocs2::kArmVelocityIndex).setConstant(0.4);
  u(0) = 1.0;
  u(2) = 0.5;
  u.tail<6>().setConstant(2.0);
  const auto dx = dynamics.computeFlowMap(0.0, x, u, ocs2::PreComputation{});
  EXPECT_NEAR(dx(0), 0.0, 1e-10);
  EXPECT_NEAR(dx(1), 1.0, 1e-10);
  EXPECT_NEAR(dx(2), -0.5, 1e-10);
  EXPECT_NEAR(dx(3), 0.5, 1e-10);
  EXPECT_NEAR(dx(4), -1.0, 1e-10);
  EXPECT_NEAR(dx(5), 1.5, 1e-10);
  EXPECT_TRUE(dx.segment<6>(6).isApprox(Eigen::VectorXd::Constant(6, 0.4)));
  EXPECT_TRUE(dx.tail<6>().isApprox(Eigen::VectorXd::Constant(6, 2.0)));
}
