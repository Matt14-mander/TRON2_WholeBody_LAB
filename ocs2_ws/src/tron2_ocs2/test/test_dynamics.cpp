#include <gtest/gtest.h>

#include <ocs2_core/PreComputation.h>

#include "tron2_ocs2/Tron2Dynamics.h"
#include "tron2_ocs2/Tron2Initializer.h"

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

TEST(Tron2Dynamics, AnalyticMatchesAdAndFiniteDifferences) {
  tron2_ocs2::ModelSettings settings;
  settings.heightResponseGain = 3.2;
  settings.pitchResponseGain = 4.3;
  settings.rollResponseGain = 6.1;
  tron2_ocs2::Tron2Dynamics ad(settings, "/tmp/tron2_ocs2_test", true, false);
  tron2_ocs2::AnalyticTron2Dynamics analytic(settings);
  const ocs2::PreComputation preComp;
  constexpr double eps = 1e-6;
  for (double yaw : {-2.8, -0.3, 0.0, 1.5707963267948966, 2.7}) {
    ocs2::vector_t x = ocs2::vector_t::LinSpaced(tron2_ocs2::kStateDim, -0.5, 0.8);
    ocs2::vector_t u = ocs2::vector_t::LinSpaced(tron2_ocs2::kInputDim, -0.4, 1.3);
    x(3) = yaw;
    const auto linear = analytic.linearApproximation(0.4, x, u, preComp);
    const auto generated = ad.linearApproximation(0.4, x, u, preComp);
    EXPECT_TRUE(linear.f.isApprox(generated.f, 1e-10));
    EXPECT_TRUE(linear.dfdx.isApprox(generated.dfdx, 1e-10));
    EXPECT_TRUE(linear.dfdu.isApprox(generated.dfdu, 1e-10));
    for (int i = 0; i < x.size(); ++i) {
      auto plus = x, minus = x;
      plus(i) += eps; minus(i) -= eps;
      const ocs2::vector_t derivative = (analytic.computeFlowMap(0.4, plus, u, preComp) -
          analytic.computeFlowMap(0.4, minus, u, preComp)) / (2 * eps);
      EXPECT_TRUE(linear.dfdx.col(i).isApprox(derivative, 1e-8));
    }
    for (int i = 0; i < u.size(); ++i) {
      auto plus = u, minus = u;
      plus(i) += eps; minus(i) -= eps;
      const ocs2::vector_t derivative = (analytic.computeFlowMap(0.4, x, plus, preComp) -
          analytic.computeFlowMap(0.4, x, minus, preComp)) / (2 * eps);
      EXPECT_TRUE(linear.dfdu.col(i).isApprox(derivative, 1e-8));
    }
    std::unique_ptr<tron2_ocs2::AnalyticTron2Dynamics> copy(analytic.clone());
    EXPECT_TRUE(copy->computeFlowMap(0.4, x, u, preComp).isApprox(linear.f));
  }
}

TEST(Tron2Initializer, ConsistentConstantInputAndZeroInterval) {
  tron2_ocs2::ModelSettings settings;
  tron2_ocs2::AnalyticTron2Dynamics dynamics(settings);
  const ocs2::PreComputation preComp;
  for (double yawRate : {0.0, 1e-10, 0.7, -0.9}) {
    tron2_ocs2::Tron2Initializer init(settings, Eigen::Vector3d(-0.29, -0.08, yawRate));
    auto x = ocs2::vector_t::LinSpaced(tron2_ocs2::kStateDim, -0.3, 0.7);
    ocs2::vector_t u, next;
    init.compute(1.0, x, 1.0, u, next);
    EXPECT_TRUE(next.isApprox(x));
    EXPECT_TRUE(u.head<3>().isApprox(Eigen::Vector3d(-0.29, -0.08, yawRate)));
    EXPECT_TRUE(u.tail<6>().isZero());
    // Compare exact initialization against an independently integrated model.
    init.compute(1.0, x, 1.1, u, next);
    ocs2::vector_t integrated = x;
    constexpr double dt = 0.0001;
    for (int i = 0; i < 1000; ++i) {
      const auto k1 = dynamics.computeFlowMap(0.0, integrated, u, preComp);
      const auto k2 = dynamics.computeFlowMap(0.0, integrated + dt / 2 * k1, u, preComp);
      const auto k3 = dynamics.computeFlowMap(0.0, integrated + dt / 2 * k2, u, preComp);
      const auto k4 = dynamics.computeFlowMap(0.0, integrated + dt * k3, u, preComp);
      integrated += dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4);
    }
    EXPECT_TRUE(next.isApprox(integrated, 1e-9));
  }
}
