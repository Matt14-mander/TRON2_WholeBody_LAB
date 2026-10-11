#include <gtest/gtest.h>
#include <thread>
#include "tron2_ocs2/ProfiledStateCost.h"

namespace {
class QuadraticCost final : public ocs2::StateCost {
 public:
  QuadraticCost* clone() const override { return new QuadraticCost(*this); }
  bool isActive(double time) const override { return time >= 0; }
  double getValue(double, const ocs2::vector_t& x, const ocs2::TargetTrajectories&,
      const ocs2::PreComputation&) const override { return x.squaredNorm(); }
  ocs2::ScalarFunctionQuadraticApproximation getQuadraticApproximation(double,
      const ocs2::vector_t& x, const ocs2::TargetTrajectories&, const ocs2::PreComputation&) const override {
    auto out = ocs2::ScalarFunctionQuadraticApproximation::Zero(x.size(), 0);
    out.f = x.squaredNorm(); out.dfdx = 2 * x;
    out.dfdxx = 2 * ocs2::matrix_t::Identity(x.size(), x.size());
    return out;
  }
};
}

TEST(ProfiledStateCost, PreservesValuesAndSharesThreadSafeCountersAcrossClones) {
  auto timings = std::make_shared<tron2_ocs2::CostTimings>();
  tron2_ocs2::ProfiledStateCost cost(std::make_unique<QuadraticCost>(), timings);
  const auto x = ocs2::vector_t::Ones(3);
  ocs2::TargetTrajectories target;
  ocs2::PreComputation preComp;
  EXPECT_FALSE(cost.isActive(-1));
  EXPECT_EQ(cost.getValue(0, x, target, preComp), 3);
  auto approximation = cost.getQuadraticApproximation(0, x, target, preComp);
  EXPECT_EQ(approximation.f, 3);
  EXPECT_TRUE(approximation.dfdx.isApprox(2 * x));
  std::unique_ptr<ocs2::StateCost> first(cost.clone()), second(cost.clone());
  auto run = [&](ocs2::StateCost* clone) {
    for (int i = 0; i < 100; ++i) clone->getValue(0, x, target, preComp);
  };
  std::thread a(run, first.get()), b(run, second.get());
  a.join(); b.join();
  EXPECT_EQ(timings->value.calls.load(), 201);
  EXPECT_EQ(timings->quadratic.calls.load(), 1);
  timings->reset();
  EXPECT_EQ(timings->value.calls.load(), 0);
  EXPECT_EQ(timings->quadratic.nanoseconds.load(), 0);
}
