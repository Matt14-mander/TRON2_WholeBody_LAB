#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <utility>
#include <ocs2_core/cost/StateCost.h>

namespace tron2_ocs2 {
struct CostCallTimer {
  std::atomic<std::uint64_t> calls{0}, nanoseconds{0};
  void reset() { calls.store(0); nanoseconds.store(0); }
  double milliseconds() const { return nanoseconds.load() / 1e6; }
};
struct CostTimings {
  CostCallTimer value, quadratic;
  void reset() { value.reset(); quadratic.reset(); }
};

// Cloned cost objects share counters, while retaining independent underlying
// Pinocchio/AD data. Times sum work across workers; they are not wall times.
class ProfiledStateCost final : public ocs2::StateCost {
 public:
  ProfiledStateCost(std::unique_ptr<ocs2::StateCost> cost, std::shared_ptr<CostTimings> timings)
      : cost_(std::move(cost)), timings_(std::move(timings)) {}
  ProfiledStateCost* clone() const override {
    return new ProfiledStateCost(std::unique_ptr<ocs2::StateCost>(cost_->clone()), timings_);
  }
  bool isActive(ocs2::scalar_t time) const override { return cost_->isActive(time); }
  ocs2::scalar_t getValue(ocs2::scalar_t t, const ocs2::vector_t& x,
      const ocs2::TargetTrajectories& target, const ocs2::PreComputation& preComp) const override {
    Scope timer(timings_->value);
    return cost_->getValue(t, x, target, preComp);
  }
  ocs2::ScalarFunctionQuadraticApproximation getQuadraticApproximation(ocs2::scalar_t t,
      const ocs2::vector_t& x, const ocs2::TargetTrajectories& target,
      const ocs2::PreComputation& preComp) const override {
    Scope timer(timings_->quadratic);
    return cost_->getQuadraticApproximation(t, x, target, preComp);
  }
 private:
  class Scope {
   public:
    explicit Scope(CostCallTimer& timer) : timer_(timer), started_(std::chrono::steady_clock::now()) {}
    ~Scope() {
      const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now() - started_).count();
      timer_.nanoseconds.fetch_add(ns, std::memory_order_relaxed);
      timer_.calls.fetch_add(1, std::memory_order_relaxed);
    }
   private:
    CostCallTimer& timer_;
    std::chrono::steady_clock::time_point started_;
  };
  std::unique_ptr<ocs2::StateCost> cost_;
  std::shared_ptr<CostTimings> timings_;
};
}  // namespace tron2_ocs2
