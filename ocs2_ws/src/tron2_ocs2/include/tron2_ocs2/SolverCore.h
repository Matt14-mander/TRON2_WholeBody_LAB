#pragma once

#include <memory>
#include <string>

#include <ocs2_core/initialization/DefaultInitializer.h>
#include <ocs2_ddp/GaussNewtonDDP_MPC.h>
#include <ocs2_oc/oc_problem/OptimalControlProblem.h>
#include <ocs2_oc/rollout/TimeTriggeredRollout.h>
#include <ocs2_oc/synchronized_module/ReferenceManager.h>
#include <ocs2_pinocchio_interface/PinocchioInterface.h>

#include "tron2_ocs2/Types.h"
#include "tron2_ocs2/WrenchEstimator.h"

namespace tron2_ocs2 {

class SolverCore {
 public:
  SolverCore(const std::string& taskFile, const std::string& urdfFile,
             const std::string& libraryFolder);
  Solution solve(const Observation& observation, const EndEffectorTarget& target);
  bool trySolve(const Observation& observation, const EndEffectorTarget& target,
                Solution& solution, std::string* errorMessage = nullptr) noexcept;
  void reset();

  const ocs2::OptimalControlProblem& problem() const { return problem_; }

 private:
  ocs2::vector_t observationToState(const Observation& observation) const;
  ocs2::TargetTrajectories makeTarget(double time, const EndEffectorTarget& target) const;
  std::unique_ptr<ocs2::StateInputCost> makeBoxConstraints(
      const ocs2::vector_t& nominalState, const std::string& taskFile) const;

  ModelSettings settings_;
  std::unique_ptr<ocs2::PinocchioInterface> pinocchioInterface_;
  ocs2::OptimalControlProblem problem_;
  std::shared_ptr<ocs2::ReferenceManager> referenceManager_;
  std::unique_ptr<ocs2::TimeTriggeredRollout> rollout_;
  std::unique_ptr<ocs2::DefaultInitializer> initializer_;
  std::unique_ptr<ocs2::GaussNewtonDDP_MPC> mpc_;
  std::unique_ptr<WrenchEstimator> wrenchEstimator_;
  ocs2::vector_t nominalState_;
  double horizon_ = 1.0;
};

}  // namespace tron2_ocs2
