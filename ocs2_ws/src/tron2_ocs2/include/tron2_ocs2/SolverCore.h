#pragma once

#include <memory>
#include <string>
#include <vector>

#include <ocs2_core/initialization/DefaultInitializer.h>
#include <ocs2_ddp/GaussNewtonDDP_MPC.h>
#include <ocs2_oc/oc_problem/OptimalControlProblem.h>
#include <ocs2_oc/oc_data/PrimalSolution.h>
#include <ocs2_oc/rollout/TimeTriggeredRollout.h>
#include <ocs2_oc/synchronized_module/ReferenceManager.h>
#include <ocs2_pinocchio_interface/PinocchioInterface.h>

#include "tron2_ocs2/Types.h"
#include "tron2_ocs2/WrenchEstimator.h"
#include "tron2_ocs2/ProfiledStateCost.h"

namespace tron2_ocs2 {

struct SolveProfile {
  double referenceMs = 0.0;
  double optimizationMs = 0.0;
  double extractionMs = 0.0;
  double validationSampleMs = 0.0;
  double cost = 0.0;
  double equalitySSE = 0.0;
  double minimumArmPositionMargin = 0.0;
  double initialArmPositionMargin = 0.0;
  std::size_t worstArmPositionIndex = 0;  // Zero-based index in the six-arm state.
  double worstArmPositionTime = 0.0;  // Absolute simulation time at a trajectory knot.
  double worstArmPositionValue = 0.0;
  ocs2::vector_t armLowerPositionLimits = ocs2::vector_t::Zero(kArmDof);
  ocs2::vector_t armUpperPositionLimits = ocs2::vector_t::Zero(kArmDof);
  double maximumArmSpeed = 0.0;
  double maximumArmAcceleration = 0.0;
  bool collisionChecked = false;
  double collisionValidationMs = 0.0;
  double initialCollisionMargin = 0.0;
  double minimumCollisionMargin = 0.0;  // Distance minus configured minimum distance.
  double worstCollisionTime = 0.0;
  std::size_t worstCollisionPairIndex = 0;
  std::size_t iterations = 0;
  std::size_t knots = 0;
  bool cold = true;
};

class SolverCore {
 public:
  SolverCore(const std::string& taskFile, const std::string& urdfFile,
             const std::string& libraryFolder);
  EndEffectorTarget currentEndEffectorTarget(const Observation& observation) const;
  ocs2::vector_t collisionMargins(const Observation& observation) const;
  double collisionMinimumDistance() const { return collisionMinimumDistance_; }
  Solution solve(const Observation& observation, const EndEffectorTarget& target);
  std::vector<Solution> solveTrajectory(const Observation& observation,
                                        const EndEffectorTarget& target,
                                        double samplePeriod = 0.02,
                                        double terminalTransitionDuration = 0.5);
  bool trySolve(const Observation& observation, const EndEffectorTarget& target,
                Solution& solution, std::string* errorMessage = nullptr) noexcept;
  void reset();

  const ocs2::OptimalControlProblem& problem() const { return problem_; }
  double initialBaseHeight() const { return settings_.initialBaseHeight; }
  const SolveProfile& lastSolveProfile() const { return profile_; }

 private:
  ocs2::vector_t observationToState(const Observation& observation) const;
  ocs2::TargetTrajectories makeTarget(double time, const ocs2::vector_t& initialState,
                                      const EndEffectorTarget& target) const;
  ocs2::PrimalSolution runMpc(const Observation& observation,
                              const EndEffectorTarget& target);
  void validateTrajectory(double initialTime, const ocs2::PrimalSolution& trajectory) const;
  Solution sampleSolution(double sampleTime, double outputTime,
                          const ocs2::PrimalSolution& trajectory);
  Solution terminalHoldSolution(double outputTime,
                                const ocs2::PrimalSolution& trajectory);
  std::unique_ptr<ocs2::StateInputCost> makeBoxConstraints(
      const ocs2::vector_t& nominalState, const std::string& taskFile) const;

  ModelSettings settings_;
  std::unique_ptr<ocs2::PinocchioInterface> pinocchioInterface_;
  ocs2::OptimalControlProblem problem_;
  std::shared_ptr<ocs2::ReferenceManager> referenceManager_;
  std::unique_ptr<ocs2::TimeTriggeredRollout> rollout_;
  std::unique_ptr<ocs2::Initializer> initializer_;
  std::unique_ptr<ocs2::GaussNewtonDDP_MPC> mpc_;
  std::unique_ptr<WrenchEstimator> wrenchEstimator_;
  std::unique_ptr<ocs2::StateConstraint> collisionDiagnosticConstraint_;
  double collisionMinimumDistance_ = 0.04;
  ocs2::vector_t nominalState_;
  double horizon_ = 1.0;
  SolveProfile profile_;
  bool havePreviousSolve_ = false;
  double previousSolveTime_ = 0.0;
  bool coldStart_ = false;
  bool profileInternal_ = false;
  std::shared_ptr<CostTimings> collisionTimings_ = std::make_shared<CostTimings>();
  std::shared_ptr<CostTimings> endEffectorTimings_ = std::make_shared<CostTimings>();
};

}  // namespace tron2_ocs2
