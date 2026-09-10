#pragma once

#include <string>

#include <pinocchio/multibody/data.hpp>
#include <pinocchio/multibody/model.hpp>
#include <ocs2_oc/oc_data/PrimalSolution.h>

#include "tron2_ocs2/Tron2PinocchioMapping.h"
#include "tron2_ocs2/Types.h"

namespace tron2_ocs2 {

class WrenchEstimator {
 public:
  WrenchEstimator(pinocchio::Model model, std::string baseFrame, std::string armRootJoint,
                  ModelSettings settings);

  Eigen::Matrix<double, 6, 1> armOnBaseWrench(const ocs2::vector_t& state,
                                               const ocs2::vector_t& input);
  Eigen::Matrix<double, 6, 1> armEffort(const ocs2::vector_t& state,
                                        const ocs2::vector_t& input);
  Eigen::Matrix<double, 5, 6, Eigen::RowMajor> predict(
      double initialTime, const ocs2::PrimalSolution& trajectory);

 private:
  void runRnea(const ocs2::vector_t& state, const ocs2::vector_t& input);
  std::pair<ocs2::vector_t, ocs2::vector_t> sample(
      double time, const ocs2::PrimalSolution& trajectory) const;

  pinocchio::Model model_;
  pinocchio::Data data_;
  pinocchio::FrameIndex baseFrameId_;
  pinocchio::JointIndex armRootJointId_;
  ModelSettings settings_;
  Tron2PinocchioMappingD mapping_;
};

}  // namespace tron2_ocs2
