#include "tron2_ocs2/SolverCore.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <cmath>
#include <utility>

#include <boost/property_tree/info_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#include <ocs2_core/misc/LinearInterpolation.h>
#include <ocs2_core/misc/LoadData.h>
#include <ocs2_core/penalties/Penalties.h>
#include <ocs2_core/soft_constraint/StateInputSoftBoxConstraint.h>
#include <ocs2_core/soft_constraint/StateSoftConstraint.h>
#include <ocs2_ddp/DDP_Settings.h>
#include <ocs2_mobile_manipulator/FactoryFunctions.h>
#include <ocs2_mobile_manipulator/constraint/EndEffectorConstraint.h>
#include <ocs2_mpc/MPC_Settings.h>
#include <ocs2_oc/rollout/RolloutSettings.h>
#include <ocs2_pinocchio_interface/PinocchioEndEffectorKinematicsCppAd.h>
#include <ocs2_self_collision/PinocchioGeometryInterface.h>
#include <ocs2_self_collision/SelfCollisionConstraintCppAd.h>

#include "tron2_ocs2/NominalCost.h"
#include "tron2_ocs2/Tron2Dynamics.h"
#include "tron2_ocs2/Tron2PinocchioMapping.h"

namespace tron2_ocs2 {
namespace {

constexpr char kBaseFrame[] = "base_Link";
constexpr char kEndEffectorFrame[] = "gripper_base_Link";
constexpr char kArmRootJoint[] = "arm1_Joint";

const std::vector<std::string> kFixedJoints{
    "proximal_pitch_L_Joint", "proximal_roll_L_Joint", "proximal_yaw_L_Joint",
    "knee_L_Joint", "ankle_pitch_L_Joint", "proximal_pitch_R_Joint",
    "proximal_roll_R_Joint", "proximal_yaw_R_Joint", "knee_R_Joint",
    "ankle_pitch_R_Joint", "gripper1_Joint", "gripper2_Joint"};

std::string makeMeshUrisPortable(const std::string& urdfFile,
                                 const std::string& libraryFolder) {
  std::ifstream stream(urdfFile);
  if (!stream) throw std::runtime_error("Cannot read URDF: " + urdfFile);
  std::string xml((std::istreambuf_iterator<char>(stream)), std::istreambuf_iterator<char>());
  const std::string oldPrefix = "package://bipedal_robot/meshes/";
  const auto meshDirectory =
      std::filesystem::absolute(std::filesystem::path(urdfFile).parent_path() / ".." / "meshes")
          .lexically_normal();
  const std::string newPrefix = "file://" + meshDirectory.generic_string() + "/";
  for (std::size_t offset = 0; (offset = xml.find(oldPrefix, offset)) != std::string::npos;) {
    xml.replace(offset, oldPrefix.size(), newPrefix);
    offset += newPrefix.size();
  }
  const auto output = std::filesystem::path(libraryFolder) / "tron2_ocs2_resolved.urdf";
  std::ofstream resolved(output);
  if (!resolved) throw std::runtime_error("Cannot write resolved URDF: " + output.string());
  resolved << xml;
  return output.string();
}

std::unique_ptr<ocs2::StateCost> makeEndEffectorCost(
    const ocs2::PinocchioInterface& interface, const std::shared_ptr<ocs2::ReferenceManager>& reference,
    const ModelSettings& settings, const std::string& libraryFolder, const std::string& modelName,
    double positionWeight, double orientationWeight) {
  Tron2PinocchioMappingAD mapping(settings);
  ocs2::PinocchioEndEffectorKinematicsCppAd kinematics(
      interface, mapping, {kEndEffectorFrame}, kStateDim, kInputDim, modelName,
      libraryFolder, settings.recompileLibraries, settings.verbose);
  auto constraint = std::make_unique<ocs2::mobile_manipulator::EndEffectorConstraint>(
      kinematics, *reference);
  std::vector<std::unique_ptr<ocs2::PenaltyBase>> penalties;
  penalties.reserve(6);
  for (int i = 0; i < 3; ++i) penalties.emplace_back(std::make_unique<ocs2::QuadraticPenalty>(positionWeight));
  for (int i = 0; i < 3; ++i) penalties.emplace_back(std::make_unique<ocs2::QuadraticPenalty>(orientationWeight));
  return std::make_unique<ocs2::StateSoftConstraint>(std::move(constraint), std::move(penalties));
}

}  // namespace

SolverCore::SolverCore(const std::string& taskFile, const std::string& urdfFile,
                       const std::string& libraryFolder) {
  if (!std::filesystem::exists(taskFile)) throw std::invalid_argument("Task file not found: " + taskFile);
  if (!std::filesystem::exists(urdfFile)) throw std::invalid_argument("URDF not found: " + urdfFile);
  std::filesystem::create_directories(libraryFolder);
  const std::string resolvedUrdf = makeMeshUrisPortable(urdfFile, libraryFolder);

  boost::property_tree::ptree pt;
  boost::property_tree::read_info(taskFile, pt);
  settings_.desiredBaseHeight = pt.get<double>("model.desiredBaseHeight", settings_.desiredBaseHeight);
  settings_.heightResponseGain = pt.get<double>("model.heightResponseGain", settings_.heightResponseGain);
  settings_.pitchResponseGain = pt.get<double>("model.pitchResponseGain", settings_.pitchResponseGain);
  settings_.rollResponseGain = pt.get<double>("model.rollResponseGain", settings_.rollResponseGain);
  settings_.commandLeadTime = pt.get<double>("model.commandLeadTime", settings_.commandLeadTime);
  settings_.recompileLibraries = pt.get<bool>("model.recompileLibraries", settings_.recompileLibraries);
  settings_.verbose = pt.get<bool>("model.verbose", settings_.verbose);
  if (settings_.desiredBaseHeight <= 0.0 || settings_.heightResponseGain <= 0.0 ||
      settings_.pitchResponseGain <= 0.0 || settings_.rollResponseGain <= 0.0 ||
      settings_.commandLeadTime < 0.0) {
    throw std::invalid_argument("Model height/gains must be positive and commandLeadTime non-negative.");
  }

  pinocchioInterface_ = std::make_unique<ocs2::PinocchioInterface>(
      ocs2::mobile_manipulator::createPinocchioInterface(
          resolvedUrdf, ocs2::mobile_manipulator::ManipulatorModelType::FloatingArmManipulator,
          kFixedJoints));
  const auto& pinModel = pinocchioInterface_->getModel();
  if (pinModel.nq != kBasePoseDim + kArmDof || pinModel.nv != kBasePoseDim + kArmDof) {
    throw std::runtime_error("Reduced Pinocchio model must have nq=nv=12; got nq=" +
                             std::to_string(pinModel.nq) + ", nv=" + std::to_string(pinModel.nv));
  }

  nominalState_ = ocs2::vector_t::Zero(kStateDim);
  nominalState_(2) = settings_.desiredBaseHeight;
  ocs2::vector_t nominalArmPosition = ocs2::vector_t::Zero(kArmDof);
  ocs2::loadData::loadEigenMatrix(taskFile, "nominalArmPosition", nominalArmPosition);
  nominalState_.segment(kArmPositionIndex, kArmDof) = nominalArmPosition;
  ocs2::vector_t stateWeights = ocs2::vector_t::Zero(kStateDim);
  ocs2::vector_t inputWeights = ocs2::vector_t::Zero(kInputDim);
  ocs2::loadData::loadEigenMatrix(taskFile, "cost.stateWeights", stateWeights);
  ocs2::loadData::loadEigenMatrix(taskFile, "cost.inputWeights", inputWeights);

  referenceManager_ = std::make_shared<ocs2::ReferenceManager>();
  problem_.costPtr->add("nominal", std::make_unique<NominalCost>(
                                      nominalState_, stateWeights, inputWeights));
  problem_.softConstraintPtr->add("limits", makeBoxConstraints(nominalState_, taskFile));
  const double eePositionWeight = pt.get<double>("cost.eePositionWeight", 100.0);
  const double eeOrientationWeight = pt.get<double>("cost.eeOrientationWeight", 20.0);
  const double terminalScale = pt.get<double>("cost.terminalScale", 5.0);
  problem_.stateSoftConstraintPtr->add(
      "end_effector", makeEndEffectorCost(*pinocchioInterface_, referenceManager_, settings_,
                                           libraryFolder, "tron2_ee_running",
                                           eePositionWeight, eeOrientationWeight));
  problem_.finalSoftConstraintPtr->add(
      "end_effector", makeEndEffectorCost(*pinocchioInterface_, referenceManager_, settings_,
                                           libraryFolder, "tron2_ee_terminal",
                                           terminalScale * eePositionWeight,
                                           terminalScale * eeOrientationWeight));
  if (pt.get<bool>("selfCollision.activate", true)) {
    const std::vector<std::pair<std::string, std::string>> collisionPairs{
        {"base_Link", "arm3_Link"}, {"base_Link", "arm4_Link"},
        {"base_Link", "arm5_Link"}, {"base_Link", "gripper_base_Link"},
        {"arm1_Link", "arm4_Link"}, {"arm1_Link", "arm5_Link"},
        {"arm2_Link", "arm5_Link"}, {"arm2_Link", "gripper_base_Link"}};
    ocs2::PinocchioGeometryInterface geometry(*pinocchioInterface_, collisionPairs);
    Tron2PinocchioMappingD mapping(settings_);
    auto constraint = std::make_unique<ocs2::SelfCollisionConstraintCppAd>(
        *pinocchioInterface_, mapping, std::move(geometry),
        pt.get<double>("selfCollision.minimumDistance", 0.04),
        "tron2_self_collision", libraryFolder, settings_.recompileLibraries,
        settings_.verbose);
    auto penalty = std::make_unique<ocs2::RelaxedBarrierPenalty>(
        ocs2::RelaxedBarrierPenalty::Config{
            pt.get<double>("selfCollision.mu", 1e-2),
            pt.get<double>("selfCollision.delta", 1e-3)});
    problem_.stateSoftConstraintPtr->add(
        "self_collision", std::make_unique<ocs2::StateSoftConstraint>(
                              std::move(constraint), std::move(penalty)));
  }

  problem_.dynamicsPtr = std::make_unique<Tron2Dynamics>(
      settings_, libraryFolder, settings_.recompileLibraries, settings_.verbose);
  const auto rolloutSettings = ocs2::rollout::loadSettings(taskFile, "rollout", settings_.verbose);
  rollout_ = std::make_unique<ocs2::TimeTriggeredRollout>(*problem_.dynamicsPtr, rolloutSettings);
  initializer_ = std::make_unique<ocs2::DefaultInitializer>(kInputDim);
  auto mpcSettings = ocs2::mpc::loadSettings(taskFile, "mpc", settings_.verbose);
  auto ddpSettings = ocs2::ddp::loadSettings(taskFile, "ddp", settings_.verbose);
  horizon_ = mpcSettings.timeHorizon_;
  if (horizon_ + 1e-9 < kWrenchPredictionTimes.back()) {
    throw std::invalid_argument("MPC horizon must be at least 0.8 s for wrench prediction.");
  }
  if (settings_.commandLeadTime > horizon_) {
    throw std::invalid_argument("commandLeadTime must not exceed the MPC horizon.");
  }
  mpc_ = std::make_unique<ocs2::GaussNewtonDDP_MPC>(
      mpcSettings, ddpSettings, *rollout_, problem_, *initializer_);
  mpc_->getSolverPtr()->setReferenceManager(referenceManager_);
  wrenchEstimator_ = std::make_unique<WrenchEstimator>(
      pinModel, kBaseFrame, kArmRootJoint, settings_);
}

std::unique_ptr<ocs2::StateInputCost> SolverCore::makeBoxConstraints(
    const ocs2::vector_t& nominalState, const std::string& taskFile) const {
  using Box = ocs2::StateInputSoftBoxConstraint::BoxConstraint;
  std::vector<Box> stateBoxes;
  std::vector<Box> inputBoxes;
  const auto& model = pinocchioInterface_->getModel();
  boost::property_tree::ptree pt;
  boost::property_tree::read_info(taskFile, pt);
  const double armVelocityLimit = pt.get<double>("limits.armVelocity", 5.0);
  const double armAccelerationLimit = pt.get<double>("limits.armAcceleration", 20.0);
  const double forwardVelocityLimit = pt.get<double>("limits.forwardVelocity", 1.5);
  const double lateralVelocityLimit = pt.get<double>("limits.lateralVelocity", 1.0);
  const double yawRateLimit = pt.get<double>("limits.yawRate", 2.0);
  if (armVelocityLimit <= 0.0 || armAccelerationLimit <= 0.0 ||
      forwardVelocityLimit <= 0.0 || lateralVelocityLimit <= 0.0 ||
      yawRateLimit <= 0.0) {
    throw std::invalid_argument("All velocity and acceleration limits must be positive.");
  }
  auto addBox = [](std::vector<Box>& boxes, std::size_t index, double lower, double upper) {
    Box box;
    box.index = index;
    box.lowerBound = lower;
    box.upperBound = upper;
    box.penaltyPtr = std::make_unique<ocs2::RelaxedBarrierPenalty>(
        ocs2::RelaxedBarrierPenalty::Config{1e-3, 1e-3});
    boxes.emplace_back(std::move(box));
  };
  for (std::size_t i = 0; i < kArmDof; ++i) {
    addBox(stateBoxes, kArmPositionIndex + i,
           model.lowerPositionLimit(model.nq - kArmDof + i),
           model.upperPositionLimit(model.nq - kArmDof + i));
    addBox(stateBoxes, kArmVelocityIndex + i, -armVelocityLimit, armVelocityLimit);
  }
  addBox(inputBoxes, 0, -forwardVelocityLimit, forwardVelocityLimit);
  addBox(inputBoxes, 1, -lateralVelocityLimit, lateralVelocityLimit);
  addBox(inputBoxes, 2, -yawRateLimit, yawRateLimit);
  for (std::size_t i = 0; i < kArmDof; ++i) {
    addBox(inputBoxes, 3 + i, -armAccelerationLimit, armAccelerationLimit);
  }
  auto limits = std::make_unique<ocs2::StateInputSoftBoxConstraint>(
      std::move(stateBoxes), std::move(inputBoxes));
  limits->initializeOffset(0.0, nominalState, ocs2::vector_t::Zero(kInputDim));
  return limits;
}

ocs2::vector_t SolverCore::observationToState(const Observation& observation) const {
  if (!std::isfinite(observation.time) || !observation.basePositionWorld.allFinite() ||
      !observation.baseOrientationWorld.coeffs().allFinite() ||
      observation.baseOrientationWorld.norm() < 1e-9 || !observation.baseTwistBody.allFinite() ||
      !observation.armPosition.allFinite() || !observation.armVelocity.allFinite()) {
    throw std::invalid_argument("Observation contains non-finite values.");
  }
  ocs2::vector_t x = ocs2::vector_t::Zero(kStateDim);
  x.head<3>() = observation.basePositionWorld;
  const Eigen::Vector3d zyx = observation.baseOrientationWorld.normalized()
                                  .toRotationMatrix().eulerAngles(2, 1, 0);
  x.segment<3>(3) = zyx;
  x.segment<6>(kArmPositionIndex) = observation.armPosition;
  x.segment<6>(kArmVelocityIndex) = observation.armVelocity;
  return x;
}

ocs2::TargetTrajectories SolverCore::makeTarget(double time,
                                                 const EndEffectorTarget& target) const {
  if (!target.positionWorld.allFinite() || !target.orientationWorld.coeffs().allFinite() ||
      target.orientationWorld.norm() < 1e-9) {
    throw std::invalid_argument("End-effector target contains non-finite values.");
  }
  Eigen::Quaterniond q = target.orientationWorld.normalized();
  ocs2::vector_t pose(7);
  pose.head<3>() = target.positionWorld;
  pose.tail<4>() = q.coeffs();  // Eigen/OCS2 convention: [qx, qy, qz, qw].
  return ocs2::TargetTrajectories({time, time + horizon_}, {pose, pose},
                                  {ocs2::vector_t(), ocs2::vector_t()});
}

Solution SolverCore::solve(const Observation& observation, const EndEffectorTarget& target) {
  const ocs2::vector_t initialState = observationToState(observation);
  referenceManager_->setTargetTrajectories(makeTarget(observation.time, target));
  if (!mpc_->run(observation.time, initialState)) {
    throw std::runtime_error("OCS2 MPC did not produce a new policy.");
  }
  ocs2::PrimalSolution trajectory =
      mpc_->getSolverPtr()->primalSolution(observation.time + horizon_);
  if (trajectory.timeTrajectory_.empty() || trajectory.stateTrajectory_.empty() ||
      trajectory.inputTrajectory_.empty()) {
    throw std::runtime_error("OCS2 returned an empty primal solution.");
  }
  if (trajectory.stateTrajectory_.size() != trajectory.timeTrajectory_.size() ||
      trajectory.timeTrajectory_.front() > observation.time + 1e-6 ||
      trajectory.timeTrajectory_.back() < observation.time + kWrenchPredictionTimes.back() - 1e-6 ||
      !std::is_sorted(trajectory.timeTrajectory_.begin(), trajectory.timeTrajectory_.end())) {
    throw std::runtime_error("OCS2 returned a malformed or too-short trajectory.");
  }
  for (const auto& state : trajectory.stateTrajectory_) {
    requireSize(state, kStateDim, "trajectory state");
    if (!state.allFinite()) throw std::runtime_error("OCS2 trajectory state is non-finite.");
  }
  for (const auto& input : trajectory.inputTrajectory_) {
    requireSize(input, kInputDim, "trajectory input");
    if (!input.allFinite()) throw std::runtime_error("OCS2 trajectory input is non-finite.");
  }
  const auto commandState = ocs2::LinearInterpolation::interpolate(
      observation.time + settings_.commandLeadTime, trajectory.timeTrajectory_,
      trajectory.stateTrajectory_);
  const auto commandInput = ocs2::LinearInterpolation::interpolate(
      observation.time, trajectory.timeTrajectory_, trajectory.inputTrajectory_);
  const auto commandInputLead = ocs2::LinearInterpolation::interpolate(
      observation.time + settings_.commandLeadTime, trajectory.timeTrajectory_,
      trajectory.inputTrajectory_);
  requireSize(commandState, kStateDim, "command state");
  requireSize(commandInput, kInputDim, "command input");
  requireSize(commandInputLead, kInputDim, "lead command input");

  Solution out;
  out.time = observation.time;
  out.armPosition = commandState.segment<6>(kArmPositionIndex);
  out.armVelocity = commandState.segment<6>(kArmVelocityIndex);
  out.baseVelocityCommand = commandInput.head<3>();
  out.armEffort = wrenchEstimator_->armEffort(commandState, commandInputLead);
  const auto effortLimits = pinocchioInterface_->getModel().effortLimit.tail(kArmDof);
  out.armEffort = out.armEffort.cwiseMax(-effortLimits).cwiseMin(effortLimits);
  out.baseWrenchPrediction = wrenchEstimator_->predict(observation.time, trajectory);
  out.valid = out.armPosition.allFinite() && out.armVelocity.allFinite() &&
              out.armEffort.allFinite() && out.baseVelocityCommand.allFinite() &&
              out.baseWrenchPrediction.allFinite();
  if (!out.valid) throw std::runtime_error("OCS2 solution contains non-finite values.");
  return out;
}

bool SolverCore::trySolve(const Observation& observation, const EndEffectorTarget& target,
                          Solution& solution, std::string* errorMessage) noexcept {
  try {
    solution = solve(observation, target);
    if (errorMessage != nullptr) errorMessage->clear();
    return true;
  } catch (const std::exception& error) {
    solution = Solution{};
    solution.time = observation.time;
    try {
      if (errorMessage != nullptr) *errorMessage = error.what();
    } catch (...) {
    }
    try {
      reset();
    } catch (...) {
      // Preserve the original solve error. Reconstruct the core if reset fails.
    }
    return false;
  } catch (...) {
    solution = Solution{};
    solution.time = observation.time;
    try {
      if (errorMessage != nullptr) *errorMessage = "Unknown OCS2 solver failure.";
    } catch (...) {
    }
    try {
      reset();
    } catch (...) {
    }
    return false;
  }
}

void SolverCore::reset() { mpc_->reset(); }

}  // namespace tron2_ocs2
