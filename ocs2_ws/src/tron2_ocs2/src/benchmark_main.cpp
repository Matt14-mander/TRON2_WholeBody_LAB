// Computational benchmark: frozen observation, monotonically advancing MPC time.
// No Isaac, socket, robot actuation or simulator clock is involved.
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>

#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>

#include "tron2_ocs2/SolverCore.h"

namespace {
template <typename Vector>
void readVector(const boost::property_tree::ptree& tree, const char* key, Vector& output) {
  const auto& array = tree.get_child(key);
  if (array.size() != static_cast<std::size_t>(output.size())) {
    throw std::invalid_argument(std::string("Invalid vector size: ") + key);
  }
  int index = 0;
  for (const auto& item : array) {
    const double value = item.second.get_value<double>();
    if (!std::isfinite(value)) throw std::invalid_argument(std::string("Non-finite: ") + key);
    output(index++) = value;
  }
}
template <typename Vector>
void printVector(const Vector& vector) {
  std::cout << '[';
  for (int i = 0; i < vector.size(); ++i) {
    if (i) std::cout << ',';
    std::cout << vector(i);
  }
  std::cout << ']';
}
}  // namespace

int main(int argc, char** argv) {
  if (argc < 4 || argc > 7) {
    std::cerr << "usage: tron2_ocs2_benchmark TASK_INFO ROBOT_URDF LIBRARY_DIR [SAMPLES=20] [PERIOD=0.1] [OBSERVATION_JSON]\n";
    return 2;
  }
  try {
    const int samples = argc > 4 ? std::stoi(argv[4]) : 20;
    const double period = argc > 5 ? std::stod(argv[5]) : 0.1;
    if (samples < 2 || !std::isfinite(period) || period <= 0.0) {
      throw std::invalid_argument("Require samples >= 2 and finite positive period.");
    }
    tron2_ocs2::SolverCore solver(argv[1], argv[2], argv[3]);
    tron2_ocs2::Observation observation;
    observation.basePositionWorld.z() = solver.initialBaseHeight();
    observation.armPosition << 0.0, 1.57079632679, -1.48352986420, 0.0, 0.0, 0.0;
    auto target = solver.currentEndEffectorTarget(observation);
    if (argc > 6) {
      boost::property_tree::ptree tree;
      boost::property_tree::read_json(argv[6], tree);
      observation.time = tree.get<double>("time");
      readVector(tree, "base_position_world", observation.basePositionWorld);
      Eigen::Vector4d quaternion;
      readVector(tree, "base_quaternion_world", quaternion);
      observation.baseOrientationWorld = Eigen::Quaterniond(quaternion(0), quaternion(1), quaternion(2), quaternion(3));
      readVector(tree, "base_twist_body", observation.baseTwistBody);
      readVector(tree, "arm_position", observation.armPosition);
      readVector(tree, "arm_velocity", observation.armVelocity);
      readVector(tree, "end_effector_target_position_world", target.positionWorld);
      readVector(tree, "end_effector_target_quaternion_world", quaternion);
      target.orientationWorld = Eigen::Quaterniond(quaternion(0), quaternion(1), quaternion(2), quaternion(3));
      target.arrivalTime = tree.get<double>("end_effector_arrival_time", 0.0);
    }
    const double initialTime = observation.time;
    // Preserve one solver instance: the first request is cold, subsequent ones
    // can reuse the previous policy even when wall-clock execution is slow.
    std::cout << std::setprecision(17);
    for (int index = 0; index < samples; ++index) {
      observation.time = initialTime + index * period;
      const auto started = std::chrono::steady_clock::now();
      const auto solution = solver.solve(observation, target);
      const double duration = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
      const auto& profile = solver.lastSolveProfile();
      if (!solution.valid || !std::isfinite(profile.cost) || !std::isfinite(profile.equalitySSE) ||
          !std::isfinite(profile.minimumArmPositionMargin)) {
        throw std::runtime_error("Invalid benchmark solution or performance index.");
      }
      std::cout << "{\"sample\":" << index << ",\"cold\":" << (profile.cold ? "true" : "false")
                << ",\"observation_sim_time\":" << observation.time
                << ",\"solve_duration_s\":" << duration << ",\"period_s\":" << period
                << ",\"deadline_missed\":" << (duration > period ? "true" : "false")
                << ",\"reference_ms\":" << profile.referenceMs
                << ",\"optimization_ms\":" << profile.optimizationMs
                << ",\"extraction_ms\":" << profile.extractionMs
                << ",\"validation_sample_ms\":" << profile.validationSampleMs
                << ",\"iterations\":" << profile.iterations << ",\"knots\":" << profile.knots
                << ",\"cost\":" << profile.cost << ",\"equality_sse\":" << profile.equalitySSE
                << ",\"minimum_arm_position_margin\":" << profile.minimumArmPositionMargin
                << ",\"maximum_arm_speed\":" << profile.maximumArmSpeed
                << ",\"maximum_arm_acceleration\":" << profile.maximumArmAcceleration
                << ",\"arm_position\":";
      printVector(solution.armPosition);
      std::cout << ",\"arm_velocity\":"; printVector(solution.armVelocity);
      std::cout << ",\"base_velocity_command\":"; printVector(solution.baseVelocityCommand);
      std::cout << "}" << std::endl;
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "tron2_ocs2_benchmark failed: " << error.what() << '\n';
    return 1;
  }
}
