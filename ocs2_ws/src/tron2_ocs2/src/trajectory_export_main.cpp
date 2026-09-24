#include <exception>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

#include "tron2_ocs2/SolverCore.h"

namespace {

double parseDouble(const char* text, const char* name) {
  std::size_t consumed = 0;
  const double value = std::stod(text, &consumed);
  if (text[consumed] != '\0' || !std::isfinite(value)) {
    throw std::invalid_argument(std::string(name) + " must be a finite number.");
  }
  return value;
}

void writeHeader(std::ostream& stream) {
  stream << "time";
  for (int i = 1; i <= 6; ++i) stream << ",arm_q" << i;
  for (int i = 1; i <= 6; ++i) stream << ",arm_dq" << i;
  for (int i = 1; i <= 6; ++i) stream << ",arm_tau" << i;
  stream << ",base_vx,base_vy,base_wz";
  constexpr const char* components[] = {"fx", "fy", "fz", "tx", "ty", "tz"};
  for (int sample = 0; sample < 5; ++sample) {
    for (const char* component : components) {
      stream << ",w" << sample << '_' << component;
    }
  }
  stream << '\n';
}

template <typename Derived>
void writeEigen(std::ostream& stream, const Eigen::MatrixBase<Derived>& value) {
  for (Eigen::Index index = 0; index < value.size(); ++index) {
    stream << ',' << value.derived().coeff(index);
  }
}

}  // namespace

int main(int argc, char** argv) {
  const bool relativeTarget = argc == 11 && std::string(argv[5]) == "--relative-target";
  if (!relativeTarget && argc != 5 && argc != 12 && argc != 13 && argc != 14) {
    std::cerr
        << "usage: tron2_ocs2_trajectory_export TASK_INFO ROBOT_URDF "
           "GENERATED_LIBRARY_DIR OUTPUT_CSV "
           "[TARGET_X TARGET_Y TARGET_Z TARGET_QW TARGET_QX TARGET_QY TARGET_QZ "
           "[SAMPLE_PERIOD [ARRIVAL_TIME]]]\n"
           "   or: ... OUTPUT_CSV --relative-target DX DY DZ SAMPLE_PERIOD ARRIVAL_TIME\n";
    return 2;
  }
  try {
    tron2_ocs2::Observation observation;
    observation.basePositionWorld.z() = 0.74;
    observation.armPosition << 0.0, 1.57079632679, -1.48352986420, 0.0, 0.0, 0.0;
    tron2_ocs2::SolverCore solver(argv[1], argv[2], argv[3]);
    tron2_ocs2::EndEffectorTarget target;
    target.positionWorld << 0.35, 0.0, 0.85;
    double samplePeriod = 0.02;
    if (relativeTarget) {
      // Keep the home end-effector orientation. A fixed identity orientation
      // can demand an unreachable wrist pose even for a small XYZ move.
      target = solver.currentEndEffectorTarget(observation);
      target.positionWorld += Eigen::Vector3d(
          parseDouble(argv[6], "DX"), parseDouble(argv[7], "DY"),
          parseDouble(argv[8], "DZ"));
      samplePeriod = parseDouble(argv[9], "SAMPLE_PERIOD");
      target.arrivalTime = parseDouble(argv[10], "ARRIVAL_TIME");
    } else if (argc >= 12) {
      target.positionWorld << parseDouble(argv[5], "TARGET_X"),
          parseDouble(argv[6], "TARGET_Y"), parseDouble(argv[7], "TARGET_Z");
      target.orientationWorld = Eigen::Quaterniond(
          parseDouble(argv[8], "TARGET_QW"), parseDouble(argv[9], "TARGET_QX"),
          parseDouble(argv[10], "TARGET_QY"), parseDouble(argv[11], "TARGET_QZ"));
      if (argc >= 13) samplePeriod = parseDouble(argv[12], "SAMPLE_PERIOD");
      if (argc == 14) target.arrivalTime = parseDouble(argv[13], "ARRIVAL_TIME");
    }

    const auto samples = solver.solveTrajectory(observation, target, samplePeriod);
    std::ofstream output(argv[4]);
    if (!output) throw std::runtime_error(std::string("Cannot write trajectory: ") + argv[4]);
    output << std::setprecision(17);
    writeHeader(output);
    for (const auto& sample : samples) {
      output << sample.time;
      writeEigen(output, sample.armPosition);
      writeEigen(output, sample.armVelocity);
      writeEigen(output, sample.armEffort);
      writeEigen(output, sample.baseVelocityCommand);
      writeEigen(output, sample.baseWrenchPrediction);
      output << '\n';
    }
    output.close();
    if (!output) throw std::runtime_error(std::string("Failed while writing trajectory: ") + argv[4]);
    std::cout << "exported " << samples.size() << " samples to " << argv[4]
              << " at dt=" << samplePeriod << " s\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "tron2_ocs2_trajectory_export failed: " << error.what() << '\n';
    return 1;
  }
}
