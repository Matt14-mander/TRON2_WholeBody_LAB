#include <exception>
#include <iostream>

#include "tron2_ocs2/SolverCore.h"

int main(int argc, char** argv) {
  if (argc != 4) {
    std::cerr << "usage: tron2_ocs2_smoke TASK_INFO ROBOT_URDF GENERATED_LIBRARY_DIR\n";
    return 2;
  }
  try {
    tron2_ocs2::SolverCore solver(argv[1], argv[2], argv[3]);
    tron2_ocs2::Observation observation;
    observation.basePositionWorld.z() = 0.74;
    observation.armPosition << 0.0, 1.57079632679, -1.48352986420, 0.0, 0.0, 0.0;
    tron2_ocs2::EndEffectorTarget target;
    // Deliberately use the current nominal vicinity; this executable checks model
    // construction and one complete MPC/RNEA pass, not task performance.
    target.positionWorld << 0.35, 0.0, 0.85;
    const auto solution = solver.solve(observation, target);
    std::cout << "valid=" << std::boolalpha << solution.valid
              << " base_cmd=" << solution.baseVelocityCommand.transpose()
              << " wrench0=" << solution.baseWrenchPrediction.row(0) << '\n';
    return solution.valid ? 0 : 1;
  } catch (const std::exception& error) {
    std::cerr << "tron2_ocs2_smoke failed: " << error.what() << '\n';
    return 1;
  }
}
