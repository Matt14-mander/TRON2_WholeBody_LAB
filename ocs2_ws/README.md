# Reproducible TRON2 OCS2 workspace

This directory keeps only the project-owned `tron2_ocs2` package in Git. The
upstream OCS2 and robotic-assets repositories are pinned by revision files and
ignored locally. Use the supplied scripts instead of cloning or patching those
dependencies by hand.

## Prerequisites

- ROS 2 Humble and `colcon`, either under `/opt/ros/humble` or already active
  in the shell (for example through Conda);
- an activated Conda environment (normally `tron2_ocs2`);
- Pinocchio 2.7 and hpp-fcl 2.4 C++ packages in that environment.

If the two CMake packages are absent, install them without loading optional
Conda plugins:

```bash
export CONDA_NO_PLUGINS=true
conda --no-plugins install --solver=classic -c conda-forge \
  'pinocchio=2.7.*' 'hpp-fcl=2.4.*'
```

Do not use `pip install pin` for this workspace: the build requires discoverable
C++ headers, libraries, and CMake package configuration files.

## Prepare and build

From the repository root:

```bash
bash ocs2_ws/scripts/prepare_sources.sh
conda activate tron2_ocs2
bash ocs2_ws/scripts/build_tron2_ocs2.sh
source ocs2_ws/install/setup.bash
ros2 pkg executables tron2_ocs2
```

`prepare_sources.sh` checks out the recorded upstream revisions and applies the
tracked ROS 2/urdfdom compatibility patches exactly once. The build script:

- uses `/opt/ros/$ROS_DISTRO/setup.bash` when available, otherwise checks that
  the active environment provides `ros2`, `colcon`, and `ament_cmake`;
- refuses to build until `pinocchioConfig.cmake` and `hpp-fclConfig.cmake` are
  actually present;
- passes their exact directories to CMake;
- works around the old vendored GoogleTest `uintptr_t` failure with
  `-include cstdint`;
- keeps CMake's legacy FindBoost policy enabled for the pinned OCS2 revision.

After a successful build, continue with the smoke and trajectory-export commands
in [`src/tron2_ocs2/README.md`](src/tron2_ocs2/README.md).
