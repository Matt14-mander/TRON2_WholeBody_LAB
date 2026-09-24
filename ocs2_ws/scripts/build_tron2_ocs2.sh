#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
REPOSITORY_DIR="$(cd -- "$WORKSPACE_DIR/.." && pwd)"
ROS_DISTRO_NAME="${ROS_DISTRO:-humble}"
ROS_SETUP="/opt/ros/$ROS_DISTRO_NAME/setup.bash"

if [[ -z "${CONDA_PREFIX:-}" ]]; then
  echo "ERROR: activate the tron2_ocs2 Conda environment first." >&2
  exit 1
fi
PYTHON_BIN="$CONDA_PREFIX/bin/python"
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "ERROR: Conda Python not found: $PYTHON_BIN" >&2
  exit 1
fi
if [[ -f "$ROS_SETUP" ]]; then
  # A system ROS installation may add ros2 and colcon to PATH here.
  # shellcheck disable=SC1090
  source "$ROS_SETUP"
  ROS_SOURCE="$ROS_SETUP"
else
  # Conda-based ROS installations may already be active without /opt/ros.
  ROS_SOURCE="active environment"
fi
if ! NUMPY_INCLUDE="$("$PYTHON_BIN" -c 'import numpy; print(numpy.get_include())')"; then
  echo "ERROR: NumPy cannot be imported by $PYTHON_BIN after ROS setup." >&2
  echo "Check PYTHONPATH/PYTHONHOME for another environment, then install NumPy" >&2
  echo "into the active Conda environment if it is genuinely missing." >&2
  exit 1
fi
if [[ ! -f "$NUMPY_INCLUDE/numpy/arrayobject.h" ]]; then
  echo "ERROR: NumPy C headers not found under $NUMPY_INCLUDE." >&2
  exit 1
fi
if ! "$PYTHON_BIN" -c 'import ament_package; from ament_package.templates import get_environment_hook_template_path' >/dev/null; then
  echo "ERROR: ament_package is unavailable to $PYTHON_BIN after ROS setup." >&2
  echo "The ROS/ament Python packages and the CMake interpreter must belong" >&2
  echo "to the same environment; do not build with /opt/anaconda3/bin/python3.11." >&2
  exit 1
fi
if ! command -v ros2 >/dev/null 2>&1; then
  echo "ERROR: ros2 is unavailable; source a ROS 2 environment first." >&2
  exit 1
fi
if ! command -v colcon >/dev/null 2>&1; then
  echo "ERROR: colcon is unavailable in the active ROS 2 environment." >&2
  exit 1
fi
if ! ros2 pkg prefix ament_cmake >/dev/null 2>&1; then
  echo "ERROR: ROS 2 package ament_cmake is unavailable in the active environment." >&2
  exit 1
fi

find_config() {
  local prefix="$1"
  shift
  local pattern
  for pattern in "$@"; do
    local result
    result="$(find "$prefix" -type f -name "$pattern" -print -quit 2>/dev/null || true)"
    if [[ -n "$result" ]]; then
      printf '%s\n' "$result"
      return 0
    fi
  done
  return 1
}

PINOCCHIO_CONFIG="$(find_config "$CONDA_PREFIX" pinocchioConfig.cmake pinocchio-config.cmake || true)"
HPP_FCL_CONFIG="$(find_config "$CONDA_PREFIX" hpp-fclConfig.cmake hpp-fcl-config.cmake || true)"

if [[ -z "$PINOCCHIO_CONFIG" ]]; then
  echo "ERROR: Pinocchio CMake configuration was not found under $CONDA_PREFIX." >&2
  echo "Install a compatible C++ package, for example:" >&2
  echo "  conda --no-plugins install --solver=classic -c conda-forge 'pinocchio=2.7.*' 'hpp-fcl=2.4.*'" >&2
  exit 1
fi
if [[ -z "$HPP_FCL_CONFIG" ]]; then
  echo "ERROR: hpp-fcl CMake configuration was not found under $CONDA_PREFIX." >&2
  exit 1
fi

PINOCCHIO_DIR="$(dirname "$PINOCCHIO_CONFIG")"
HPP_FCL_DIR="$(dirname "$HPP_FCL_CONFIG")"

# The pinned ROS 2 OCS2 revision vendors an old GoogleTest which uses
# uintptr_t without including <cstdint>. New GCC releases reject it. Injecting
# the standard header through CXXFLAGS fixes the third-party source without
# mutating files inside the Conda environment.
case " ${CXXFLAGS:-} " in
  *" -include cstdint "*) ;;
  *) export CXXFLAGS="${CXXFLAGS:+$CXXFLAGS }-include cstdint" ;;
esac

export CMAKE_PREFIX_PATH="$CONDA_PREFIX:$PINOCCHIO_DIR:$HPP_FCL_DIR:${CMAKE_PREFIX_PATH:-}"
export PKG_CONFIG_PATH="$CONDA_PREFIX/lib/pkgconfig:${PKG_CONFIG_PATH:-}"

echo "Build configuration:"
echo "  workspace:       $WORKSPACE_DIR"
echo "  conda prefix:    $CONDA_PREFIX"
echo "  Python3:         $PYTHON_BIN"
echo "  NumPy includes:  $NUMPY_INCLUDE"
echo "  pinocchio_DIR:   $PINOCCHIO_DIR"
echo "  hpp-fcl_DIR:     $HPP_FCL_DIR"
echo "  ROS source:      $ROS_SOURCE"

cd "$WORKSPACE_DIR"
colcon build \
  --base-paths "$WORKSPACE_DIR/src" "$REPOSITORY_DIR/robot_description" \
  --packages-up-to tron2_ocs2 \
  --cmake-clean-cache \
  --cmake-args \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_POLICY_DEFAULT_CMP0167=OLD \
    -DPython3_EXECUTABLE="$PYTHON_BIN" \
    -DPYTHON_EXECUTABLE="$PYTHON_BIN" \
    -DPython_EXECUTABLE="$PYTHON_BIN" \
    -DPython3_ROOT_DIR="$CONDA_PREFIX" \
    -DPython3_NumPy_INCLUDE_DIR="$NUMPY_INCLUDE" \
    -Dpinocchio_DIR="$PINOCCHIO_DIR" \
    -Dhpp-fcl_DIR="$HPP_FCL_DIR"

echo
echo "Build complete. Load the overlay with:"
echo "  source '$WORKSPACE_DIR/install/setup.bash'"
echo "Then verify with:"
echo "  ros2 pkg executables tron2_ocs2"
