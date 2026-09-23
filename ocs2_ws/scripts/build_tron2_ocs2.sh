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
if [[ ! -f "$ROS_SETUP" ]]; then
  echo "ERROR: ROS setup not found: $ROS_SETUP" >&2
  exit 1
fi
# Load ROS before checking for colcon because some installations expose it
# only through the ROS environment hooks.
# shellcheck disable=SC1090
source "$ROS_SETUP"
if ! command -v colcon >/dev/null 2>&1; then
  echo "ERROR: colcon is not available." >&2
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
echo "  pinocchio_DIR:   $PINOCCHIO_DIR"
echo "  hpp-fcl_DIR:     $HPP_FCL_DIR"
echo "  ROS setup:       $ROS_SETUP"

cd "$WORKSPACE_DIR"
colcon build \
  --base-paths "$WORKSPACE_DIR/src" "$REPOSITORY_DIR/robot_description" \
  --packages-up-to tron2_ocs2 \
  --cmake-clean-cache \
  --cmake-args \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_POLICY_DEFAULT_CMP0167=OLD \
    -Dpinocchio_DIR="$PINOCCHIO_DIR" \
    -Dhpp-fcl_DIR="$HPP_FCL_DIR"

echo
echo "Build complete. Load the overlay with:"
echo "  source '$WORKSPACE_DIR/install/setup.bash'"
echo "Then verify with:"
echo "  ros2 pkg executables tron2_ocs2"
