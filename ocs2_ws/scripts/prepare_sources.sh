#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
SOURCE_DIR="$WORKSPACE_DIR/src"
PATCH_DIR="$WORKSPACE_DIR/patches"

OCS2_URL="https://github.com/leggedrobotics/ocs2.git"
OCS2_REVISION="$(tr -d '[:space:]' < "$PATCH_DIR/ocs2.base-revision")"
ASSETS_URL="https://github.com/leggedrobotics/ocs2_robotic_assets.git"
ASSETS_REVISION="$(tr -d '[:space:]' < "$PATCH_DIR/ocs2-robotic-assets.base-revision")"

clone_at_revision() {
  local url="$1"
  local destination="$2"
  local revision="$3"
  local branch="${4:-}"

  if [[ ! -d "$destination/.git" ]]; then
    if [[ -n "$branch" ]]; then
      git clone --branch "$branch" "$url" "$destination"
    else
      git clone "$url" "$destination"
    fi
    git -C "$destination" checkout --detach "$revision"
    return
  fi

  if [[ "$(git -C "$destination" rev-parse HEAD)" != "$revision" ]]; then
    echo "ERROR: $destination is not at the pinned revision $revision." >&2
    echo "       Current revision: $(git -C "$destination" rev-parse HEAD)" >&2
    echo "       Move local work aside or check out the pinned revision explicitly." >&2
    exit 1
  fi
}

apply_patch_once() {
  local repository="$1"
  local patch_file="$2"
  local patch_options=()

  # The frame-field compatibility patch changes isolated lines and therefore
  # has no surrounding context. git apply requires this explicit opt-in.
  if [[ "${3:-}" == "unidiff-zero" ]]; then
    patch_options+=(--unidiff-zero)
  fi

  if git -C "$repository" apply "${patch_options[@]}" --reverse --check "$patch_file" >/dev/null 2>&1; then
    echo "Already applied: $(basename "$patch_file")"
  elif git -C "$repository" apply "${patch_options[@]}" --check "$patch_file" >/dev/null 2>&1; then
    git -C "$repository" apply "${patch_options[@]}" "$patch_file"
    echo "Applied: $(basename "$patch_file")"
  else
    echo "ERROR: patch does not apply cleanly: $patch_file" >&2
    echo "       Verify that the dependency is at its pinned base revision." >&2
    exit 1
  fi
}

mkdir -p "$SOURCE_DIR"
clone_at_revision "$OCS2_URL" "$SOURCE_DIR/ocs2" "$OCS2_REVISION" ros2
clone_at_revision "$ASSETS_URL" "$SOURCE_DIR/ocs2_robotic_assets" "$ASSETS_REVISION"

apply_patch_once "$SOURCE_DIR/ocs2" "$PATCH_DIR/ocs2-urdfdom5.patch"
apply_patch_once "$SOURCE_DIR/ocs2" "$PATCH_DIR/ocs2-pinocchio2-frame-parent.patch" unidiff-zero
apply_patch_once "$SOURCE_DIR/ocs2" "$PATCH_DIR/ocs2-pinocchio2-frame-jacobian.patch" unidiff-zero
apply_patch_once "$SOURCE_DIR/ocs2" "$PATCH_DIR/ocs2-pinocchio2-remaining-frame-parents.patch"
apply_patch_once "$SOURCE_DIR/ocs2_robotic_assets" "$PATCH_DIR/ocs2-robotic-assets-ament.patch"

if grep -R -n -E --include='*.cpp' --include='*.h' --include='*.hpp' \
    'frame[.]parentJoint|frames\[[^]]+\][.]parentJoint' "$SOURCE_DIR/ocs2/ocs2_pinocchio"; then
  echo "ERROR: Pinocchio 3 Frame::parentJoint references remain in OCS2 sources." >&2
  exit 1
fi

echo "OCS2 sources are ready:"
echo "  ocs2:                 $OCS2_REVISION"
echo "  ocs2_robotic_assets:  $ASSETS_REVISION"
