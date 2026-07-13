#!/bin/bash
set -e

if [ -z "$1" ]; then
  echo "Error: Team name not provided."
  echo "Usage: $0 <TEAM_NAME>"
  exit 1
fi

TEAM_NAME="$1"

BUILD_DIR="./build"
DIST_DIR="$BUILD_DIR/dist"
WORK_DIR="$BUILD_DIR/build"
ONEFILE="--onefile"

rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"

echo "Building executable with PyInstaller..."
pyinstaller \
  --add-data './mujococodebase/skills/keyframe:mujococodebase/skills/keyframe' \
  --add-data './mujococodebase/skills/walk/*.onnx:mujococodebase/skills/walk' \
  ${ONEFILE} \
  --distpath "$DIST_DIR" \
  --workpath "$WORK_DIR" \
  --noconfirm \
  --name "${TEAM_NAME,,}" \
  ./run_player.py

echo "Creating start.sh..."
cat > "${DIST_DIR}/start.sh" << EOF
#!/bin/bash
export OMP_NUM_THREADS=1

host=\${1:-localhost}
port=\${2:-60000}

for i in {1..11}; do
  ./$(echo "${TEAM_NAME,,}") -t ${TEAM_NAME} -n \$i --host \$host --port \$port &
done
EOF

echo "Creating kill.sh..."
cat > "${DIST_DIR}/kill.sh" << EOF
#!/bin/bash
pkill -9 -e $(echo "${TEAM_NAME,,}")
EOF

chmod a+x "${DIST_DIR}/start.sh"
chmod a+x "${DIST_DIR}/kill.sh"

#TEMPORARY!
echo "Creating start3v3.sh..."
cat > "${DIST_DIR}/start3v3.sh" << EOF
#!/bin/bash
export OMP_NUM_THREADS=1

host=\${1:-localhost}
port=\${2:-60000}

for i in {1..3}; do
  ./$(echo "${TEAM_NAME,,}") -t ${TEAM_NAME} -f hl_adult -n \$i --host \$host --port \$port &
done
EOF

chmod a+x "${DIST_DIR}/start3v3.sh"

echo "Packing into ${BUILD_DIR}/${TEAM_NAME}.tar.gz..."
tar -czf "${BUILD_DIR}/${TEAM_NAME}.tar.gz" -C "$DIST_DIR" . --transform "s|^|${TEAM_NAME}/|"

echo "Removing build files..."
rm -rf $DIST_DIR
rm -rf $WORK_DIR
echo "Removed build files!"

echo "Generated file: ${BUILD_DIR}/${TEAM_NAME}.tar.gz"
echo "Build completed successfully!"