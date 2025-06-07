source env.source
# 获取 NumPy 包含目录路径
NUMPY_INCLUDE_DIR=$(python3 -c "import numpy; print(numpy.get_include())")
export NUMPY_INCLUDE_DIR

# 输出一些信息以便确认
echo "PYTHONPATH set to: $PYTHONPATH"
echo "NumPy include directory: $NUMPY_INCLUDE_DIR"
echo "Running CMake configue..."
mkdir -p build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=${INSTALL_PATH} -DPython3_NumPy_INCLUDE_DIRS=$NUMPY_INCLUDE_DIR -G Ninja -DPython3_EXECUTABLE=$(which python3)
echo "Running build"
cmake --build .
echo "Running install"
cmake --install .
echo "Build and installation completed successfully."

