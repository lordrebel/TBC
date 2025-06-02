# TODO
## pepare env
```bash
source env.source
```
## how to build

### llvm prepare

#### 1. pull code

```bash
cd ..
git clone --branch llvmorg-19.1.7 --depth 1  https://gitee.com/mirrors/LLVM.git
```

#### 2. compile llvm

```bash

cd LLVM&&mkdir -p build
cmake -G Ninja ../llvm     -DLLVM_ENABLE_PROJECTS="mlir"     -DLLVM_INSTALL_UTILS=ON     -DLLVM_TARGETS_TO_BUILD=""     -DLLVM_ENABLE_ASSERTIONS=ON     -DMLIR_INCLUDE_TESTS=OFF     -DLLVM_INSTALL_GTEST=ON     -DMLIR_ENABLE_BINDINGS_PYTHON=ON     -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=../../llvm_release -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DLLVM_ENABLE_BINDINGS=ON   -DLLVM_ENABLE_LLD=ON -DLLVM_ENABLE_PIC=ON  -DMLIR_INCLUDE_INTEGRATION_TESTS=ON -DPython3_EXECUTABLE=$(which python3)

cmake --build . --target install

```

### 3. install omp

```bash
sudo apt-get install libgomp-dev
```

### 4. build
```bash
mkdir -p build && cd build && cmake .. && cmake --build . --target install
```
## how to use
