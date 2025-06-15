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

cd LLVM&&mkdir -p build && cd build
cmake -G Ninja ../llvm     -DLLVM_ENABLE_PROJECTS="mlir"     -DLLVM_INSTALL_UTILS=ON     -DLLVM_TARGETS_TO_BUILD=""     -DLLVM_ENABLE_ASSERTIONS=ON     -DMLIR_INCLUDE_TESTS=OFF     -DLLVM_INSTALL_GTEST=ON     -DMLIR_ENABLE_BINDINGS_PYTHON=ON     -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=../../llvm_release -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DLLVM_ENABLE_BINDINGS=ON   -DLLVM_ENABLE_LLD=ON -DLLVM_ENABLE_PIC=ON  -DMLIR_INCLUDE_INTEGRATION_TESTS=ON -DPython3_EXECUTABLE=$(which python3)

cmake --build . --target install

```

### 3. install omp

```bash
sudo apt-get install libgomp-dev
```

### 4. build
```bash
bash build.sh
```
## how to use
### simple test
1. download lenet.onnx from [here](https://github.com/ONNC/onnc-tutorial/blob/master/models/lenet/lenet.onnx)
2. run
```bash
env env.source
model_transform.py --model_name lenet --model_path ./your/path/to/onnx --platform onnx - --mlir ./lenet.mlir
```
3. for more details see `model_transform.py --help`

## TODO List
- [ ] test fp8/fp16 onnx model import
- [ ] add svjson converter
- [ ] add platform dependent pass for optmize imported model
- [ ] kernel dialect design
