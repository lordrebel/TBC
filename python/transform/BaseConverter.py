

import numpy as np
import sys
from mlir.ir import *

class ValueInfo:
    def __init__(self,shape,dtype):
        self.shape = shape
        self.dtype = dtype

class BaseConverter(object):

    def __init__(self):
        self.operands = dict()
        self.tensors = dict()  #weights or consts
        self.value_infos = dict()  # name -> ValueInfo
        self.input_names = list()
        self.output_names = list()
        self.input_shape_assgined=False

    def generate_mlir(self, mlir_file: str):
        raise NotImplementedError('generate_mlir')

    def addValueInfo(self, name, shape, dtype="F32"):
        if shape is None:
            shape = []
        elif len(shape) == 0:
            shape = [1]
        if isinstance(shape, tuple):
            shape = list(shape)
        elif not isinstance(shape, list):
            raise KeyError("{}:{} unknown shape".format(name, shape))
        if name in self.value_infos:
            if self.value_infos[name].shape != shape or self.value_infos[name].dtype != dtype:
                raise KeyError("shape {} conflict {} vs {}".format(name, self.shapes[name], shape))
        self.value_infos[name] = ValueInfo(shape, dtype)

    def getValueInfo(self, name) -> ValueInfo:
        if name not in self.value_infos:
            raise KeyError("value info {} not found".format(name))
        if(self.input_shape_assgined and name not in self.input_names and name not in self.tensors):
            self.value_infos[name].shape = []
        return self.value_infos[name]

    def setValueInfo(self, name, shape, dtype):
        self.value_infos[name] = ValueInfo(shape, dtype)

    def getShape(self, name):
        return self.getValueInfo(name).shape

    def setShape(self, name, shape):
        if name not in self.value_infos:
            raise KeyError("value info {} not found".format(name))
        if len(shape) == 0:
            shape = [1]
        if isinstance(shape, tuple):
            shape = list(shape)
        elif not isinstance(shape, list):
            raise KeyError("{}:{} unknown shape".format(name, shape))
        self.value_infos[name].shape = shape

    def setDtype(self, name, dtype):
        if name not in self.value_infos:
            raise KeyError("value info {} not found".format(name))
        self.value_infos[name].dtype = dtype
    def getDtype(self, name):
        if name not in self.value_infos:
            raise KeyError("value info {} not found".format(name))
        return self.value_infos[name].dtype

    def addOperand(self, name, op):
        if name in self.operands:
            if self.operands[name] != op:
                raise KeyError("operand {} conflict".format(name))
            return
        self.operands[name] = op

    def getOperand(self, name):
        if name not in self.operands:
            raise KeyError("operand  or weight {} not found".format(name))
        return self.operands[name]

    def getOp(self, name):
        if self.isWeight(name):
            return self.getWeightOp(name)
        return self.getOperand(name)

    def addWeight(self, name, data,origin_type="F32"):
        if not isinstance(data, np.ndarray):
            raise KeyError("tensor data must be numpy array")

        if name in self.tensors:
            if np.all(self.tensors[name] == data):
                return
            raise KeyError("tensor {} conflict".format(name))
        if len(data.shape) == 0:
            data = data.reshape([1])

        self.tensors[name] = data
        self.addValueInfo(name, data.shape,origin_type)

    def valueInfoToTensorType(self, name):
        value_info=self.value_infos.get(name, ValueInfo([], 'F32'))
        if(self.input_shape_assgined and name not in self.input_names and name not in self.tensors):
            value_info.shape = []

        return self.mlir.get_tensor_type(value_info.shape, value_info.dtype)
    def updateOperandTensorType(self, name):
        op=self.getOperand(name)
        if isinstance(op, Operation):
            tensor_type = self.valueInfoToTensorType(name)
            if op.results[0].type != tensor_type:
                op.results[0].set_type(tensor_type)
        else:
            op.set_type(self.valueInfoToTensorType(name))


    def isWeight(self, name):
        if name in self.tensors:
            return True
        return False

    def getWeight(self, name):
        if name not in self.tensors:
            raise KeyError("No {} tensor in model".format(name))
        return self.tensors[name]

    def isScalar(self, name):
        if not self.isWeight(name): return False
        if np.prod(self.getShape(name)) == 1: return True
        w = self.getWeight(name)
        return np.all(w == w.flatten()[0])

    def isScalar_(self, name, x):
        assert (isinstance(x, (int, float)))
        if not self.isWeight(name): return False
        if np.prod(self.getShape(name)) == 1: return True
        w = self.getWeight(name)
        return np.all(w == x)

    def getScalar(self, name):
        if not self.isScalar(name):
            raise RuntimeError("Not Scalar")
        return self.getWeight(name).flatten()[0]

    def getWeightOp(self, name, shape: list = []):
        if name not in self.tensors:
            raise KeyError("Should addWeight first:{}!!!".format(name))
        old_shape = self.getShape(name)
        if shape and old_shape != shape:
            assert (np.prod(old_shape) == np.prod(shape))
            old_shape = shape
        # ori_type = str(self.tensors[name].dtype)
        ori_type = self.getDtype(name)

        op = self.mlir.create_weight_op(name, old_shape, ori_type)
        self.addOperand(name, op)
        return op

    def WeightToNpz(self, weight_file):
        tensor_npz = {}
        for name in self.tensors:
            if name in self.operands:
                tensor_npz[name] = self.tensors[name]
        np.savez(weight_file, **tensor_npz)

    def get_loc(self, names):
        if isinstance(names, str):
            return Location.fused([Location.name(names)], context=self.mlir.ctx)
        elif isinstance(names, list):
            return Location.fused([Location.name(n) for n in names], context=self.mlir.ctx)
        else:
            raise RuntimeError("Unknown names:{}".format(names))
