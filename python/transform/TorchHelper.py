# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================

import torch


class BaseNode():

    def __init__(self, info):
        self.name = str(info["name"])
        self.op_type = str(info["op_type"])
        self.inputs = list(info["inputs"])
        self.outputs = list(info["outputs"])

class TorchValue:
    def __init__(self,value,dtype:str):
        self.value=value
        self.dtype=dtype


class TorchNode(BaseNode):

    def __init__(self, node):
        info = dict()
        op_type = node.kind()
        info["op_type"] = op_type if not op_type.endswith("_") else op_type[:-1]
        info["inputs"] = [inp.debugName() for inp in node.inputs()]
        info["outputs"] = [outp.debugName() for outp in node.outputs()]
        info["name"] = info["outputs"][0]
        super().__init__(info)
        self.node_proto = node


def get_attr(model: torch.jit.RecursiveScriptModule, node: torch.Node):
    if node.kind() == 'prim::Param':
        return (model, '')
    if node.kind() == 'prim::GetAttr':
        name = node.s('name')
        obj, parent = get_attr(model, node.input().node())
        return (getattr(obj, name), parent + '.' + name if len(parent) > 0 else name)


def get_constant(node: torch.Node):
    """Retrieve a constant associated with this prim::Constant node"""
   
    attribute_names = node.attributeNames()
    num_attributes = len(attribute_names)
    name = node.output().debugName()
    is_tensor = False
    type = node.output().type().kind()
    value = None
    originDtype= None
    if type == "NoneType":
        return name, None, True
    elif num_attributes == 1:
        attr_name = attribute_names[0]
        if type in ["IntType", "LongType"]:
            value = node.i(attr_name)
            if type == "IntType":
              originDtype="INT32"
            else:
              originDtype="INT64"
        elif type == "BoolType":
            value = bool(node.i(attr_name))
            originDtype="BOOL"
        elif type in ["FloatType"]:
            value = node.f(attr_name)
            originDtype = "F32"  # 对应 torch.float32
        elif type in ["HalfType"]:  # 对应 torch.float16
            value = node.f(attr_name)
            originDtype = "F16"
        elif type in ["BFloat16Type"]:  # 对应 torch.bfloat16
            value = node.f(attr_name)
            originDtype = "BF16"
        elif type in ["Float8E4M3Type"]:  # 对应 torch.float8_e4m3fn
            value = node.f(attr_name)
            originDtype = "F8E4M3"
        elif type in ["Float8E5M2Type"]:  # 对应 torch.float8_e5m2
            value = node.f(attr_name)
            originDtype = "F8E5M2"
        elif type in ["StringType"]:
            value = node.s(attr_name)
            originDtype="STR"
        elif type in ["DeviceObjType"]:
            value = node.s(attr_name)
            if 'cuda' in value:
                device = (value if torch.cuda.is_available() else "cpu")
                value = device
        elif type in ["TensorType", "CompleteTensorType"]:
            is_tensor = True
            tensor = node.t(attr_name)
            if tensor.is_cuda:
                tensor = tensor.cpu()
            value,originDtype=torch_tensor_to_numpy(tensor)
        else:
            raise NotImplementedError("Unsupported type: %s" % type)
    else:
        assert num_attributes == 0
        return None
    return name, value, is_tensor, originDtype

def torch_tensor_to_numpy(tensor):
    torchDtypeToString={
        torch.uint8:"UINT8",
        torch.int8:"INT8",
        torch.int16:"INT16",
        torch.int32:"INT32",
        torch.int64:"INT64",
        torch.float16:"F16",
        torch.float32:"F32",
        torch.float64:"F64",
        torch.bool:"BOOL",
        torch.half:"F16",
        torch.bfloat16:"BF16",
        torch.float8_e4m3fn:"F8E4M3",
        torch.float8_e5m2:"F8E5M2",
    }

    originDtype=torchDtypeToString[tensor.dtype]
    #todo for fp8 bf16
    if tensor.dtype not in [torch.float16,torch.bfloat16,torch.float8_e4m3fn, torch.float8_e5m2]:
        value = tensor.numpy()
    elif tensor.dtype in [torch.float16, torch.bfloat16]:
        value=tensor.view(torch.uint16).numpy()
    elif tensor.dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
        value=tensor.view(torch.uint8).numpy()
    else:
        raise NotImplementedError("not support torch tensor dtype: %s" % tensor.dtype)
    return value,originDtype

