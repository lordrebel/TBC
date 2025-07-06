#!/usr/bin/env python3
# ==============================================================================
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================

import numpy as np
import argparse
import os
from utils.dtype_convert import *


def show_fake_cmd(in_npz: str, model: str, out_npz: str):
    print("[CMD]: model_runner.py --input {} --model {} --output {}".format(in_npz, model, out_npz))




def model_inference(inputs: dict, model_file: str, dump_all = True) -> dict:
    raise NotImplementedError("not finished yet")


def mlir_inference(inputs: dict, mlir_file: str, dump_all: bool = True, debug=None) -> dict:
    raise NotImplementedError("not finished yet")

def syjson_inference(inputs: dict, onnx_file: str, dump_all: bool = True)->dict:
    raise NotImplementedError("not finished yet")


def onnx_inference(inputs: dict, onnx_file: str, dump_all: bool = True) -> dict:
    import onnx
    import onnxruntime

    def generate_onnx_with_all(onnx_file: str):
        # for dump all activations
        # plz refre https://github.com/microsoft/onnxruntime/issues/1455
        output_keys = []
        model = onnx.load(onnx_file)
        no_list = ["Cast", "Constant", "Dropout", "Loop"]

        # tested commited #c3cea486d https://github.com/microsoft/onnxruntime.git
        for x in model.graph.node:
            if x.op_type in no_list:
                continue
            for name in x.output:
                if not name:
                    continue
                intermediate_layer_value_info = onnx.helper.ValueInfoProto()
                intermediate_layer_value_info.name = name
                model.graph.output.append(intermediate_layer_value_info)
                output_keys.append(intermediate_layer_value_info.name + '_' + x.op_type)
        dump_all_tensors_onnx = onnx_file.replace('.onnx', '_all.onnx', 1)
        onnx.save(model, dump_all_tensors_onnx, save_as_external_data=True)
        return output_keys, dump_all_tensors_onnx

    output_keys = []
    if dump_all:
        output_keys, onnx_file = generate_onnx_with_all(onnx_file)
    session = onnxruntime.InferenceSession(onnx_file, providers=['CPUExecutionProvider'])
    inodes = session.get_inputs()
    only_one = len(inputs) == 1
    if only_one:
        assert (len(inodes) == 1)
    data = {}
    for node in inodes:
        name = node.name
        dtype = np.float32
        if node.type == 'tensor(int64)':
            dtype = np.int64
        elif node.type == 'tensor(bool)':
            dtype = np.bool_
        elif node.type == 'tensor(int32)':
            dtype = np.int32
        elif node.type == 'tensor(float16)':
            dtype = np.float16
        elif node.type == 'tensor(float8e4m3fn)' or node.type == 'tensor(float8e5m2)':
            # 对于fp8类型，由于numpy没有直接支持，通常会先转为float16或float32
            dtype = np.float32
            print(f"Warning: Converting {node.type} to float32 for inference")
        elif node.type == 'tensor(bfloat16)':
            dtype = np.float32
            print(f"Warning: Converting {node.type} to float32 for inference")
        if not only_one:
            assert (name in inputs)
            if node.type not in ['tensor(bfloat16)','tensor(float8e4m3fn)','tensor(float8e5m2)']:
                data[name] = inputs[name].astype(dtype)
            else:
                if node.type == 'tensor(float8e4m3fn)':
                    data[name] = float8e4m3fn_to_float32(inputs[name])
                elif node.type== 'tensor(float8e5m2)':
                    data[name] = float8e5m2_to_float32(inputs[name])
                elif node.type == 'tensor(bfloat16)':
                    data[name] = bf16_to_float32(inputs[name])
                else:
                    raise ValueError(f"Unsupported dtype: {node.type}")
        else:
            tensor= list(inputs.values())[0]
            if node.type not in ['tensor(bfloat16)','tensor(float8e4m3fn)','tensor(float8e5m2)']:
                data[name] = inputs[name].astype(dtype)
            else:
                if node.type == 'tensor(float8e4m3fn)':
                    data[name] = float8e4m3fn_to_float32(tensor)
                elif node.type== 'tensor(float8e5m2)':
                    data[name] = float8e5m2_to_float32(tensor)
                elif node.type == 'tensor(bfloat16)':
                    data[name] = bf16_to_float32(tensor)
                else:
                    raise ValueError(f"Unsupported dtype: {node.type}")

    outs = session.run(None, data)
    outputs = dict()
    if not dump_all:
        onodes = session.get_outputs()
        for node, out in zip(onodes, outs):
            outputs[node.name] = out.astype(np.float32)
        return outputs
    else:
        output_num = len(outs) - len(output_keys)
        outs = outs[output_num:]
        os.remove(onnx_file)
        return dict(filter(lambda x: isinstance(x[1], np.ndarray), zip(output_keys, outs)))


def torch_inference(inputs: dict, model: str, dump_all: bool = True) -> dict:
    import torch

    if dump_all:
        from transform.TorchInterpreter import TorchInterpreter
        net = TorchInterpreter(model)
        net.run_model(inputs)
        return net.ref_tensor
    net = torch.jit.load(model, map_location=torch.device('cpu'))
    net.eval()
    in_tensors=[]
    for k,v in inputs.items():
      if not isinstance(v, torch.Tensor):
          in_tensors.append(torch.from_numpy(v))
      else:
          in_tensors.append(v)
    with torch.no_grad():
        out_tensors = net(*in_tensors)

    names = []
    graph_alive = net.inlined_graph
    for out in graph_alive.outputs():
        if out.node().kind() == 'prim::TupleConstruct' or out.node().kind(
        ) == 'prim::ListConstruct':
            ins = out.node().inputs()
            names.extend([i.debugName() for i in ins])
        else:
            names.append(out.debugName())

    idx = 0

    def torch_outputs(outputs: dict, names: list, tensors):
        nonlocal idx
        if isinstance(tensors, torch.Tensor):
            outputs[names[idx]] = tensors.numpy()
            idx += 1
            return
        if isinstance(tensors, tuple) or isinstance(tensors, list):
            for t in tensors:
                torch_outputs(outputs, names, t)
        else:
            raise RuntimeError("Not Implemented")

    outputs = {}
    torch_outputs(outputs, names, out_tensors)
    return outputs


if __name__ == '__main__':
    # yapf: disable
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="input npz file")
    parser.add_argument("--model", type=str, required=True,
                        help="mlir/pytorch/onnx/tlf file.")
    parser.add_argument("--output", default='_output.npz', help="output npz file")
    parser.add_argument("--dump_all_tensors", action='store_true',
                        help="dump all tensors to output file")
    parser.add_argument("--debug", type=str, nargs="?", const="",
                        help="configure the debugging information.")

    # yapf: enable
    args = parser.parse_args()
    data = np.load(args.input)
    output = dict()
    if args.model.endswith(".mlir"):
        output = mlir_inference(data, args.model, args.dump_all_tensors, args.debug)
    elif args.model.endswith('.onnx'):
        output = onnx_inference(data, args.model, args.dump_all_tensors)

    elif args.model.endswith(".pt") or args.model.endswith(".pth"):
        output = torch_inference(data, args.model, args.dump_all_tensors)
    elif args.model.endswith(".tlf"):
        output = model_inference(data, args.model)
    else:
        raise RuntimeError("not support modle file:{}".format(args.model))
    print("\nSaving ...")
    if output:
        np.savez(args.output, **output)
        print("\nResult saved to:{}".format(args.output))
