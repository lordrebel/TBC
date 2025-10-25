#!/usr/bin/env python3
import abc
import numpy as np
import argparse

from transform.BaseConverter import BaseConverter
from utils.mlir_shell import *
from utils.mlir_parser import *
from utils.misc import *
from utils.auto_remove import file_mark, file_clean
#import torch


class ModelTransformer(object):

    def __init__(self, model_name, model_path,chip,without_simplfy=False):
        self.model_name = model_name
        self.model_path = model_path
        self.converter = BaseConverter()
        self.chip = chip
        self.without_simplfy = without_simplfy

    def cleanup(self):
        file_clean()

    @staticmethod
    def ensure_batch_size(arr: np.ndarray, batch_size):
        """arr: [old_batch_size, ...]"""
        old_batch_size = arr.shape[0]
        if old_batch_size > 1:
            return arr
        repeat_factor = int(np.ceil(batch_size / old_batch_size))
        repeated_arr = np.repeat(arr, repeat_factor, axis=0)
        trimmed_arr = repeated_arr[:batch_size]
        return trimmed_arr

    def model_transform(self, mlir_file: str):
        self.mlir_file = mlir_file
        mlir_origin = mlir_file.replace('.mlir', '_origin.mlir', 1)
        ops_mlir = mlir_file.replace('.mlir', '_ops.mlir', 1)
        kls_mlir = mlir_file.replace('.mlir', '_kls.mlir', 1)
        file_mark(mlir_origin)
        file_mark(ops_mlir)
        file_mark(kls_mlir)
        self.converter.generate_mlir(mlir_origin)
        mlir_opt_for_operator(mlir_origin, ops_mlir)
        mlir_lowering_to_kernel(ops_mlir, kls_mlir, self.chip)

        mlir_lowering_to_hal(kls_mlir, mlir_file)
        print("Mlir file generated:{}".format(mlir_file))

        self.module_parsered = MlirParser(self.mlir_file)
        self.input_num = self.module_parsered.get_input_num()


    @abc.abstractmethod
    def origin_inference(self, inputs: dict) -> dict:
        pass


class OnnxTransformer(ModelTransformer):

    def __init__(self,
                 model_name,
                 model_path,
                 chip,
                 mode="F32",
                 input_shapes: list = [],
                 output_names: list = [],
                 without_simplfy=False,
                 onnx_sim=''):
        super().__init__(model_name, model_path,chip,without_simplfy)
        from transform.OnnxConverter import OnnxConverter
        self.converter = OnnxConverter(self.model_name,
                                       self.model_path,
                                       mode,
                                       input_shapes,
                                       output_names,
                                       self.without_simplfy,
                                       onnx_sim=onnx_sim)

    def origin_inference(self, inputs: dict):
        from tools.model_runner import onnx_inference
        return onnx_inference(inputs, self.converter.onnx_file)

class SvJsonTransformer(ModelTransformer):

    def __init__(self,
                 model_name,
                 model_path,
                 chip,
                 mode="F32",
                 input_shapes: list = [],
                 output_names: list = [],
                 without_simplfy=False,
                 onnx_sim=''):
        super().__init__(model_name, model_path,chip,without_simplfy)
        from transform.SVjsonConverter import SvJsonConverter
        self.converter = SvJsonConverter(self.model_name,
                                       self.model_path,
                                       mode,
                                       input_shapes,
                                       output_names)

    def origin_inference(self, inputs: dict):
        raise NotImplementedError("SVJsonTransformer does not support origin inference directly. ")


class TorchTransformer(ModelTransformer):

    def __init__(self,
                 model_name,
                 model_path,
                 chip,
                 mode="F32",
                 input_shapes: list = [],
                 input_types: list = [],
                 output_names: list = [],
                 without_simplfy=False):
        super().__init__(model_name, model_path,chip,without_simplfy)
        from transform.TorchConverter import TorchConverter
        self.converter = TorchConverter(self.model_name,
                                       self.model_path,
                                       mode,
                                       input_shapes,
                                       input_types,
                                       output_names)
    def origin_inference(self, inputs: dict):
        from tools.model_runner import torch_inference
        return torch_inference(inputs, self.model_def)

def get_model_transform(args):

    if not args.mlir.endswith('.mlir'):
        raise RuntimeError("your mlir file should endswith .mlir, not:{}".format(args.mlir))
    tool = None
    if args.platform=="onnx":
        tool = OnnxTransformer(args.model_name,
                               args.model_path,
                               args.chip,
                               args.mode,
                               args.input_shapes,
                               args.output_names,
                               args.without_simplfy,
                               onnx_sim=args.onnx_sim)
    elif args.platform == "torch":
        tool = TorchTransformer(args.model_name, args.model_path, args.chip,args.mode,args.input_shapes,
                                args.input_types, args.output_names,args.without_simplfy)
    elif args.platform == "svjson":
        tool = SvJsonTransformer(args.model_name, args.model_path, args.chip,args.mode,args.input_shapes,
                                 args.output_names, args.without_simplfy)

    else:
        # TODO: support more deep learning model types
        raise RuntimeError("unsupport model:{}".format(args.model_path))
    return tool


if __name__ == '__main__':
    print("TBC Toolchain 1.0.0")
    parser = argparse.ArgumentParser()
    # yapf: disable
    parser.add_argument("--model_name", required=True, help="model name")
    parser.add_argument("--model_path", required=True, help="model definition file.")
    parser.add_argument("--platform", type=str, required=True,choices=["onnx", "torch","svjson"],
                        help="model platform, like:onnx,torch")
    parser.add_argument("--chip", type=str, required=True,choices=["npu_v1","npu_v1"],
                        help="compile target chip, like:npu_v1,NPU_npu_v2")
    parser.add_argument("--mode",type=str,default="F32",choices=["INT8","UINT8","INT4","BF16","F16","F32","W8F16","W8BF16","W4F16","W4BF16","F8E4M3","F8E5M2","W4F8E4M3","W4F8E5M2",],
                        help="the precision of model, like:INT8,UINT8,INT4,BF16,F16,F32,W8F16,W8BF16,W4 default is F32")
    parser.add_argument("--input_shapes", type=str2shape, default=list(),
                        help="list of input shapes, like:[[1,3,224,224],[10],[16]]")
    parser.add_argument("--input_types", type=str2list, default=list(),
                        help="list of input types, like:float32,int32. if not set, float32 as default for torch model only")
    parser.add_argument("--output_names", type=str2list, default=list(),
                        help="if set, will find names in model and set as real outputs")
    parser.add_argument("--without_simplfy", action='store_true',
                        help="if set, will not simplify the model during import")

    parser.add_argument("--onnx_sim", default="", type=str, choices=['', 'skip_fuse_bn'],
                        help="pass options of onnx-sim, sep by quote without space")
    parser.add_argument("--debug", action='store_true', help='to keep all intermediate files for debug')
    parser.add_argument("--mlir", type=str, required=True, help="output mlir model file")
    # yapf: enable
    args, unknown_args = parser.parse_known_args()
    if unknown_args:
        args.unknown_params += unknown_args
    tool = get_model_transform(args)
    tool.model_transform(args.mlir)

    # if not args.debug:
    #     tool.cleanup()
