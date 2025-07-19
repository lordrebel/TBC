from .MLIRImporter import MLIRImporter, Platform
from .BaseConverter import BaseConverter
from mlir.ir import *
import mlir.dialects.operators as operators
import numpy as np
import json
from utils.auto_remove import file_mark, file_clean
from .SvHelper import SvNode, SvGraph,HwDtypeToMlirDtype
class SvJsonConverter(BaseConverter):
    def __init__(self,
                 model_name: str,
                 sv_file,
                 mode:str,
                 input_shapes: list,
                 output_names: list,
                 ):
        super().__init__()

        self.model_name = model_name
        self.weight_file = "{}_origin_weight.npz".format(model_name)
        self.model = None
        self.mlir = None
        self.mode=mode
        self.converted_nodes = list()

        self.load_sv_model(sv_file, input_shapes, output_names)
        self.init_MLIRImporter()

        self.svop_factory={
            "Activation4":lambda node: self.convert_activation_op(node),
            "Getmaxidx":lambda node: self.convert_argmax_op(node),
            "Bn":lambda node: self.convert_Batchnorm_op(node),
            "V2V":lambda node: self.convert_bitwise_op(node),
            "VV2V":lambda node: self.convert_bitwise_op(node),
            "X2V":lambda node: self.convert_bitwise_op(node),
            "V2X":lambda node: self.convert_bitwise_op(node),
            "Broadcast2": lambda node: self.convert_broadcast_op(node),
            "Concat2": lambda node: self.convert_concat_op(node),
            "ConstantPad2d":lambda node: self.convert_pad_op(node),
            "Conv2d":lambda node: self.convert_conv2d_op(node),
            "Conv1d":lambda node: self.convert_conv1d_op(node),
            "Wino2d":lambda node: self.convert_conv2d_op(node),
            "ConvTranspose2d":lambda node: self.convert_conv_transpose2d_op(node),
            "ConvTranspose1d":lambda node: self.convert_conv_transpose1d_op(node),
            "Deconv2d":lambda node: self.convert_conv_transpose2d_op(node),
            "Eltwise3":lambda node: self.convert_eltwise_op(node),
            "Compare":lambda node: self.convert_compare_op(node),
            "Fc3":lambda node: self.convert_fc_op(node),
            "MultAdd":lambda node: self.convert_mult_add_op(node),
            "MultAcc":lambda node: self.convert_mult_acc_op(node),
            "GlobalPool2d":lambda node: self.convert_global_pool2d_op(node),
            "Interp3":lambda node: self.convert_interp3_op(node),
            "Localconv3":lambda node: self.convert_local_conv_op(node),
            "Matmul":lambda node: self.convert_matmul_op(node),
            "Matmul2":lambda node: self.convert_matmul_op(node),
            "NMS":lambda node: self.convert_nms_op(node),
            "Permute2":lambda node: self.convert_permute_op(node),
            "PixelShuffle2":lambda node: self.convert_pixel_shuffle_op(node),
            "PixelUnshuffle2":lambda node: self.convert_pixel_unshuffle_op(node),
            "Pool2d":lambda node: self.convert_pool2d_op(node),
            "Pool2d2":lambda node: self.convert_pool2d_op(node),
            "Reduce":lambda node: self.convert_reduce_op(node),
            "ReduceExt":lambda node: self.convert_reduce_ext_op(node),
            "ReflectionPad2d":lambda node: self.convert_pad_op(node),
            "ReplicationPad2d":lambda node: self.convert_pad_op(node),
            "Reshape3":lambda node: self.convert_reshape_op(node),
            "Rmsnorm":lambda node: self.convert_rmsnorm_op(node),
            "Ln":lambda node: self.convert_ln_op(node),
            "Slice2":lambda node: self.convert_slice_op(node),
            "Softmax2":lambda node: self.convert_softmax_op(node),
            "Topk":lambda node: self.convert_topk_op(node),
            "yuv2rgb":lambda node: self.convert_yuv2rgb_op(node)
        }

    def get_value_infos(self, model: SvGraph):
        for v in model.TensorInfos.values():
            self.addValueInfo(v.name, v.shape, v.dtype)
    def select_output(self, output_names: list):
        # set new output
        self.all_outputs = []
        self.all_inputs = {}
        for x in self.model.inputs:
            self.all_inputs[x] = self.model.TensorInfos[x]

        for x in self.model.outputs:
            if x in output_names:
                self.all_outputs.append(x)
                output_names.remove(x)
                if len(output_names) == 0:
                    break
        for x in self.model.TensorInfos.values():

            if x.name not in output_names:
                continue
            self.model.outputs.append(x.name)
            self.all_outputs.append(x.name)
            output_names.remove(x.name)

        if len(output_names) != 0:
            raise RuntimeError("Error, can't find {} in model".format(output_names))


    def input_shape_assign(self, input_shapes):
        inputs = self.model.inputs
        shape_changed = False
        no_shape = True

        def check_shape(l, r):
            if no_shape == False and l != r:
                raise KeyError("input shapes error:{}, {} vs {}".format(input_shapes, l, r))

        if len(input_shapes) > 0:
            no_shape = False
            check_shape(self.num_input, len(input_shapes))
        for idx, input in enumerate(inputs):
            shape = self.model.TensorInfos[input].shape
            # for 1-element scalars that has no shape, assign [1] as shape to convert to tensor
            if  shape is None or len(shape) == 0:
                shape=[1]
            num_dims = len(shape)
            if no_shape == False:
                check_shape(num_dims, len(input_shapes[idx]))
            _shape = []
            for _i, _dim in enumerate(shape):
                if _dim <= 0:
                    if no_shape:
                        assert 0, "Please check --input_shapes formula or check if there is any dynamic dim"
                    else:
                        _dim = input_shapes[idx][_i]
                        shape_changed = True
                # elif not no_shape:
                #     check_shape(_dim_value, input_shapes)
                elif not no_shape and input_shapes[idx][_i] != _dim:
                    _dim = input_shapes[idx][_i]
                    shape_changed = True
                _shape.append(_dim)
            self.value_infos[input].shape = _shape

        self.input_shape_assgined=shape_changed

    def load_sv_model(self, sv_file, input_shapes, output_names):
        self.model =SvGraph(json.load(open(sv_file, 'r')))
        self.get_value_infos(self.model)
        if output_names:
            self.select_output(output_names)
        self.input_names = self.model.inputs
        self.num_input = len(self.input_names)
        self.input_shape_assign(input_shapes)
        print("Input_shape assigned")

        self.input_shapes = [self.model.TensorInfos[item].shape for item in self.input_names]
        self.input_types = [self.model.TensorInfos[item].dtype for item in self.input_names]
        self.output_types = [self.model.TensorInfos[item].dtype for item in self.model.outputs]

        self.output_names = self.model.outputs

    def init_MLIRImporter(self):
        input_shapes = list()
        for _name in self.input_names:
            input_shapes.append(self.getShape(_name))
        output_shapes = list()
        for _name in self.output_names:
            output_shapes.append(self.getShape(_name))
        # init importer
        self.mlir = MLIRImporter(input_shapes, output_shapes, self.model_name, Platform.SV_JSON,
                                 self.mode,
                                 self.input_types,self.output_types)
        self.weight_file = self.mlir.weight_file

    def __del__(self):
        if self.mlir != None:
            del self.mlir
            self.mlir = None

    def cleanup(self):
        file_clean()

    def generate_mlir(self, mlir_file: str):
        """convert all to mlir"""
        # add input op
        for idx, _name in enumerate(self.input_names):
            input_ = self.mlir.create_input_op(self.get_loc(_name), idx)
            self.addOperand(_name, input_)

        def NoneAndRaise(node):
            raise RuntimeError("{} Op not support now".format(node.opType))

        self.converted_nodes.clear()
        for n in self.model.nodes:
            self.converted_nodes.append(n)
        # checkout all type is supported
        unsupported = set()
        for n in self.converted_nodes:
            if n.opType not in self.svop_factory:
                unsupported.add(n.opType)
        if unsupported:
            raise RuntimeError("Op not support:{}".format(unsupported))

        for n in self.converted_nodes:
            self.svop_factory.get(n.opType, lambda x: NoneAndRaise(x))(n)
        # add return op
        return_op = list()
        # Set output
        for idx, _name in enumerate(self.output_names):
            op = self.getOp(_name)
            return_op.append(op)

        self.mlir.create_return_op(return_op)
        mlir_txt = self.mlir.print_module()
        with open(mlir_file, "w") as f:
            f.write(mlir_txt)
        self.WeightToNpz(self.weight_file)
        print("Save mlir file: {}".format(mlir_file))


    def convert_activation_op(self, sv_node: SvNode):

        if(sv_node.hwType == "F16"):
            np_array_type=np.uint16
        elif(sv_node.hwType == "F8E4M3"):
            np_array_type=np.uint8
        else:
            raise RuntimeError("unknown hw type:{}".format(sv_node.hwType))

        lut_x=np.array(sv_node.data["lut_x"],dtype=np_array_type)
        lut_y=np.array(sv_node.data["lut_y"],dtype=np_array_type)
        lut_k=np.array(sv_node.data["lut_k"],dtype=np_array_type)
        self.addWeight("{}.{}".format(sv_node.name, "lut_x"), lut_x,sv_node.hwType)
        self.addWeight("{}.{}".format(sv_node.name, "lut_y"), lut_y,sv_node.hwType)
        self.addWeight("{}.{}".format(sv_node.name, "lut_k"), lut_k,sv_node.hwType)

        new_op = operators.ActivationLutOp(
            self.valueInfoToTensorType(sv_node.outputs[0]),
            self.getOp(sv_node.inputs[0]),
            self.getOp("{}.{}".format(sv_node.name, "lut_x")),
            self.getOp("{}.{}".format(sv_node.name, "lut_y")),
            self.getOp("{}.{}".format(sv_node.name, "lut_k")),
            sig_mode=sv_node.params.get("sig_mode", 0),
            bin_mode=sv_node.params.get("bin_mode", 0),
            cal_mode=sv_node.params.get("cal_mode", 0),
            loc=self.get_loc("{}_{}".format(sv_node.name, sv_node.opType)),
            ip=self.mlir.insert_point).output
        self.addOperand(sv_node.outputs[0], new_op)

    def convert_argmax_op(self, sv_node: SvNode):
        assert sv_node.opType == "Getmaxidx", "sv node type should be Getmaxidx, not {}".format(sv_node.opType)
        input=self.getOp(sv_node.inputs[0])
        axis=0
        keepDims=False
        """
        out_op = operators.ArgOp(*self.mlir.get_tensor_type(out_shapes),
                           op,
                           axis=axis,
                           keepdims=keepdims,
                           mode=StringAttr.get(onnx_node.op_type),
                           select_last_index=select_last_index,
                           loc=self.get_loc(loc_names),
                           ip=self.mlir.insert_point)
        out_ops = [out_op.indices, out_op.values]
        for idx, need in enumerate(out_needs):
        """
        new_op = operators.ArgOp(
            self.valueInfoToTensorType(sv_node.outputs[0]),
            self.mlir.get_tensor_type([], self.get_value_infos(sv_node.inputs[0]).dtype),
            input,
            axis=axis,
            keepdims=keepDims,
            mode=StringAttr.get("ArgMax"),
            loc=self.get_loc("{}_{}".format(sv_node.name, sv_node.opType)),
            ip=self.mlir.insert_point).output
        self.addOperand(sv_node.outputs[0], new_op.indices)

    def convert_Batchnorm_op(self, sv_node: SvNode):
        raise NotImplementedError("not implement yet")

    def convert_bitwise_op(self, sv_node: SvNode):
        raise NotImplementedError("not implement yet")

    def convert_bitwise_op(self, sv_node: SvNode):
        raise NotImplementedError("not implement yet")

    def convert_bitwise_op(self, sv_node: SvNode):
        raise NotImplementedError("not implement yet")

    def convert_broadcast_op(self, sv_node: SvNode):
        raise NotImplementedError("not implement yet")

    def convert_concat_op(self, sv_node: SvNode):
        raise NotImplementedError("not implement yet")

    def convert_pad_op(self, sv_node: SvNode):
        raise NotImplementedError("not implement yet")

    def convert_conv2d_op(self, sv_node: SvNode):
        raise NotImplementedError("not implement yet")

    def convert_conv1d_op(self, sv_node: SvNode):
        raise NotImplementedError("not implement yet")

    def convert_conv_transpose2d_op(self, sv_node: SvNode):
        raise NotImplementedError("not implement yet")


    def convert_conv_transpose1d_op(self, sv_node: SvNode):
        raise NotImplementedError("not implement yet")

    def convert_eltwise_op(self, sv_node: SvNode):
        raise NotImplementedError("not implement yet")

    def convert_compare_op(self, sv_node: SvNode):
        raise NotImplementedError("not implement yet")

    def convert_fc_op(self, sv_node: SvNode):
        raise NotImplementedError("not implement yet")


    def convert_mult_add_op(self, sv_node: SvNode):
        raise NotImplementedError("not implement yet")

    def convert_mult_acc_op(self, sv_node: SvNode):
        raise NotImplementedError("not implement yet")

    def convert_global_pool2d_op(self, sv_node: SvNode):
        raise NotImplementedError("not implement yet")

    def convert_interp3_op(self, sv_node: SvNode):
        raise NotImplementedError("not implement yet")

    def convert_local_conv_op(self, sv_node: SvNode):
        raise NotImplementedError("not implement yet")

    def convert_matmul_op(self, sv_node: SvNode):
        raise NotImplementedError("not implement yet")

    def convert_nms_op(self, sv_node: SvNode):
        raise NotImplementedError("not implement yet")

    def convert_permute_op(self, sv_node: SvNode):
        raise NotImplementedError("not implement yet")

    def convert_pixel_shuffle_op(self, sv_node: SvNode):
        raise NotImplementedError("not implement yet")

    def convert_pixel_unshuffle_op(self, sv_node: SvNode):
        raise NotImplementedError("not implement yet")

    def convert_pool2d_op(self, sv_node: SvNode):
        raise NotImplementedError("not implement yet")

    def convert_reduce_op(self, sv_node: SvNode):
        raise NotImplementedError("not implement yet")

    def convert_reduce_ext_op(self, sv_node: SvNode):
        raise NotImplementedError("not implement yet")

    def convert_reshape_op(self, sv_node: SvNode):
        raise NotImplementedError("not implement yet")

    def convert_rmsnorm_op(self, sv_node: SvNode):
        raise NotImplementedError("not implement yet")

    def convert_ln_op(self, sv_node: SvNode):
        raise NotImplementedError("not implement yet")

    def convert_slice_op(self, sv_node: SvNode):
        raise NotImplementedError("not implement yet")

    def convert_softmax_op(self, sv_node: SvNode):
        raise NotImplementedError("not implement yet")

    def convert_topk_op(self, sv_node: SvNode):
        raise NotImplementedError("not implement yet")

    def convert_yuv2rgb_op(self, sv_node: SvNode):
        raise NotImplementedError("not implement yet")

