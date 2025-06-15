
import shutil
import os
import subprocess
import logging

compile_phase=[
    "IMPORTED",
"OPERATOR_OPTED",
"KERNEL",
"KERNEL_OPTED",
"HAL",
"HAL_OPTED",
"HAL_ADDRESSED",
"CODEGEN",
]
def _os_system_log(cmd_str):
    # use subprocess to redirect the output stream
    # the file for saving the output stream should be set if using this function
    logging.info("[Running]: %s", cmd_str)

    process = subprocess.Popen(cmd_str,
                               shell=True,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT,
                               universal_newlines=True)

    while True:
        output = process.stdout.readline().strip()
        if output == '' and process.poll() is not None:
            break
        if output:
            logging.info(output)

    process.wait()
    ret = process.returncode

    if ret == 0:
        logging.info("[Success]: %s", cmd_str)
    else:
        raise RuntimeError("[!Error]: {}".format(cmd_str))


def _os_system(cmd: list, save_log: bool = False):
    cmd_str = ""
    for s in cmd:
        cmd_str += str(s) + " "
    if not save_log:
        print("[Running]: {}".format(cmd_str))
        ret = os.system(cmd_str)
        if ret == 0:
            print("[Success]: {}".format(cmd_str))
        else:
            raise RuntimeError("[!Error]: {}".format(cmd_str))
    else:
        _os_system_log(cmd_str)


def mlir_opt_for_operator(mlirfile, opt_mlirfile):
    cmd = ["tbc-opt", mlirfile, "--shape-infer"]
    cmd.extend(["--canonicalize"," --assign-compile-phase=\"compile_phase=OPERATOR_OPTED\" " ,"-o", opt_mlirfile])
    _os_system(cmd)


def mlir_lowering(top_mlir: str,
                  tpu_mlir: str,
                  mode: str,
                  chip: str,
                  cali_table: str = None,
                  asymmetric: bool = False,
                  quantize_table: str = None,
                  customization_format: str = None,
                  fuse_preprocess: bool = False,
                  aligned_input: bool = False,
                  ignore_f16_overflow: bool = False,
                  do_winograd: bool = False,
                  q_group_size: int = 0):
    pass




def mlir_to_model(tpu_mlir: str,
                  model: str,
                  final_mlir: str,
                  dynamic: bool = False,
                  quant_input: bool = False,
                  quant_output: bool = False,
                  quant_input_list: str = "",
                  quant_output_list: str = "",
                  disable_layer_group: bool = False,
                  opt: int = 2,
                  merge_weight: bool = False,
                  op_divide: bool = False,
                  num_device: int = 1,
                  num_core: int = 1,
                  embed_debug_info: bool = False,
                  model_version: str = ""):
    pass


# OperatorTOTOSA
def operator_to_tosa(top_mlir: str,
                tosa_mlir: str,
                includeWeight: bool = False):
    cmd = ["tpuc-opt", top_mlir]
    lower_param = "--convert-top-to-tosa=\"includeWeight="
    if includeWeight:
        lower_param += "True\""
    else:
        lower_param += "False\""
    cmd.extend([
        lower_param,
        "--canonicalize",
        "-o",
        tosa_mlir
    ])
    _os_system(cmd)

# TOSATOObj
def tosa_to_llvm(tosa_mlir: str,
                 objfile: str):
    cmd = ["mlir-opt", tosa_mlir]
    lower_param = ("--pass-pipeline=\"builtin.module("
                   "func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-arith, tosa-to-tensor, tosa-to-scf), "
                   "convert-tensor-to-linalg, "
                   "func.func(canonicalize, linalg-bufferize, convert-linalg-to-affine-loops, affine-loop-fusion, affine-simplify-structures, lower-affine), "
                   "func-bufferize, "
                   "func.func(tensor-bufferize, llvm-request-c-wrappers), "
                   "arith-expand, arith-bufferize, normalize-memrefs, convert-scf-to-cf, "
                   "convert-math-to-llvm, convert-arith-to-llvm, convert-func-to-llvm, convert-cf-to-llvm, "
                   "convert-bufferization-to-memref, memref-expand, expand-strided-metadata, finalize-memref-to-llvm, "
                   "canonicalize, llvm-legalize-for-export, reconcile-unrealized-casts)\""
                   "| mlir-translate --mlir-to-llvmir "
                   "| llc -mtriple=x86_64-unknown-linux-gnu --filetype=obj")
    cmd.extend([
        lower_param,
        "-o",
        objfile
    ])
    _os_system(cmd)



# Extra tool: delete file in current directory
def delete_file(file: str):
    cmd = ["rm -f", file]
    _os_system(cmd)
