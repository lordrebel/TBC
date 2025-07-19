
def HwDtypeToMlirDtype(hw_dtype):
    if(hw_dtype == 1):
        return "F16"
    elif(hw_dtype == 3):
        return "F8E4M3"
    elif(hw_dtype == 5):
        return "INT8"
    elif(hw_dtype == 7):
        return "INT32"
    elif(hw_dtype == 8):
        return "UINT16"

    raise RuntimeError("unknown hw dtype:{}".format(hw_dtype))

def HwstrDtypeToMlirDtype(hw_dtype_str):
    if(hw_dtype_str == "f16"):
        return "F16"
    elif(hw_dtype_str == "f8"):
        return "F8E4M3"
    elif(hw_dtype_str == "i8"):
        return "INT8"
    elif(hw_dtype_str == "i32"):
        return "INT32"
    elif(hw_dtype_str == "u16"):
        return "UINT16"

class SvTensorInfo:
    def __init__(self,name,dtype,shape,isPack=False):
        self.name = name
        self.dtype = dtype
        self.shape = shape
        self.isPack = isPack

class SvNode:
    def __init__(self, raw: dict):
        self.inputs=[]
        self.outputs=[]
        self.params=None
        self.data=None
        self.name=None
        self.opType=None
        self.hwType=None
        self.__parse(raw)

    def __parse(self, raw: dict):
        self.name = raw.get("inst_name", None)
        self.opType = raw.get("type_name", None)
        self.inputs = raw.get("bottom", [])
        self.outputs = raw.get("top", [])
        self.data = raw.get("data", None)
        self.params = raw.get("param", None)
        self.hwType = HwDtypeToMlirDtype(raw.get("hw", None))

class SvGraph:
    def __init__(self, raw: dict):
        self.nodes = []
        self.inputs = []
        self.outputs = []
        self.TensorInfos = {}
        self.__parse(raw)
    def __parse(self, raw: dict):
        self.inputs = raw.get("input_tensor", [])
        self.outputs = raw.get("output_tensor", [])

        self.__passIOTensorInfos(raw)

        for node in raw["layer"]:
            sv_node = SvNode(node)
            self.nodes.append(sv_node)
            for output in sv_node.outputs:
                output_dtype= HwDtypeToMlirDtype(sv_node.data["output_dtype"]) if sv_node.data and "output_dtype" in sv_node.data else sv_node.hwType
                if output not in self.TensorInfos:

                    self.TensorInfos[output] = SvTensorInfo(output, output_dtype, [], False)
                else:
                    self.TensorInfos[output].dtype =  output_dtype

    def __passIOTensorInfos(self,raw: dict):
        assert len(raw["input_tensor"]) == len(raw["input_shape"])
        assert len(raw["output_tensor"]) == len(raw["output_shape"])

        for tensorname,shape in zip(raw["input_tensor"], raw["input_shape"]):
            self.TensorInfos[tensorname] = SvTensorInfo(tensorname,HwstrDtypeToMlirDtype(shape["dtype"]),shape.get("dim",None),shape["pack"]==1)

        for tensorname,shape in zip(raw["output_tensor"], raw["output_shape"]):
            self.TensorInfos[tensorname] = SvTensorInfo(tensorname,HwstrDtypeToMlirDtype(shape["dtype"]),shape.get("dim",None),shape["pack"]==1)
