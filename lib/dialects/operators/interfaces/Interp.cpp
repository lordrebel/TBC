//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "support/module.h"

using namespace std;

void ops::InterpOp::shape_inference() {
    auto in_shape = module::getShape(getInput());
    auto target_shape_ = getTargetShape();
    auto scale_h_ = getScaleH().convertToDouble();
    auto scale_w_ = getScaleW().convertToDouble();
    auto nof_dims = in_shape.size();
    int32_t hidx = nof_dims >= 4 ? nof_dims - 2 : -1;
    int32_t widx = nof_dims - 1;
    std::vector<int64_t> out_shape(in_shape);
    if (getMode() == "nearest" || getMode() == "linear") {
        if (!module::isNone(target_shape_)) {
            auto target_shape = target_shape_.getDefiningOp<ops::WeightOp>().read<float>();
            if (nof_dims == 5) {
                assert(target_shape->at(0) == in_shape[nof_dims - 3]); // upsample_nearest_3d only support scale_d = 1
            }
            out_shape[widx] = (int)target_shape->at(nof_dims - 2 - 1);
            if (nof_dims >= 4) {
                out_shape[hidx] =
                    nof_dims >= 4 ? (int)target_shape->at(nof_dims - 2 - 2) : 1;
                setScaleH(APFloat((double)out_shape[hidx] / in_shape[hidx]));
            } else {
                setScaleH(APFloat(1.0));
            }
            setScaleW(APFloat((double)out_shape[widx] / in_shape[widx]));
            setOperand(1, module::getNoneOp(getOperation()));
        } else if (nof_dims >= 4 && scale_h_ > 0 && scale_w_ > 0) {
            out_shape[hidx] = floor(out_shape[hidx] * scale_h_);
            out_shape[widx] = floor(out_shape[widx] * scale_w_);
        } else if (nof_dims >= 3 && scale_w_ > 0){
            out_shape[widx] = floor(out_shape[widx] * scale_w_);
        } else {
            llvm::errs() << "You must specify either target shape or scale.\n";
            llvm_unreachable("You must specify either target shape or scale.\n");
        }
        module::setShapeOrVerify(getOutput(), out_shape);
    } else {
        llvm_unreachable("Unsupport Interp mode type.\n");
    }
}

void ops::InterpOp::type_inference() {
    module::setElementType(getOutput(), module::getElementType(getInput()));
}
