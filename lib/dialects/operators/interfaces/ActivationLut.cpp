#include "support/module.h"


void ops::ActivationLutOp::shape_inference() { common_shape_inference(getOperation()); }

void ops::ActivationLutOp::type_inference() { common_type_inference(getOperation()); }
