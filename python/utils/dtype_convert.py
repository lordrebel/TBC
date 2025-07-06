import numpy as np
def float8e4m3fn_to_float32(uint8_array):
    """
    将 uint8 存储的 float8e4m3fn 转换为 float32

    float8e4m3fn 格式:
    1 bit: 符号位
    4 bits: 指数位 (biased by 7)
    3 bits: 尾数位

    参考: https://arxiv.org/abs/2209.05433
    """
    # 确保输入是 uint8 类型
    uint8_array = np.asarray(uint8_array, dtype=np.uint8)

    # 提取符号位、指数位和尾数位
    sign_bit = (uint8_array >> 7) & 0x1
    exp_bits = (uint8_array >> 3) & 0xF
    frac_bits = uint8_array & 0x7

    # 特殊值处理
    is_zero = (uint8_array & 0x7F) == 0  # 如果指数和尾数都是0，则为0
    is_nan = (exp_bits == 0xF) & (frac_bits != 0)  # NaN
    is_inf = (exp_bits == 0xF) & (frac_bits == 0)  # Infinity

    # 计算浮点值
    result = np.zeros_like(uint8_array, dtype=np.float32)

    # 处理正常值
    normal_mask = ~(is_zero | is_nan | is_inf)
    if np.any(normal_mask):
        # 指数偏移值为7
        exponent = exp_bits[normal_mask].astype(np.int32) - 7
        # 尾数 (隐含的前导1)
        fraction = 1.0 + (frac_bits[normal_mask].astype(np.float32) / 8.0)
        # 计算值
        value = fraction * (2.0 ** exponent)
        # 应用符号位
        result[normal_mask] = np.where(sign_bit[normal_mask], -value, value)

    # 处理特殊值
    result[is_nan] = np.nan
    result[is_inf & sign_bit] = -np.inf
    result[is_inf & ~sign_bit] = np.inf
    # 零值已经在初始化时设置

    return result

def float8e5m2_to_float32(uint8_array):
    """
    将 uint8 存储的 float8e5m2 转换为 float32

    float8e5m2 格式:
    1 bit: 符号位
    5 bits: 指数位 (biased by 15)
    2 bits: 尾数位

    参考: https://arxiv.org/abs/2209.05433
    """
    # 确保输入是 uint8 类型
    uint8_array = np.asarray(uint8_array, dtype=np.uint8)

    # 提取符号位、指数位和尾数位
    sign_bit = (uint8_array >> 7) & 0x1
    exp_bits = (uint8_array >> 2) & 0x1F
    frac_bits = uint8_array & 0x3

    # 特殊值处理
    is_zero = (uint8_array & 0x7F) == 0  # 如果指数和尾数都是0，则为0
    is_nan = (exp_bits == 0x1F) & (frac_bits != 0)  # NaN
    is_inf = (exp_bits == 0x1F) & (frac_bits == 0)  # Infinity

    # 计算浮点值
    result = np.zeros_like(uint8_array, dtype=np.float32)

    # 处理正常值
    normal_mask = ~(is_zero | is_nan | is_inf)
    if np.any(normal_mask):
        # 指数偏移值为15
        exponent = exp_bits[normal_mask].astype(np.int32) - 15
        # 尾数 (隐含的前导1)
        fraction = 1.0 + (frac_bits[normal_mask].astype(np.float32) / 4.0)
        # 计算值
        value = fraction * (2.0 ** exponent)
        # 应用符号位
        result[normal_mask] = np.where(sign_bit[normal_mask], -value, value)

    # 处理特殊值
    result[is_nan] = np.nan
    result[is_inf & sign_bit] = -np.inf
    result[is_inf & ~sign_bit] = np.inf
    # 零值已经在初始化时设置

    return result

def float32_to_float8e4m3fn(float32_array):
    """
    将 float32 转换为 uint8 存储的 float8e4m3fn

    float8e4m3fn 格式:
    1 bit: 符号位
    4 bits: 指数位 (biased by 7)
    3 bits: 尾数位

    参考: https://arxiv.org/abs/2209.05433
    """
    # 确保输入是 float32 类型
    float32_array = np.asarray(float32_array, dtype=np.float32)

    # 创建输出数组
    result = np.zeros(float32_array.shape, dtype=np.uint8)

    # 处理特殊值：0, inf, nan
    is_zero = np.isclose(float32_array, 0.0, atol=1e-10)
    is_inf = np.isinf(float32_array)
    is_nan = np.isnan(float32_array)

    # 提取符号位 (1 表示负数)
    sign_bit = np.signbit(float32_array).astype(np.uint8)

    # 处理正常值
    normal_mask = ~(is_zero | is_inf | is_nan)
    if np.any(normal_mask):
        # 取绝对值
        abs_values = np.abs(float32_array[normal_mask])

        # 计算指数和尾数
        # log2(x) 给出以2为底的对数，可以用来确定指数
        log2_values = np.log2(abs_values)
        exponents = np.floor(log2_values).astype(np.int32)

        # 确保指数在有效范围内 (-7 到 7)，对应于有偏指数 0 到 14
        # float8e4m3fn 的指数偏移值为 7
        clipped_exponents = np.clip(exponents, -7, 7)
        biased_exponents = (clipped_exponents + 7).astype(np.uint8)

        # 计算尾数
        # 将值规范化到 [1.0, 2.0) 范围
        normalized_values = abs_values / (2.0 ** clipped_exponents)
        # 提取尾数的3位 (0 到 7)
        fractions = np.round((normalized_values - 1.0) * 8.0).astype(np.uint8)
        fractions = np.clip(fractions, 0, 7)  # 确保在有效范围内

        # 组合符号位、指数位和尾数位
        result[normal_mask] = (sign_bit[normal_mask] << 7) | (biased_exponents << 3) | fractions

    # 处理特殊值
    # 零: 所有位为0
    # result[is_zero] = 0  # 已经初始化为0

    # 无穷大: 指数全为1，尾数为0
    inf_value = 0xF8  # 指数位全1 (0xF << 3)，尾数位全0
    result[is_inf & ~sign_bit] = inf_value  # 正无穷
    result[is_inf & sign_bit] = inf_value | 0x80  # 负无穷 (设置符号位)

    # NaN: 指数全为1，尾数非0
    nan_value = 0xFC  # 指数位全1 (0xF << 3)，尾数位非0 (这里用 0x4)
    result[is_nan] = nan_value

    return result

def float32_to_float8e5m2(float32_array):
    """
    将 float32 转换为 uint8 存储的 float8e5m2

    float8e5m2 格式:
    1 bit: 符号位
    5 bits: 指数位 (biased by 15)
    2 bits: 尾数位

    参考: https://arxiv.org/abs/2209.05433
    """
    # 确保输入是 float32 类型
    float32_array = np.asarray(float32_array, dtype=np.float32)

    # 创建输出数组
    result = np.zeros(float32_array.shape, dtype=np.uint8)

    # 处理特殊值：0, inf, nan
    is_zero = np.isclose(float32_array, 0.0, atol=1e-10)
    is_inf = np.isinf(float32_array)
    is_nan = np.isnan(float32_array)

    # 提取符号位 (1 表示负数)
    sign_bit = np.signbit(float32_array).astype(np.uint8)

    # 处理正常值
    normal_mask = ~(is_zero | is_inf | is_nan)
    if np.any(normal_mask):
        # 取绝对值
        abs_values = np.abs(float32_array[normal_mask])

        # 计算指数和尾数
        log2_values = np.log2(abs_values)
        exponents = np.floor(log2_values).astype(np.int32)

        # 确保指数在有效范围内 (-15 到 15)，对应于有偏指数 0 到 30
        # float8e5m2 的指数偏移值为 15
        clipped_exponents = np.clip(exponents, -15, 15)
        biased_exponents = (clipped_exponents + 15).astype(np.uint8)

        # 计算尾数
        # 将值规范化到 [1.0, 2.0) 范围
        normalized_values = abs_values / (2.0 ** clipped_exponents)
        # 提取尾数的2位 (0 到 3)
        fractions = np.round((normalized_values - 1.0) * 4.0).astype(np.uint8)
        fractions = np.clip(fractions, 0, 3)  # 确保在有效范围内

        # 组合符号位、指数位和尾数位
        result[normal_mask] = (sign_bit[normal_mask] << 7) | (biased_exponents << 2) | fractions

    # 处理特殊值
    # 零: 所有位为0
    # result[is_zero] = 0  # 已经初始化为0

    # 无穷大: 指数全为1，尾数为0
    inf_value = 0x7C  # 指数位全1 (0x1F << 2)，尾数位全0
    result[is_inf & ~sign_bit] = inf_value  # 正无穷
    result[is_inf & sign_bit] = inf_value | 0x80  # 负无穷 (设置符号位)

    # NaN: 指数全为1，尾数非0
    nan_value = 0x7E  # 指数位全1 (0x1F << 2)，尾数位非0 (这里用 0x2)
    result[is_nan] = nan_value

    return result

def bf16_to_float32(uint16_array):
    """
    将 uint16 存储的 bfloat16 (Brain Floating Point) 转换为 float32

    bfloat16 格式:
    1 bit: 符号位
    8 bits: 指数位 (与float32相同)
    7 bits: 尾数位 (float32的尾数前7位)

    这个格式保留了float32的动态范围，但精度降低了
    """
    # 确保输入是 uint16 类型
    uint16_array = np.asarray(uint16_array, dtype=np.uint16)

    # 将 uint16 转换为 uint32，左移 16 位
    # 这样 bf16 的位模式就会被放在 float32 的高 16 位上
    uint32_array = np.uint32(uint16_array) << 16

    # 使用 view 将 uint32 数组解释为 float32
    return uint32_array.view(np.float32)

def float32_to_bf16(float32_array):
    """
    将 float32 转换为 uint16 存储的 bfloat16

    这个函数执行的是 bf16_to_float32 的逆操作
    """
    # 确保输入是 float32 类型
    float32_array = np.asarray(float32_array, dtype=np.float32)

    # 将 float32 解释为 uint32
    uint32_array = float32_array.view(np.uint32)

    # 右移 16 位，只保留高 16 位，然后转换为 uint16
    return np.uint16(uint32_array >> 16)
