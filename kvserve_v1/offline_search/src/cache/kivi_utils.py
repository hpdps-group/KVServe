import torch
from typing import Optional, Any, List, Tuple, Dict
from collections import deque
from transformers.cache_utils import DynamicCache
from hqq.core.quantize import Quantizer as HQQQuantizer

class KIVICacheConfig:
    """
    Configuration class for quantized cache settings.

    Attributes:
        backend (`str`, *optional*, defaults to `"quanto"`):
            Backend to use when performing quantization, Can be one of [`quanto`, `HQQ`]
        nbits (`Optional[int]`, *optional*, defaults to 4):
            Number of bits, can be 2 or 4 for the `quanto` backend and one of [1, 2, 3, 4, 8] for the `HQQ` backend. Defaults to 2.
        axis_key (`int`, *optional*, defaults to 0):
            Axis over which to perform grouping for the key tensors. Can be [0, -1] for `quanto` backend and [0, 1] for `HQQ` backend.
        axis_value (`int`, *optional*, defaults to 0):
            Axis over which to perform grouping for the value tensors. Can be [0, -1] for `quanto` backend and [0, 1] for `HQQ` backend.
        q_group_size (`Optional[int]`, *optional*, defaults to 64):
            Size of the quantization group, should be a divisor of the model's hidden dimension.
            Defaults to 64.
        residual_length (`Optional[int]`, *optional*, defaults to 128):
            Length of the residual cache which will always be stored in original precision.
            Defaults to 128.
    """

    def __init__(
        self,
        nbits: Optional[int] = 4,
        axis_key: Optional[int] = 0,
        axis_value: Optional[int] = 0,
        q_group_size: Optional[int] = 32,
        residual_length: Optional[int] = 128,
        comp_cr: Optional[bool] = False,
    ):
        self.nbits = nbits
        self.axis_key = axis_key
        self.axis_value = axis_value
        self.q_group_size = q_group_size
        self.residual_length = residual_length
        self.comp_cr = comp_cr

        print("\n============== Using KIVICacheConfig... ==============")


class QuantizedCache(DynamicCache):
    """
    A quantizer cache similar to what is described in the [KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache paper](https://arxiv.org/abs/2402.02750).
    It allows the model to generate longer sequence length without allocating too much memory for Key and Value cache by applying quantization.

    The cache has two types of storage, one for original precision and one for the quantized cache. A `residual length` is set as a maximum capacity for the
    original precision cache. When the length goes beyond maximum capacity, the original precision cache is discarded and moved into the quantized cache. The
    quantization is done per-channel with a set `q_group_size` for both Keys and Values, in contrast to what was described in the paper.

    It stores Keys and Values a list of quantized tensors (tuples in case we need to store metadata), one for each layer. Additionally, it stores the Key and
    Value in original precision states as a list of tensors, one for each layer. The size of each tensor
    is `[batch_size, num_heads, seq_len - residual_length, head_dim]`
    """

    def __init__(self, cache_config: KIVICacheConfig) -> None:
        super().__init__()
        self._quantized_key_cache: List[torch.Tensor] = []
        self._quantized_value_cache: List[torch.Tensor] = []

        self.nbits = cache_config.nbits
        self.residual_length = cache_config.residual_length
        self.q_group_size = cache_config.q_group_size
        self.axis_key = cache_config.axis_key
        self.axis_value = cache_config.axis_value
        self.comp_cr = cache_config.comp_cr
        self.compression_ratio = 0

        super().__init__()

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Update the number of seen tokens
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]

        if len(self.key_cache) < layer_idx:
            raise ValueError("QuantizedCache does not support model usage where layers are skipped. Use DynamicCache.")
        elif len(self.key_cache) == layer_idx:
            seq_len = key_states.shape[-2]
            # compute the length to quantize
            quantize_len = (seq_len // self.residual_length) * self.residual_length

            if quantize_len == 0:
                key_to_quantize = torch.zeros(0, dtype=key_states.dtype, device=key_states.device)
                value_to_quantize = torch.zeros(0, dtype=value_states.dtype, device=value_states.device)
                key_residual = key_states
                value_residual = value_states
            elif quantize_len == seq_len:
                key_to_quantize = key_states
                value_to_quantize = value_states
                key_residual = torch.zeros(0, dtype=key_states.dtype, device=key_states.device)
                value_residual = torch.zeros(0, dtype=value_states.dtype, device=value_states.device)
            else:
                key_to_quantize = key_states[:, :, :quantize_len, :].contiguous()
                value_to_quantize = value_states[:, :, :quantize_len, :].contiguous()
                key_residual = key_states[:, :, quantize_len:, :].contiguous()
                value_residual = value_states[:, :, quantize_len:, :].contiguous()

            self._quantized_key_cache.append(self._quantize(key_to_quantize, axis=self.axis_key))
            self._quantized_value_cache.append(self._quantize(value_to_quantize, axis=self.axis_value))
            self.key_cache.append(key_residual)
            self.value_cache.append(value_residual)

            keys_to_return, values_to_return = key_states, value_states
        else:
            if self.comp_cr and self.compression_ratio == 0:
                self.compression_ratio = self.compute_compression_ratio(key_states.dtype, key_states.device)

            dequant_key = self._dequantize(self.axis_key, *self._quantized_key_cache[layer_idx])
            dequant_value = self._dequantize(self.axis_value, *self._quantized_value_cache[layer_idx])
            keys_to_return = [dequant_key, self.key_cache[layer_idx], key_states]
            values_to_return = [dequant_value, self.value_cache[layer_idx], value_states]

            keys_to_return = torch.cat(keys_to_return, dim=-2)
            values_to_return = torch.cat(values_to_return, dim=-2)
            if (
                self.key_cache[layer_idx].dim() == 4
                and self.key_cache[layer_idx].shape[-2] + 1 >= self.residual_length
            ):
                self._quantized_key_cache[layer_idx] = self._quantize(keys_to_return.contiguous(), axis=self.axis_key)
                self._quantized_value_cache[layer_idx] = self._quantize(
                    values_to_return.contiguous(), axis=self.axis_value
                )
                self.key_cache[layer_idx] = torch.zeros(0, dtype=key_states.dtype, device=key_states.device)
                self.value_cache[layer_idx] = torch.zeros(0, dtype=key_states.dtype, device=key_states.device)
            else:
                self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
                self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)

        return keys_to_return, values_to_return

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        if len(self.key_cache) <= layer_idx:
            return 0
        # since we cannot get the seq_length of each layer directly and rely on `_seen_tokens` which is
        # updated every "layer_idx" == 0, this is a hack to get the actual seq_length for the given layer_idx
        # this part of code otherwise fails when used to verify attn_weight shape in some models
        return self._seen_tokens if layer_idx == 0 else self._seen_tokens - 1

    def pack_to_float32(self, tensor: torch.Tensor, bits: int) -> torch.Tensor:
        """
        将 uint8 tensor 压缩打包存储到 float32 tensor 中。
        
        Args:
            tensor: 输入的 uint8 tensor (包含 4-bit 或 2-bit 数据)
            bits: 每个元素占用的位数 (支持 4 或 2)
        
        Returns:
            packed_tensor: dtype 为 float32 的压缩 tensor
        """
        assert bits in [2, 4], "目前只支持 2-bit 或 4-bit 压缩"
        
        # 1. 计算打包比例
        # 32位容器 / 4位 = 8 个数
        # 32位容器 / 2位 = 16 个数
        pack_ratio = 32 // bits 
        
        # 2. Flatten 拉平
        flat_tensor = tensor.flatten().to(torch.int32) # 转换为 int32 以便进行位移操作
        
        # 3. Padding (如果长度不能被整除)
        num_elements = flat_tensor.numel()
        padding = (pack_ratio - (num_elements % pack_ratio)) % pack_ratio
        if padding > 0:
            flat_tensor = torch.nn.functional.pad(flat_tensor, (0, padding), value=0)
        
        # 4. Reshape 为 (新长度, pack_ratio)
        # 例如 4-bit: [N, 8]
        reshaped = flat_tensor.view(-1, pack_ratio)
        
        # 5. 构造位移量 (Shift Vector)
        # 4-bit: [0, 4, 8, 12, 16, 20, 24, 28]
        # 注意：通常低位放第一个数还是高位放第一个数取决于你的解码习惯。
        # 这里采用 Little Endian 风格：第一个数在最低位。
        shift_vals = torch.arange(0, 32, bits, device=tensor.device, dtype=torch.int32)
        
        # 6. 执行位打包
        # 利用广播机制：reshaped << shift_vals
        shifted = reshaped << shift_vals
        
        # 按行进行 Bitwise OR (或者 Sum，因为位不重叠，Sum 等效于 OR)
        packed_int32 = torch.sum(shifted, dim=1).to(torch.int32)
        
        # 7. Reinterpret Cast (关键步骤)
        # 将 int32 的二进制位直接看作 float32，不改变位本身
        packed_float32 = packed_int32.view(torch.float32)
        
        return packed_float32

    def compute_compression_ratio(self, ori_dtype, device="cuda"):
        import pickle
        import sys
        import os
        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
        from offline_search.evaluation.compression_ratio.nvcomp_wrapper import to_device, TensorData

        num_layers = len(self._quantized_key_cache)
        if num_layers == 0:
            return 0.0

        meta_data = []
        original_key_tensors = []
        original_value_tensors = []
        compressed_key_tensors = None
        compressed_value_tensors = None

        for i in range(num_layers):
            keys_quantized, keys_meta_data = self._quantized_key_cache[i]
            values_quantized, values_meta_data = self._quantized_value_cache[i]

            keys_quantized = to_device(keys_quantized, device)
            values_quantized = to_device(values_quantized, device)
            keys_meta_data = to_device(keys_meta_data, device)
            values_meta_data = to_device(values_meta_data, device)
            
            original_key_tensors.append(torch.empty(keys_quantized.shape, dtype=ori_dtype, device=device))
            original_value_tensors.append(torch.empty(values_quantized.shape, dtype=ori_dtype, device=device))

            # Compressed size calculation
            if self.nbits in [2, 4]:
                keys_packed = self.pack_to_float32(keys_quantized, self.nbits).unsqueeze(0)
                values_packed = self.pack_to_float32(values_quantized, self.nbits).unsqueeze(0)
            else:
                keys_packed = keys_quantized.unsqueeze(0)
                values_packed = values_quantized.unsqueeze(0)

            if compressed_key_tensors is None:                 
                 current_len = keys_packed.shape[1]
                 final_shape = (num_layers, current_len)
                 compressed_key_tensors = torch.empty(final_shape, dtype=keys_packed.dtype, device=device)
                 compressed_value_tensors = torch.empty(final_shape, dtype=values_packed.dtype, device=device)

            compressed_key_tensors[i, ...] = keys_packed
            compressed_value_tensors[i, ...] = values_packed
            
            meta_data.append([keys_meta_data, values_meta_data])

        original_size = (len(pickle.dumps(original_key_tensors)) + len(pickle.dumps(original_value_tensors))) / 1024 / 1024
        
        del original_key_tensors, original_value_tensors
        
        meta_data = to_device(meta_data, device)
        
        compressed_data = TensorData(torch.cat([compressed_key_tensors, compressed_value_tensors], dim=0), meta_data)
        
        compressed_size = (
            len(pickle.dumps(compressed_data)) + 
            len(pickle.dumps(self.key_cache)) + 
            len(pickle.dumps(self.value_cache))
        ) / 1024 / 1024

        if compressed_size == 0:
            return 0.0
            
        return round(original_size / compressed_size, 4)

    def _quantize(self, tensor: torch.Tensor, axis: List[int]) -> Tuple[torch.Tensor, dict]:
        """Quantizes a key/value using a defined quantization method."""
        raise NotImplementedError("Make sure to implement `_quantize` in a subclass.")

    def _dequantize(self, q_tensor):
        """Dequantizes back the tensor that was quantized by `self._quantize()`"""
        raise NotImplementedError("Make sure to implement `_dequantize` in a subclass.")

class KIVICache(QuantizedCache):
    """
    Quantized Cache class that uses `HQQ` as a backend to perform quantization. Current implementation supports `int2`, `int4`, `int8` dtypes.

    Parameters:
        cache_config (`QuantizedCacheConfig`):
            A configuration containing all the arguments to be used by the quantizer, including axis, qtype and group size.

    Example:

        ```python
        >>> # Run pip install hqq first if you don't have it yet
        >>> from transformers import AutoTokenizer, AutoModelForCausalLM, HQQQuantizedCache, QuantizedCacheConfig

        >>> model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
        >>> tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")

        >>> inputs = tokenizer(text="My name is Qwen2", return_tensors="pt")

        >>> # Prepare a cache class and pass it to model's forward
        >>> cache_config = QuantizedCacheConfig(nbits=4, axis_key=1, axis_value=1)
        >>> past_key_values = HQQQuantizedCache(cache_config=cache_config)
        >>> outputs = model(**inputs, past_key_values=past_key_values, use_cache=True)
        >>> outputs.past_key_values # access cache filled with key/values from generation
        HQQQuantizedCache()
        ```
    """

    def __init__(self, cache_config: KIVICacheConfig) -> None:
        super().__init__(cache_config)
        if self.nbits not in [1, 2, 3, 4, 8]:
            raise ValueError(
                f"`nbits` for `HQQ` backend has to be one of [`1`, `2`, `3`, `4`, `8`] but got {self.nbits}"
            )

        if self.axis_key not in [0, 1, 2, 3]:
            raise ValueError(f"`axis_key` for `HQQ` backend has to be one of [`0`, `1`, `2`, `3`] but got {self.axis_key}")

        if self.axis_value not in [0, 1, 2, 3]:
            raise ValueError(f"`axis_value` for `HQQ` backend has to be one of [`0`, `1`, `2`, `3`] but got {self.axis_value}")

        self.quantizer = HQQQuantizer

    def _quantize(
        self,
        tensor: torch.Tensor, 
        axis: List[int], 
    ) -> Tuple[torch.Tensor, dict]:
        """
        Corrected quantization implementation.
        Maps [min, max] -> [0, max_value]
        """
        assert axis in [2, 3], "axis should be 2 or 3"
        if tensor.numel() == 0:
            return tensor, None

        batch_size, num_heads, seq_len, head_dim = tensor.shape
        if axis == 2:
            tensor = tensor.reshape(batch_size, num_heads, seq_len // self.q_group_size, self.q_group_size, head_dim)
            axis = -2
        elif axis == 3:
            tensor = tensor.reshape(batch_size, num_heads, seq_len, head_dim // self.q_group_size, self.q_group_size)
            axis = -1

        max_value = 2 ** self.nbits
        # 1. 计算 Min/Max
        max_ = torch.amax(tensor, dim=axis, keepdim=True)
        min_ = torch.amin(tensor, dim=axis, keepdim=True)

        # 2. 计算 Scale (添加 eps 防止除零)
        scale = (max_ - min_).clamp_(min=1e-5).div_(max_value - 1) 
        
        # 3. 量化核心逻辑
        # 公式： Q = clamp(round((x - min) / scale), 0, max_val)
        tensor_sub = tensor.sub(min_) # 此时 tensor_sub >= 0
        
        quant_float = tensor_sub.div_(scale) # In-place division
        
        # clamp 防止精度溢出导致的回绕
        quantized_tensor = quant_float.round_().clamp_(0, max_value).to(torch.uint8)

        # 4. Metadata
        meta_data = {
            "min_val": min_.to(tensor.dtype),
            "quant_scale": scale.to(tensor.dtype),
        }
        
        return quantized_tensor, meta_data

    def _dequantize(
        self, 
        axis: int,
        quantized_tensor: torch.Tensor,
        meta_data: dict,
    ) -> torch.Tensor:
        """
        Corrected dequantization.
        Formula: Real = Quantized * Scale + Min_Value
        """
        if meta_data is None:
            return quantized_tensor

        # 1. 获取元数据
        min_val = meta_data["min_val"]
        quant_scale = meta_data["quant_scale"]

        # 2. 类型转换 (Cast)
        quant_float = quantized_tensor.to(quant_scale.dtype)

        # 3. 反量化计算
        # result = x * s + b
        dequantized_tensor = quant_float * quant_scale + min_val

        if axis == 2:
            batch_size, num_heads, num_groups, group_size, head_dim = dequantized_tensor.shape
            dequantized_tensor = dequantized_tensor.reshape(batch_size, num_heads, num_groups * group_size, head_dim)
        elif axis == 3:
            batch_size, num_heads, seq_len, num_groups, group_size = dequantized_tensor.shape
            dequantized_tensor = dequantized_tensor.reshape(batch_size, num_heads, seq_len, num_groups * group_size)

        return dequantized_tensor
