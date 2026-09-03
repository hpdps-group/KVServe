import torch
from typing import Optional, Any, List, Tuple
from collections import deque
from transformers.cache_utils import DynamicCache

class CacheGenCacheConfig:

    def __init__(
        self,

        model_layers: Optional[int] = None,
        quantization_level: Optional[int] = 1,
        high_max_value: Optional[int] = 32,
        mid_max_value: Optional[int] = 16,
        low_max_value: Optional[int] = 12,
        comp_cr: Optional[bool] = False,

    ):
        self.model_layers = model_layers
        self.quantization_level = quantization_level
        self.high_max_value = high_max_value
        self.mid_max_value = mid_max_value
        self.low_max_value = low_max_value
        self.comp_cr = comp_cr
        self.validate()

        print("\n============== Using CacheGenCacheConfig... ==============\n")

    def validate(self):
        """Validates if the arguments passed are correct"""

        # assert self.scores is not None, "scores should be provided"
        assert self.model_layers is not None, "model_layers should be provided"
        assert self.quantization_level in [1, 2, 3], "quantization_level should be in [1, 2, 3]"

        assert self.high_max_value < 256, "high_max_value should be between 1 and 255"
        assert self.high_max_value >= self.mid_max_value, "high_max_value should be greater than mid_max_value"
        assert self.mid_max_value >= self.low_max_value, "mid_max_value should be greater than low_max_value"
        assert self.low_max_value > 0, "low_max_value should be between 1 and 255"


class CacheGenCache(DynamicCache):

    def __init__(self, cache_config: CacheGenCacheConfig) -> None:
        super().__init__()

        self._quantized_key_cache: List[torch.Tensor] = []
        self._quantized_value_cache: List[torch.Tensor] = []

        self.model_layers = cache_config.model_layers
        self.quantization_level = cache_config.quantization_level
        self.high_max_value = cache_config.high_max_value
        self.mid_max_value = cache_config.mid_max_value
        self.low_max_value = cache_config.low_max_value
        self.comp_cr = cache_config.comp_cr

        self.layer_max_value = self.make_quantized_bins()
        self.compression_ratio = 0

        super().__init__()

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Update the number of seen tokens
        # print("Updating CustomCache...")
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]

        if len(self.key_cache) < layer_idx:
            raise ValueError("Does not support model usage where layers are skipped. Use DynamicCache.")
        # prefill
        elif len(self.key_cache) == layer_idx:

            self._quantized_key_cache.append(self._quantize(self.layer_max_value[layer_idx]["k"], key_states, [1, 3]))
            self._quantized_value_cache.append(self._quantize(self.layer_max_value[layer_idx]["v"], value_states, [1, 3]))
            self.key_cache.append(torch.zeros(0, dtype=key_states.dtype, device=key_states.device))
            self.value_cache.append(torch.zeros(0, dtype=key_states.dtype, device=key_states.device))            
            keys_to_return, values_to_return = key_states, value_states

        # decode
        else:
             # compute compression ratio only once
            if self.comp_cr and self.compression_ratio == 0:
                self.compression_ratio = self.compute_compression_ratio(key_states.dtype, key_states.device)

            dequant_key = self._dequantize(self.layer_max_value[layer_idx]["k"], *self._quantized_key_cache[layer_idx])
            dequant_value = self._dequantize(self.layer_max_value[layer_idx]["v"], *self._quantized_value_cache[layer_idx])

            # 更新decode阶段kvcache
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)

            keys_to_return = torch.cat([dequant_key, self.key_cache[layer_idx]], dim=-2)
            values_to_return = torch.cat([dequant_value, self.value_cache[layer_idx]], dim=-2)

        return keys_to_return, values_to_return
    
    def _quantize(
        self, 
        max_value: int,
        tensor: torch.Tensor, 
        axis: List[int], 
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Quantizes a key/value using a defined quantization method."""

        MAX = (max_value // 2 - 1)
        max1 = torch.amax(torch.abs(tensor), dim=axis, keepdim=True)
        factor = MAX / max1
        xq = torch.round(tensor * factor + MAX).to(torch.int8)
        
        return xq, max1

    def _dequantize(
        self, 
        max_value: int,
        quantized_tensor: torch.Tensor,
        max1: torch.Tensor,
    ) -> torch.Tensor:
        """Dequantizes back the tensor that was quantized by `self._quantize()`"""
        C = (max_value // 2 - 1)
        t = quantized_tensor - C
        t = t / C
        t = t * max1
        return t.to(max1.dtype)

    # def _quantize(
    #     self,
    #     max_value: int,
    #     tensor: torch.Tensor, 
    #     axis: List[int], 
    # ) -> Tuple[torch.Tensor, dict]:
    #     """
    #     Corrected quantization implementation.
    #     Maps [min, max] -> [0, max_value]
    #     """
    #     if tensor.numel() == 0:
    #         return tensor.to(torch.uint8), None

    #     # 1. 计算 Min/Max
    #     max_ = torch.amax(tensor, dim=axis, keepdim=True)
    #     min_ = torch.amin(tensor, dim=axis, keepdim=True)

    #     # 2. 计算 Scale (添加 eps 防止除零)
    #     scale = (max_ - min_).clamp_(min=1e-5).div_(max_value - 1) 
        
    #     # 3. 量化核心逻辑
    #     # 公式： Q = clamp(round((x - min) / scale), 0, max_val)
    #     tensor_sub = tensor.sub(min_) # 此时 tensor_sub >= 0
        
    #     quant_float = tensor_sub.div_(scale) # In-place division
        
    #     # clamp 防止精度溢出导致的回绕
    #     quantized_tensor = quant_float.round_().clamp_(0, max_value).to(torch.uint8)

    #     # 4. Metadata
    #     meta_data = {
    #         "min_val": min_.to(tensor.dtype),
    #         "quant_scale": scale.to(tensor.dtype),
    #     }
        
    #     return quantized_tensor, meta_data

    # def _dequantize(
    #     self, 
    #     max_value: int,
    #     quantized_tensor: torch.Tensor,
    #     meta_data: dict,
    # ) -> torch.Tensor:
    #     """
    #     Corrected dequantization.
    #     Formula: Real = Quantized * Scale + Min_Value
    #     """
    #     if meta_data is None:
    #         return quantized_tensor.to(self.key_cache[0].dtype)

    #     # 1. 获取元数据
    #     min_val = meta_data["min_val"]
    #     quant_scale = meta_data["quant_scale"]

    #     # 2. 类型转换 (Cast)
    #     quant_float = quantized_tensor.to(quant_scale.dtype)

    #     # 3. 反量化计算
    #     # result = x * s + b
    #     dequantized_tensor = quant_float * quant_scale + min_val

    #     return dequantized_tensor


    def make_quantized_bins(self):
        """
        根据模型层数和量化等级，自动计算每一层的量化最大值。
        结果存储在 self.layer_max_value 中，格式为 list[dict] -> [{'k': k_val, 'v': v_val}, ...]
        """

        layer_max_value = []

        # 定义层级划分边界 (参考 CacheGen 论文/默认配置的比例)
        # Key Layers: 约前 1/3, 中间 1/3, 后 1/3
        key_first_boundary = int(self.model_layers * 0.32)
        key_second_boundary = int(self.model_layers * 0.63)
        
        # Value Layers: 只有最前几层 (约 15%, 至少 2 层) 需要较高精度
        val_first_boundary = max(2, int(self.model_layers * 0.15))

        for i in range(self.model_layers):
            # --- Key (K) 量化策略 ---
            if self.quantization_level == 1: # Level 1: 激进压缩
                # 前 ~2/3 层使用 Mid，后 ~1/3 使用 Low
                k_max = self.mid_max_value if i < key_second_boundary else self.low_max_value
                # v_max = self.mid_max_value if i < key_second_boundary else self.low_max_value
            elif self.quantization_level == 2: # Level 2: 中等压缩
                # 前 ~1/3 层使用 High，其余使用 Mid
                k_max = self.high_max_value if i < key_first_boundary else self.mid_max_value
                # v_max = self.high_max_value if i < key_first_boundary else self.mid_max_value
            else: # Level 3: 保守/默认
                k_max = self.high_max_value
                # v_max = self.high_max_value

            # --- Value (V) 量化策略 ---
            if self.quantization_level == 1: # Level 1: 激进压缩
                # 头部层使用 Mid，其余使用 Low
                v_max = self.mid_max_value if i < val_first_boundary else self.low_max_value
            elif self.quantization_level == 2: # Level 2: 中等压缩
                # 头部层使用 High，其余使用 Mid
                v_max = self.high_max_value if i < val_first_boundary else self.mid_max_value
            else: # Level 3
                v_max = self.high_max_value

            layer_max_value.append({"k": k_max, "v": v_max})

        return layer_max_value

    def compute_compression_ratio(self, ori_dtype, device="cuda"):
        import pickle
        import sys
        import os
        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
        from offline_search.evaluation.compression_ratio.cachegen_wrapper import CacheGenWrapper
        from offline_search.evaluation.compression_ratio.nvcomp_wrapper import to_device

        num_layers = len(self._quantized_key_cache)
        if num_layers == 0:
            return 0.0
        
        meta_data = []
        original_key_tensors = []
        original_value_tensors = []
        to_compressed_key_tensors = None
        to_compressed_value_tensors = None
        current_idx = 0

        for i in range(num_layers):
            keys_quantized, keys_meta_data = self._quantized_key_cache[i]
            values_quantized, values_meta_data = self._quantized_value_cache[i]

            keys_quantized = to_device(keys_quantized, device)
            values_quantized = to_device(values_quantized, device)
            keys_meta_data = to_device(keys_meta_data, device)
            values_meta_data = to_device(values_meta_data, device)

            # Original size calculation
            original_key_tensors.append(torch.empty(keys_quantized.shape, dtype=ori_dtype, device=device))
            original_value_tensors.append(torch.empty(values_quantized.shape, dtype=ori_dtype, device=device))

            # Concatenation
            if to_compressed_key_tensors is None:
                batch_size, heads, tokens, channels = keys_quantized.shape
                final_shape = (num_layers * batch_size, heads, tokens, channels)
                to_compressed_key_tensors = torch.empty(final_shape, dtype=keys_quantized.dtype, device=device)
                to_compressed_value_tensors = torch.empty(final_shape, dtype=values_quantized.dtype, device=device)

            end_idx = current_idx + keys_quantized.shape[0]
            to_compressed_key_tensors[current_idx:end_idx, ...] = keys_quantized
            to_compressed_value_tensors[current_idx:end_idx, ...] = values_quantized
            current_idx = end_idx

            meta_data.append([keys_meta_data, values_meta_data])

        original_size = (len(pickle.dumps(original_key_tensors)) + len(pickle.dumps(original_value_tensors))) / 1024 / 1024
        
        del original_key_tensors, original_value_tensors
        
        meta_data = to_device(meta_data, device)
        cachegen_wrapper = CacheGenWrapper(
            quantized_keys=to_compressed_key_tensors,
            quantized_values=to_compressed_value_tensors,
            meta_data=meta_data,
        )
        packed_data = cachegen_wrapper.compress()
        compressed_size = len(pickle.dumps(packed_data)) / 1024 / 1024
        
        if compressed_size == 0:
            return 0.0
            
        return round(original_size / compressed_size, 4)
