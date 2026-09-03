import torch
import torch.nn as nn
import math
import fast_hadamard_transform
from typing import Optional, Any, List, Tuple, Dict
from collections import deque
from transformers.cache_utils import DynamicCache, DynamicLayer

class NoneTransform:
    """
    None transform.
    """
    def __init__(self):
        pass

    def transform(self, kv: torch.Tensor, layer_idx: int, **kwargs) -> torch.Tensor:
        return kv

    def inverse(self, transformed_kv: torch.Tensor, layer_idx: int, **kwargs) -> torch.Tensor:
        return transformed_kv

class HadamardTransform:
    """
    Vectorized Hadamard Transform.
    """

    def __init__(self, base_seed: int = 0xC0FEBABE):
        self.base_seed = base_seed
        # 缓存 signs 张量，避免重复的 CPU 生成和 Host-to-Device 传输
        self.signs_cache = {}

    def _get_seed(self, layer_idx: int, head_idx: int) -> int:
        return self.base_seed ^ (layer_idx << 16) ^ head_idx

    def _pow2_chunks(self, dim: int) -> List[int]:
        """Split dim into a list of descending powers-of-two."""
        chunks: List[int] = []
        remaining = dim
        while remaining > 0:
            chunk = 1 << (remaining.bit_length() - 1)
            chunks.append(chunk)
            remaining -= chunk
        return chunks

    def _fwht_in_chunks(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply FWHT to the last dimension of an arbitrary shaped tensor.
        If head_dim is not a power of two, it splits into chunks.
        
        Args:
            x: Tensor of shape [..., head_dim]
        """
        chunks = self._pow2_chunks(x.shape[-1])
        outputs = []
        start = 0
        for size in chunks:
            part = x.narrow(-1, start, size)

            transformed = fast_hadamard_transform.hadamard_transform(
                part.contiguous(), scale=1.0 / math.sqrt(size)
            )
            outputs.append(transformed)
            start += size
        
        return torch.cat(outputs, dim=-1)

    def get_rademacher_signs(self, 
                             num_heads: int, 
                             head_dim: int, 
                             layer_idx: int, 
                             device: torch.device, 
                             dtype: torch.dtype) -> torch.Tensor:
        """
        Generate the Rademacher sign tensor for ALL heads in a layer.
        
        Returns:
            signs: Tensor of shape [1, num_heads, 1, head_dim] for broadcasting.
        """
        # 1. 检查缓存
        # 包含 device 和 dtype 以防止上下文变化
        cache_key = (layer_idx, num_heads, head_dim, device, dtype)
        if cache_key in self.signs_cache:
            return self.signs_cache[cache_key]

        signs_list = []

        gen = torch.Generator(device="cpu")
        
        for h in range(num_heads):
            seed = self._get_seed(layer_idx, h)
            gen.manual_seed(seed)
            
            # 生成 {0, 1} -> 转为 {-1, 1}
            s = torch.randint(0, 2, (head_dim,), generator=gen, dtype=torch.int8)
            s = s.float().mul_(2).sub_(1)
            signs_list.append(s)
            
        # Stack heads -> [num_heads, head_dim]
        signs = torch.stack(signs_list).to(device=device, dtype=dtype)
        
        # Reshape for broadcasting: [1, num_heads, 1, head_dim]
        # 这样可以直接与 [bsz, num_heads, seq_len, head_dim] 相乘
        signs = signs.view(1, num_heads, 1, head_dim)
        
        # 2. 存入缓存
        self.signs_cache[cache_key] = signs
        
        return signs

    def transform(self, kv: torch.Tensor, layer_idx: int, **kwargs) -> torch.Tensor:
        """
        Apply Hadamard transform to the KV tensor in-place or out-of-place.
        
        Args:
            kv: Input tensor [bsz, num_heads, seq_len, head_dim]
            layer_idx: Current layer index for seed generation
            
        Returns:
            transformed_kv: Tensor of same shape [bsz, num_heads, seq_len, head_dim]
        """
        assert kv.dim() == 4, f"Expected 4D tensor [bsz, heads, seq, dim], got {kv.shape}"
        bsz, num_heads, seq_len, head_dim = kv.shape

        # 1. 获取所有 Heads 的随机符号向量 (Shape: [1, num_heads, 1, head_dim])
        # 如果需要极致性能，可以将这个 signs 缓存起来，避免每次 forward 都重新生成
        signs = self.get_rademacher_signs(num_heads, head_dim, layer_idx, kv.device, kv.dtype)

        # 2. Element-wise 乘法 (利用广播机制)
        # [bsz, H, S, D] * [1, H, 1, D] -> [bsz, H, S, D]
        signed_kv = kv * signs

        # 3. Apply FWHT (Batched)
        # 库函数通常会自动处理 batch 维度，只要对最后一维操作即可
        output = self._fwht_in_chunks(signed_kv)
        return output

    def inverse(self, transformed_kv: torch.Tensor, layer_idx: int, **kwargs) -> torch.Tensor:
        """
        Inverse Hadamard transform.
        Logic: Inverse(H * S * x) = S * Inverse(H) * (H * S * x) 
               Since H is symmetric orthogonal (scaled), H^{-1} is proportional to H.
               And S is its own inverse (1/1=1, 1/-1=-1).
               So, steps are: 1. FWHT, 2. Multiply by S.
        """
        assert transformed_kv.dim() == 4
        bsz, num_heads, seq_len, head_dim = transformed_kv.shape

        # 1. Apply FWHT first (Hadamard matrix is symmetric)
        rotated_back = self._fwht_in_chunks(transformed_kv)

        # 2. Multiply by signs
        signs = self.get_rademacher_signs(num_heads, head_dim, layer_idx, transformed_kv.device, transformed_kv.dtype)
        original_kv = rotated_back * signs

        return original_kv

class AffineTransform(nn.Module):
    """
    Affine transform manager for all layers.
    Supports loading parameters for multiple layers and applying
    transform/inverse by specifying layer_idx.
    """

    def __init__(
        self,
        head_dim: int,
        mode: str = "diag",
        learnable_clip: bool = True,
        clip_init: float = 4.0,
        params_path: Optional[str] = None,
        device: str = "cuda"
    ) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.mode = mode
        assert mode in ("diag", "full")
        self.learnable_clip = learnable_clip
        self.clip_init = clip_init
        self.device = device
        
        # Storage for layer parameters
        # We use a dict to store parameters for each layer
        # Keys are layer_idx, values are dicts of tensors (on device)
        self.layer_params: Dict[int, Dict[str, torch.Tensor]] = {}
        
        if params_path:
            self.load_parameters(params_path)

    def load_parameters(self, path: str):
        try:
            loaded = torch.load(path, map_location=self.device)
            # print(f"Successfully loaded parameters from {path}")
        except FileNotFoundError:
            print(f"Warning: Parameter file {path} not found.")
            return

        # Expected format: {layer_idx: {"state_dict": {...}}} 
        for layer_idx, data in loaded.items():
            # Handle potential string keys for layer_idx
            try:
                idx = int(layer_idx)
            except ValueError:
                continue
                
            if isinstance(data, dict) and "state_dict" in data:
                self.layer_params[idx] = data["state_dict"]
            else:
                # Fallback if structure is different
                self.layer_params[idx] = data
                
        # print(f"Loaded parameters for layers: {sorted(list(self.layer_params.keys()))}")

    def _get_layer_params(self, layer_idx: int) -> Dict[str, torch.Tensor]:
        if layer_idx not in self.layer_params:
            # Fallback: initialize default parameters on the fly
            # This allows usage without loading params (random/default init)
            # Or if the specific layer is missing in the loaded params
            return self._init_default_params()
        return self.layer_params[layer_idx]

    def _init_default_params(self) -> Dict[str, torch.Tensor]:
        params = {}
        if self.mode == "diag":
            params["log_scale"] = torch.zeros(self.head_dim, device=self.device)
        else:
            # Initialize identity matrix for full mode
            linear_weight = torch.eye(self.head_dim, device=self.device)
            params["linear.weight"] = linear_weight
            
        if self.learnable_clip:
            init = math.log(math.exp(self.clip_init) - 1)  # inverse softplus
            params["clip_k"] = torch.full((self.head_dim,), init, dtype=torch.float32, device=self.device)
            params["clip_v"] = torch.full((self.head_dim,), init, dtype=torch.float32, device=self.device)
        return params

    def transform(self, x: torch.Tensor, layer_idx: int, kind: str, **kwargs) -> torch.Tensor:
        """
        Apply affine transform to K/V.
        x: [..., head_dim]
        kind: "k" or "v" (for clip selection)
        layer_idx: current layer index
        """
        params = self._get_layer_params(layer_idx)
        
        if self.mode == "diag":
            log_scale = params["log_scale"]
            log_scale = torch.clamp(log_scale, min=-5.0, max=5.0)
            # Ensure scale is on same device as x
            scale = torch.exp(log_scale).to(x)
            x = x * scale
        else:
            weight = params["linear.weight"]
            x = torch.matmul(x, weight.t().to(x))
            
        if self.learnable_clip:
            if kind == "k":
                clip_param = params["clip_k"]
            else:
                clip_param = params["clip_v"]
            
            alpha = torch.nn.functional.softplus(clip_param).to(x)
            x = torch.clamp(x, -alpha, alpha)
            
        return torch.nan_to_num(x)

    def inverse(self, x: torch.Tensor, layer_idx: int, **kwargs) -> torch.Tensor:
        params = self._get_layer_params(layer_idx)
        
        if self.mode == "diag":
            log_scale = params["log_scale"]
            log_scale = torch.clamp(log_scale, min=-5.0, max=5.0)
            inv_scale = torch.exp(-log_scale).to(x)
            out = torch.nan_to_num(x * inv_scale)
            return out
        
        # full matrix inverse
        weight = params["linear.weight"].to(torch.float64) # Higher precision for inverse
        inv_w = torch.linalg.inv(weight).to(x) # Cast back to x's dtype/device
        return torch.matmul(x, inv_w.t())

    # Backward-compatible aliases (updated signature)
    def forward_transform(self, x: torch.Tensor, kind: str, layer_idx: int) -> torch.Tensor:
        return self.transform(x, kind, layer_idx)

    def inverse_transform(self, x: torch.Tensor, layer_idx: int) -> torch.Tensor:
        return self.inverse(x, layer_idx)

class CustomCacheConfig:

    def __init__(
        self,

        transform_type: Optional[str] = "none",

        scores: Optional[torch.Tensor] = None,
        heads_selection: Optional[float] = 0.5,
        high_key_max_value: Optional[int] = 64,
        high_value_max_value: Optional[int] = 64,
        low_key_max_value: Optional[int] = 32,
        low_value_max_value: Optional[int] = 32,

        axis_key: Optional[List[int]] = [2],
        axis_value: Optional[List[int]] = [1, 3],

        device: Optional[str] = "cuda",
        comp_cr: Optional[bool] = False,
    ):
        self.transform_type = transform_type

        self.scores = scores
        self.heads_selection = heads_selection
        self.high_key_max_value = high_key_max_value
        self.high_value_max_value = high_value_max_value
        self.low_key_max_value = low_key_max_value
        self.low_value_max_value = low_value_max_value

        self.axis_key = axis_key
        self.axis_value = axis_value

        self.device = device
        self.comp_cr = comp_cr

        self.validate()

        # print("\n============== Using CustomCacheConfig... ==============\n")

    def validate(self):
        """Validates if the arguments passed are correct"""

        # assert self.scores is not None, "scores should be provided"
        assert self.transform_type in ["none", "hadamard", "affine"], "transform_type should be in ['none', 'hadamard', 'affine']"
        assert self.heads_selection >= 0 and self.heads_selection <= 1, "heads_selection should be between 0 and 1"
        assert self.high_key_max_value >= self.low_key_max_value, "high_key_max_value should be greater than low_key_max_value"
        assert self.high_value_max_value >= self.low_value_max_value, "high_value_max_value should be greater than low_value_max_value"
        assert self.high_key_max_value > 0 and self.high_key_max_value < 256, "high_key_max_value should be between 1 and 255"
        assert self.high_value_max_value > 0 and self.high_value_max_value < 256, "high_value_max_value should be between 1 and 255"
        assert self.low_key_max_value > 0 and self.low_key_max_value < 256, "low_key_max_value should be between 1 and 255"
        assert self.low_value_max_value > 0 and self.low_value_max_value < 256, "low_value_max_value should be between 1 and 255"
        
        assert all(0 <= x <= 3 for x in self.axis_key), "axis_key values should be in the range [0, 3], bacause kvcache is 4D tensor [batch_size, num_heads, seq_len, head_dim]"
        assert all(0 <= x <= 3 for x in self.axis_value), "axis_value values should be in the range [0, 3], bacause kvcache is 4D tensor [batch_size, num_heads, seq_len, head_dim]"


class CustomCache(DynamicCache):

    def __init__(self, cache_config: CustomCacheConfig) -> None:
        super().__init__()

        self._high_quantized_key_cache: deque[torch.Tensor] = deque()
        self._high_quantized_value_cache: deque[torch.Tensor] = deque()        
        self._low_quantized_key_cache: deque[torch.Tensor] = deque()
        self._low_quantized_value_cache: deque[torch.Tensor] = deque()

        self.transform_type = cache_config.transform_type
        match self.transform_type:
            case "none":
                self.transform = NoneTransform()
            case "hadamard":
                self.transform = HadamardTransform(0x3333)
            case "affine":
                self.transform = AffineTransform(head_dim=128, mode="diag", params_path="/home/bingxing2/home/scx9kvs/mxy/Infer_Comm/affine_config/affine_params_v2.pt", device="cuda")
        self.scores = cache_config.scores
        self.heads_selection = cache_config.heads_selection
        self.high_key_max_value = cache_config.high_key_max_value
        self.high_value_max_value = cache_config.high_value_max_value
        self.low_key_max_value = cache_config.low_key_max_value
        self.low_value_max_value = cache_config.low_value_max_value

        self.axis_key = cache_config.axis_key
        self.axis_value = cache_config.axis_value
        self.device = cache_config.device
        self.comp_cr = cache_config.comp_cr
        self.compression_ratio = 0
        self.skip_quantization = False # Skip quantization after transform
        # transformers>=4.57 stores cache state in ``Cache.layers`` instead of
        # the legacy ``key_cache``/``value_cache`` lists.  Keep the logical
        # sequence length separately because the prefill tensors are quantized
        # and intentionally not retained in a DynamicLayer.
        self._cache_lengths: List[int] = []

        if self.scores is not None:
            pruned_num_heads = round(self.scores.numel() * self.heads_selection)
            self.scores_mask = torch.zeros_like(self.scores, dtype=torch.bool, device=self.device)
            flat_indices = torch.argsort(self.scores.flatten())[:pruned_num_heads]
            multi_indices = torch.unravel_index(flat_indices, self.scores.shape)
            self.scores_mask[multi_indices] = True

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if len(self.layers) < layer_idx:
            raise ValueError("Does not support model usage where layers are skipped. Use DynamicCache.")
        # prefill
        elif len(self.layers) == layer_idx:
            keys_to_return, values_to_return = key_states, value_states

            key_states = self.transform.transform(key_states, layer_idx, kind="k")
            value_states = self.transform.transform(value_states, layer_idx, kind="v")

            # 将key_states和value_states按头进行拆分(Duo_attention)
            self._head_layer_split(key_states, value_states, layer_idx, cache_kwargs)

        # decode
        else:
            if self.comp_cr and self.compression_ratio == 0:
                self.compression_ratio = self.compute_compression_ratio(key_states.dtype, key_states.device)

            # kvcache reconstruction
            if len(self._low_quantized_key_cache) != 0:
                self._head_layer_reconstruct(layer_idx)
                self.layers[layer_idx].keys = self.transform.inverse(self.layers[layer_idx].keys, layer_idx)
                self.layers[layer_idx].values = self.transform.inverse(self.layers[layer_idx].values, layer_idx)

            # 更新decode阶段kvcache
            self.layers[layer_idx].keys = torch.cat([self.layers[layer_idx].keys, key_states], dim=-2)
            self.layers[layer_idx].values = torch.cat([self.layers[layer_idx].values, value_states], dim=-2)
            self._cache_lengths[layer_idx] += key_states.shape[-2]

            keys_to_return = self.layers[layer_idx].keys
            values_to_return = self.layers[layer_idx].values

        return keys_to_return, values_to_return

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        if layer_idx is None or len(self._cache_lengths) <= layer_idx:
            return 0
        return self._cache_lengths[layer_idx]

    def get_mask_sizes(self, cache_position: torch.Tensor, layer_idx: int) -> tuple[int, int]:
        """Return mask dimensions using the logical (possibly quantized) cache length."""
        return self.get_seq_length(layer_idx) + cache_position.shape[0], 0

    def compute_compression_ratio(self, ori_dtype, device="cuda"):
        import pickle
        import sys
        import os
        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
        from offline_search.evaluation.compression_ratio.nvcomp_wrapper import NVCompWrapper, PackedData, to_device

        num_layers = len(self._low_quantized_key_cache)
        if num_layers == 0:
            return 0.0
        
        meta_data = []
        original_key_tensors = []
        original_value_tensors = []
        to_compressed_key_tensors = None
        to_compressed_value_tensors = None
        current_idx = 0

        # Use list indexing instead of pop to preserve deque content
        # deque supports indexing but it is O(n). However num_layers is small (e.g. 32) so it is fine.
        # Alternatively convert to list for iteration.
        low_keys_list = list(self._low_quantized_key_cache)
        low_values_list = list(self._low_quantized_value_cache)
        high_keys_list = list(self._high_quantized_key_cache)
        high_values_list = list(self._high_quantized_value_cache)

        for i in range(num_layers):
            low_keys_quantized, low_keys_meta_data = low_keys_list[i]
            low_values_quantized, low_values_meta_data = low_values_list[i]
            high_keys_quantized, high_keys_meta_data = high_keys_list[i]
            high_values_quantized, high_values_meta_data = high_values_list[i]

            # concat the original tensors and meta data in each layer
            keys_layer_block = torch.cat([
                to_device(low_keys_quantized, device),
                to_device(high_keys_quantized, device),
            ], dim=1)
            values_layer_block = torch.cat([
                to_device(low_values_quantized, device),
                to_device(high_values_quantized, device),
            ], dim=1)
            
            original_key_tensors.append(torch.empty(keys_layer_block.shape, dtype=ori_dtype, device=device))
            original_value_tensors.append(torch.empty(values_layer_block.shape, dtype=ori_dtype, device=device))

            if to_compressed_key_tensors is None and to_compressed_value_tensors is None:
                batch_size, heads, tokens, channels = keys_layer_block.shape
                final_shape = (batch_size*num_layers, heads, tokens, channels)
                to_compressed_key_tensors = torch.empty(final_shape, dtype=keys_layer_block.dtype, device=device)
                to_compressed_value_tensors = torch.empty(final_shape, dtype=values_layer_block.dtype, device=device)
            
            end_idx = current_idx + keys_layer_block.shape[0]
            to_compressed_key_tensors[current_idx:end_idx, ...] = keys_layer_block
            to_compressed_value_tensors[current_idx:end_idx, ...] = values_layer_block
            current_idx = end_idx
            
            meta_data.append([low_keys_meta_data, high_keys_meta_data, low_values_meta_data, high_values_meta_data])

        meta_data = to_device(meta_data, device)
        original_size = len(original_key_tensors) * original_key_tensors[0].numel() * original_key_tensors[0].element_size() * 2 / 1024 / 1024
        
        del original_key_tensors, original_value_tensors, low_keys_list, low_values_list, high_keys_list, high_values_list

        nvcomp_wrapper = NVCompWrapper("ANS", data_type="|u1")
        compressed_data = nvcomp_wrapper.compress(to_compressed_key_tensors, to_compressed_value_tensors)
        # packed_data = PackedData(compressed_data, meta_data)
        
        compressed_size = (compressed_data.buffer.numel() * compressed_data.buffer.element_size() + len(meta_data) * len(meta_data[0]) * meta_data[0][0].numel() * meta_data[0][0].element_size()) / 1024 / 1024
        
        if compressed_size == 0:
            return 0.0
            
        return round(original_size / compressed_size, 4)

    def _quantize(
        self,
        max_value: int,
        tensor: torch.Tensor, 
        axis: List[int], 
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Corrected quantization implementation.
        Maps [min, max] -> [0, max_value]
        """
        if tensor.numel() == 0:
            # Handle empty tensor: return dummy metadata with consistent shape
            shape = list(tensor.shape)
            for i in axis:
                idx = i if i >= 0 else i + tensor.dim()
                shape[idx] = 1
            
            # Create dummy min/scale (zeros)
            dummy_val = torch.zeros(shape, dtype=tensor.dtype, device=tensor.device)
            meta_data = torch.stack([dummy_val, dummy_val], dim=0)
            
            return tensor.to(torch.uint8), meta_data

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
        meta_data = torch.stack([
            min_.to(tensor.dtype),
            scale.to(tensor.dtype)
        ], dim=0)
        
        return quantized_tensor, meta_data

    def _dequantize(
        self, 
        quantized_tensor: torch.Tensor,
        meta_data: torch.Tensor,
    ) -> torch.Tensor:
        """
        Corrected dequantization.
        Formula: Real = Quantized * Scale + Min_Value
        """
        # 1. 处理空 tensor 情况 (Early Return)
        if quantized_tensor.numel() == 0:
            target_dtype = meta_data[1].dtype
            return quantized_tensor.to(target_dtype)

        # 2. 获取元数据
        min_val = meta_data[0]
        quant_scale = meta_data[1]

        # 3. 类型转换 (Cast)
        quant_float = quantized_tensor.to(quant_scale.dtype)

        # 4. 反量化计算
        # result = x * s + b
        dequantized_tensor = quant_float * quant_scale + min_val

        return dequantized_tensor

    def _create_identity_metadata(self, tensor: torch.Tensor, axis: List[int]) -> torch.Tensor:
        """
        Create identity metadata (min=0, scale=1) for skip_quantization.
        Ensures metadata shape is consistent with quantized version.
        """
        shape = list(tensor.shape)
        for i in axis:
            idx = i if i >= 0 else i + tensor.dim()
            shape[idx] = 1
        
        min_ = torch.zeros(shape, dtype=tensor.dtype, device=tensor.device)
        scale = torch.ones(shape, dtype=tensor.dtype, device=tensor.device)
        return torch.stack([min_, scale], dim=0)

    def _head_layer_split(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict[str, Any]] = None,        
    ) -> None:
        # 获取当前层的mask，形状为 [heads]
        # scores_mask[i] 是一个布尔张量，例如 [True, True, False, ...]
        layer_mask = self.scores_mask[layer_idx].to(key_states.device)

        # 使用布尔索引来分离 "low" (mask为True) 和 "high" (mask为False) 的头
        # low_keys 的形状会是 [batch_size, num_true_heads, seq_len, head_dim]
        if self.skip_quantization:
            # Low
            low_k = key_states[:, layer_mask, :, :].contiguous()
            low_v = value_states[:, layer_mask, :, :].contiguous()
            low_keys = (low_k, self._create_identity_metadata(low_k, self.axis_key))
            low_values = (low_v, self._create_identity_metadata(low_v, self.axis_value))
            
            # High
            high_k = key_states[:, ~layer_mask, :, :].contiguous()
            high_v = value_states[:, ~layer_mask, :, :].contiguous()
            high_keys = (high_k, self._create_identity_metadata(high_k, self.axis_key))
            high_values = (high_v, self._create_identity_metadata(high_v, self.axis_value))
        else:
            low_keys = self._quantize(
                self.low_key_max_value,
                key_states[:, layer_mask, :, :].contiguous(),
                self.axis_key,
            )
            low_values = self._quantize(
                self.low_value_max_value,
                value_states[:, layer_mask, :, :].contiguous(),
                self.axis_value,
            )

            # high_keys 的形状会是 [batch_size, num_false_heads, seq_len, head_dim]
            # 使用 '~' 操作符来反转布尔mask，选取为False的头
            high_keys = self._quantize(
                self.high_key_max_value,
                key_states[:, ~layer_mask, :, :].contiguous(),
                self.axis_key,
            )
            high_values = self._quantize(
                self.high_value_max_value,
                value_states[:, ~layer_mask, :, :].contiguous(),
                self.axis_value,
            )

        # 将分离出的张量添加到对应的列表中
        self._low_quantized_key_cache.append(low_keys)
        self._low_quantized_value_cache.append(low_values)
        self._high_quantized_key_cache.append(high_keys)
        self._high_quantized_value_cache.append(high_values)

        cache_layer = DynamicLayer()
        cache_layer.lazy_initialization(key_states)
        cache_layer.keys = key_states[..., :0, :]
        cache_layer.values = value_states[..., :0, :]
        self.layers.append(cache_layer)
        self._cache_lengths.append(key_states.shape[-2])

    def _head_layer_reconstruct(
        self,
        layer_idx: int,
    ) -> None:

        assert len(self._low_quantized_key_cache) != 0, "low_quantized_key_cache is empty, nothing to reconstruct"

        low_keys_dequantized = self._dequantize(*self._low_quantized_key_cache.popleft())
        low_values_dequantized = self._dequantize(*self._low_quantized_value_cache.popleft())
        high_keys_dequantized = self._dequantize(*self._high_quantized_key_cache.popleft())
        high_values_dequantized = self._dequantize(*self._high_quantized_value_cache.popleft())
        
        # 1. 创建一个与原始层形状、类型、设备都相同的空张量作为容器
        B, _, S, D = low_keys_dequantized.shape
        H = self.scores_mask.shape[1]
        device = low_keys_dequantized.device
        dtype = low_keys_dequantized.dtype

        reconstructed_key_layer = torch.empty(B, H, S, D, device=device, dtype=dtype)
        reconstructed_value_layer = torch.empty(B, H, S, D, device=device, dtype=dtype)

        # 2. 获取当前层的布尔掩码
        layer_mask = self.scores_mask[layer_idx].to(device)

        # 3. 使用布尔掩码将 dequantized 张量放置到正确的位置
        reconstructed_key_layer[:, layer_mask, :, :] = low_keys_dequantized
        reconstructed_value_layer[:, layer_mask, :, :] = low_values_dequantized
        
        reconstructed_key_layer[:, ~layer_mask, :, :] = high_keys_dequantized
        reconstructed_value_layer[:, ~layer_mask, :, :] = high_values_dequantized

        # 4. 将解量化后的kvcache拼接到kvcache中(每层只需要执行一次解量化)
        self.layers[layer_idx].keys = torch.cat([self.layers[layer_idx].keys, reconstructed_key_layer], dim=-2)
        self.layers[layer_idx].values = torch.cat([self.layers[layer_idx].values, reconstructed_value_layer], dim=-2)
