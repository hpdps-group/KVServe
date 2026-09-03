import torch
from typing import Optional, Any, List, Tuple
from transformers.cache_utils import DynamicCache

def search_hyperplane(X, max_iter: int = 1000):
    """
    Given a tensor X of shape (bsz, seq_len, head_dim), search for an hyperplane Y (bsz, head_dim)
    such that for every i, <X[:, i], Y> <= 0. Returns - 1e5 * Y / ||Y|| ** 2 to ensure exp(<X, Y>) = 0
    """
    Y = X.mean(1)  # this initialization is enough for most cases
    for _ in range(max_iter):
        mask = torch.bmm(X, Y.unsqueeze(-1)) <= 0
        if not mask.any():
            return -1e5 * Y / Y.norm(dim=-1, keepdim=True) ** 2
        Y += (X * mask).sum(1) / mask.sum(1).clamp(min=1)
    # Return best effort if convergence fails
    return -1e5 * Y / Y.norm(dim=-1, keepdim=True) ** 2

class DuoAttentionCacheConfig:
    """
    Configuration class for DuoAttentionCache.
    """
    def __init__(
        self,

        scores: Optional[torch.Tensor] = None,
        heads_selection: Optional[float] = 0.5,
        sink_size: Optional[int] = 128,
        recent_size: Optional[int] = 256,
        device: Optional[str] = "cuda",
        comp_cr: Optional[bool] = False,
    ):
        self.scores = scores
        self.heads_selection = heads_selection
        self.sink_size = sink_size
        self.recent_size = recent_size
        self.device = device
        self.comp_cr = comp_cr

        self.validate()

        print("\n============== Using DuoAttentionCacheConfig... ==============\n")

    def validate(self):
        """Validates if the arguments passed are correct"""
        assert self.scores is not None, "scores should be provided"
        assert self.heads_selection >= 0 and self.heads_selection <= 1, "heads_selection should be between 0 and 1"
        assert self.sink_size >= 0, "sink_size should be non-negative"
        assert self.recent_size >= 0, "recent_size should be non-negative"

class DuoAttentionCache(DynamicCache):
    """
    DuoAttention Cache that manages KV Cache for retrieval and streaming heads.
    """

    def __init__(self, cache_config: DuoAttentionCacheConfig) -> None:
        super().__init__()
        self.scores = cache_config.scores
        self.heads_selection = cache_config.heads_selection
        self.sink_size = cache_config.sink_size
        self.recent_size = cache_config.recent_size
        self.device = cache_config.device
        self.comp_cr = cache_config.comp_cr
        self.compression_ratio = 0

        if self.scores is not None:
            pruned_num_heads = round(self.scores.numel() * self.heads_selection)
            self.scores_mask = torch.zeros_like(self.scores, dtype=torch.bool, device=self.device)
            flat_indices = torch.argsort(self.scores.flatten())[:pruned_num_heads]
            multi_indices = torch.unravel_index(flat_indices, self.scores.shape)
            self.scores_mask[multi_indices] = True
            self.masked_key_indices = []


    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]

        if len(self.key_cache) < layer_idx:
            raise ValueError("Does not support model usage where layers are skipped. Use DynamicCache.")
        # prefill
        elif len(self.key_cache) == layer_idx:
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)

            # self._build_masking(key_states, layer_idx)
        # decode
        else:

            if self.comp_cr and self.compression_ratio == 0:
                self.compression_ratio = self.compute_compression_ratio(key_states.dtype, key_states.device)

            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)

            layer_mask = self._build_masking(self.key_cache[layer_idx], layer_idx)

            # DuoAttention Logic
            if "query_states" in cache_kwargs and layer_mask is not None:
                query_states = cache_kwargs["query_states"]
                
                self._apply_duo_masking(query_states, layer_idx, layer_mask)            

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def _build_masking(self, keys: torch.Tensor, layer_idx: int):

        seq_len = keys.shape[-2]
        
        # Check if masking is needed
        if seq_len <= self.sink_size + self.recent_size:
            self.masked_key_indices.append(None)
            return
            
        layer_mask = self.scores_mask[layer_idx].to(keys.device) # [num_heads]
        if not layer_mask.any():
            self.masked_key_indices.append(None)
            return

        masked_keys = torch.zeros_like(keys[..., 0], dtype=torch.bool)
        masked_keys[:, layer_mask, self.sink_size : -self.recent_size] = True
        
        return torch.nonzero(masked_keys, as_tuple=True)

    def _apply_duo_masking(self, query: torch.Tensor, layer_idx: int, layer_mask: torch.Tensor):
        """
        Apply mask to the current layer's key cache based on DuoAttention logic.
        Modifies self.key_cache[layer_idx] in-place.
        """
        # Determine which KV heads are streaming
        bsz, num_heads, seq_len, head_dim = query.shape
        num_key_value_heads = self.key_cache[layer_idx].shape[1]
        num_groups = num_heads // num_key_value_heads

        # Build a fake key k per key group such that for every query q, exp(<q, k>) = 0
        q = query.view(bsz, num_key_value_heads, num_groups, seq_len, head_dim)
        q = q.reshape(bsz * num_key_value_heads, num_groups * seq_len, head_dim)
        k = search_hyperplane(q)
        
        k = k.view(bsz, num_key_value_heads, head_dim)

        # At indices, update the keys to the fake keys
        batch_indices, head_indices, seq_indices = layer_mask
        batch_indices = batch_indices.to(k.device)
        head_indices = head_indices.to(k.device)
        seq_indices = seq_indices.to(k.device)
        self.key_cache[layer_idx][batch_indices, head_indices, seq_indices] = k[batch_indices, head_indices]

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        if len(self.key_cache) <= layer_idx:
            return 0
        return self._seen_tokens if layer_idx == 0 else self._seen_tokens

    def compute_compression_ratio(self, ori_dtype, device="cuda"):
        import pickle
        
        num_layers = len(self.key_cache)
        if num_layers == 0:
            return 0.0
            
        original_key_tensors = []
        original_value_tensors = []
        compressed_key_tensors = []
        compressed_value_tensors = []
        
        for i in range(num_layers):
            key_cache_layer = self.key_cache[i]
            value_cache_layer = self.value_cache[i]
            
            # Original size calculation
            original_key_tensors.append(torch.empty(key_cache_layer.shape, dtype=ori_dtype, device=device))
            original_value_tensors.append(torch.empty(value_cache_layer.shape, dtype=ori_dtype, device=device))
            
            masked_indices = self._build_masking(key_cache_layer, i)
            
            if masked_indices is None:
                compressed_key_tensors.append(key_cache_layer)
                compressed_value_tensors.append(value_cache_layer)
            else:
                keep_mask = torch.ones(key_cache_layer.shape[:-1], dtype=torch.bool, device=device)
                keep_mask[masked_indices] = False
                
                expanded_keep_mask = keep_mask.unsqueeze(-1).expand_as(key_cache_layer)
                
                kept_keys = key_cache_layer[expanded_keep_mask]
                kept_values = value_cache_layer[expanded_keep_mask]
                
                compressed_key_tensors.append(kept_keys)
                compressed_value_tensors.append(kept_values)

        original_size = (len(pickle.dumps(original_key_tensors)) + len(pickle.dumps(original_value_tensors))) / 1024 / 1024
        compressed_size = (len(pickle.dumps(compressed_key_tensors)) + len(pickle.dumps(compressed_value_tensors))) / 1024 / 1024
        
        del original_key_tensors, original_value_tensors, compressed_key_tensors, compressed_value_tensors
        
        if compressed_size == 0:
            return 0.0
            
        return round(original_size / compressed_size, 4)
