"""KVCompressionAdapter — wraps kvserve CompressionManager for use in kvserve_v1.

Shape contract:
  Connector block format: [num_layers, 2, num_blocks, block_size,
                           num_kv_heads, head_size]
  Legacy flat format:     [num_layers, 2, num_tokens, kv_dim]
  Original CM format:     [num_layers, 2, num_blocks, block_size,
                           num_kv_heads, head_size]

Three compression modes (set via kv_connector_extra_config["compression"]):
  "default"   – use built-in DEFAULT_COMPRESSION_CONFIG from kvserve
  dict        – custom config dict (current behaviour)
  {"mode": "controller", "library_path": "...", "service_config": {...}} –
               online adaptive selection via OnlineController + ProfileLibrary

Transport packing is handled by compression.wire, which keeps compressed data
and quantization tensors GPU-resident for NCCL transfer.
"""

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING, Any, Optional

import torch
from vllm.logger import init_logger

if TYPE_CHECKING:
    from kvserve_v1.compression.compression_manager import CompressedKVData, CompressionConfig, CompressionManager

logger = init_logger(__name__)

_COMPRESSED_SENTINEL = "__compressed__"


def is_compressed_layer_names(layer_names: list[str]) -> bool:
    return bool(layer_names) and layer_names[0] == _COMPRESSED_SENTINEL


def strip_sentinel(layer_names: list[str]) -> list[str]:
    return layer_names[1:] if is_compressed_layer_names(layer_names) else layer_names


def add_sentinel(layer_names: list[str]) -> list[str]:
    return [_COMPRESSED_SENTINEL] + layer_names


def _cfg_cache_key(cfg_dict: dict) -> str:
    """Stable string key for caching CompressionManagers by config."""
    return json.dumps(cfg_dict, sort_keys=True, default=str)


class KVCompressionAdapter:
    """Adapter between kvserve_v1 and kvserve.CompressionManager.

    Supports three modes:
      default    – compression_spec = "default"
      custom     – compression_spec = {pipeline: [...], ...}
      controller – compression_spec = {
                     "mode": "controller",
                     "library_path": "/path/to/profiles.json",
                     "epsilon": 0.1,           # optional, default 0.1
                     "alpha": 0.2,             # optional, default 0.2
                     "service_config": {       # runtime service parameters
                       "bandwidth_mbps": 1000.0,
                       "slo_ms": 100.0,
                       "accuracy_requirement": 0.92,
                       "t_model_ms": 20.0,     # optional, default 0
                     },
                   }
    """

    def __init__(self, compression_spec: Any, num_kv_heads: int, head_size: int,
                 model_name: Optional[str] = None, tp_rank: int = 0,
                 tp_size: int = 1):
        self._num_kv_heads = num_kv_heads
        self._head_size = head_size
        self._model_name = model_name  # used to patch model_name in "default" mode
        self._tp_rank = tp_rank
        self._tp_size = tp_size

        # controller-mode state
        self._controller: Optional[Any] = None
        self._service_cfg: dict = {}
        # request_id → (inserted_time, select_profile context)
        self._last_contexts: dict[str, tuple[float, dict]] = {}

        # (manager, config) cache keyed by stable JSON repr of cfg_dict
        self._manager_cache: dict[str, tuple["CompressionManager", "CompressionConfig"]] = {}

        if compression_spec == "default":
            from kvserve_v1.compression.compression_manager import get_default_compression_config
            cfg_dict = get_default_compression_config()
            self._patch_model_name(cfg_dict)
            self._manager = self._build_manager(cfg_dict)
            self._config = self._make_config(cfg_dict)
        elif isinstance(compression_spec, dict) and compression_spec.get("mode") == "controller":
            self._init_controller(compression_spec)
            self._manager = None
            self._config = None
        else:
            self._manager = self._build_manager(compression_spec)
            self._config = self._make_config(compression_spec)

    # ── Initializers ───────────────────────────────────────────────────────

    def _patch_model_name(self, cfg_dict: dict) -> None:
        """Override model_name in quantizer_config with the actual running model.

        DEFAULT_COMPRESSION_CONFIG hardcodes "Llama-3.1-8B-Instruct"; running a
        different model (e.g. Qwen2.5-7B with 4 KV heads) causes a mask shape
        mismatch in head_split.  When the adapter knows the real model name, we
        patch it here so DuoConfigGenerator loads the correct scores CSV.
        """
        if self._model_name and "quantizer_config" in cfg_dict:
            cfg_dict["quantizer_config"]["model_name"] = self._model_name
            logger.info("[KVCompressionAdapter] patched quantizer model_name -> %s",
                        self._model_name)

    def _init_controller(self, spec: dict) -> None:
        from kvserve_v1.compression.controller.profile_library import ProfileLibrary
        from kvserve_v1.compression.controller.online_controller import OnlineController
        library = ProfileLibrary(spec["library_path"])
        self._controller = OnlineController(
            profile_library=library,
            epsilon=spec.get("epsilon", 0.1),
            alpha=spec.get("alpha", 0.2),
        )
        self._service_cfg = spec.get("service_config", {})
        logger.info("[KVCompressionAdapter] controller mode: library=%s epsilon=%.2f",
                    spec["library_path"], spec.get("epsilon", 0.1))

    def _make_config(self, cfg_dict: dict) -> "CompressionConfig":
        from kvserve_v1.compression.compression_manager import CompressionConfig
        quantizer_config = cfg_dict.get("quantizer_config")
        if quantizer_config is not None:
            quantizer_config = dict(quantizer_config)
            quantizer_config["tp_rank"] = self._tp_rank
            quantizer_config["tensor_parallel_size"] = self._tp_size
        return CompressionConfig(
            enabled=cfg_dict.get("enabled", True),
            pipeline=cfg_dict.get("pipeline", []),
            transformer_config=cfg_dict.get("transformer_config"),
            quantizer_config=quantizer_config,
            codec_config=cfg_dict.get("codec_config"),
            min_compress_size=cfg_dict.get("min_compress_size", 0),
        )

    @staticmethod
    def _is_tilelang_fused(cfg_dict: dict) -> bool:
        if cfg_dict.get("impl") == "tilelang_fused":
            return True
        if cfg_dict.get("quantizer_impl") == "tilelang_fused":
            return True
        qc = cfg_dict.get("quantizer_config") or {}
        return qc.get("impl") == "tilelang_fused"

    def _build_manager(self, cfg_dict: dict) -> "CompressionManager":
        from kvserve_v1.compression.compression_manager import CompressionManager
        cfg_dict = dict(cfg_dict)
        pipeline = list(cfg_dict.get("pipeline", []))
        transformer_cls = None
        quantizer_cls = None
        codec_cls = None
        use_tilelang = self._is_tilelang_fused(cfg_dict)
        # TileLang fused path already includes Hadamard; drop a redundant transformer.
        if use_tilelang and "transformer" in pipeline:
            pipeline = [p for p in pipeline if p != "transformer"]
            cfg_dict["pipeline"] = pipeline
        if "transformer" in pipeline:
            from kvserve_v1.compression.transformer.kvserve_transformer import KVServeTransformer
            transformer_cls = KVServeTransformer
        if "quantizer" in pipeline:
            if use_tilelang:
                from kvserve_v1.compression.quantizer.tilelang_quantizer import TileLangFusedQuantizer
                quantizer_cls = TileLangFusedQuantizer
            else:
                from kvserve_v1.compression.quantizer.kvserve_quantizer import KVServeQuantizer
                quantizer_cls = KVServeQuantizer
        if "codec" in pipeline:
            from kvserve_v1.compression.codec import KVServeCodec
            codec_cls = KVServeCodec
        return CompressionManager(
            config=self._make_config(cfg_dict),
            transformer=transformer_cls,
            quantizer=quantizer_cls,
            codec=codec_cls,
        )

    def _get_cached_manager(self, cfg_dict: dict) -> tuple["CompressionManager", "CompressionConfig"]:
        """Return (manager, config) for the given config dict, building and caching if needed."""
        key = _cfg_cache_key(cfg_dict)
        if key not in self._manager_cache:
            manager = self._build_manager(cfg_dict)
            self._manager_cache[key] = (manager, self._make_config(cfg_dict))
        return self._manager_cache[key]

    # ── Public API ──────────────────────────────────────────────────────────

    def compress(self, stacked_kv: torch.Tensor,
                 request_id: str) -> Optional["CompressedKVData"]:
        """Compress connector KV tensor, preserving its original shape."""
        meta = {
            "original_dtype": str(stacked_kv.dtype).replace("torch.", ""),
            "original_shape": list(stacked_kv.shape),
        }

        if stacked_kv.ndim == 6:
            stacked_6d = stacked_kv.contiguous()
        elif stacked_kv.ndim == 4:
            num_layers, _, num_tokens, _ = stacked_kv.shape
            stacked_6d = stacked_kv.contiguous().view(
                num_layers, 2, 1, num_tokens,
                self._num_kv_heads, self._head_size)
        else:
            raise ValueError(
                f"Unsupported KV tensor rank for compression: {stacked_kv.ndim}, "
                f"shape={tuple(stacked_kv.shape)}")

        if self._controller is not None:
            return self._compress_controller(stacked_6d, request_id, meta)

        return self._manager.compress_all_layers(
            all_layers_data=stacked_6d,
            request_id=request_id,
            config=self._config,
            metadata=meta,
        )

    def decompress(self, compressed: "CompressedKVData") -> Optional[torch.Tensor]:
        """Decompress back to the connector KV tensor shape."""
        if self._controller is not None:
            cfg_dict = compressed.metadata.get("ctrl_profile_cfg")
            if cfg_dict is None:
                logger.error("[KVCompressionAdapter] controller mode: "
                             "ctrl_profile_cfg not found in metadata")
                return None
            manager, cfg = self._get_cached_manager(cfg_dict)
        else:
            manager, cfg = self._manager, self._config

        result = manager.decompress_all_layers(compressed, config=cfg)
        if result is None:
            return None
        if isinstance(result, list):
            result = torch.cat(result, dim=0)
        original_shape = compressed.metadata.get("original_shape")
        if original_shape is not None:
            return result.contiguous().view(*[int(x) for x in original_shape])

        num_layers = result.shape[0]
        kv_dim = self._num_kv_heads * self._head_size
        return result.contiguous().view(num_layers, 2, -1, kv_dim)

    def update_controller(self, request_id: str, observed_latency_ms: float) -> None:
        """Feed observed transfer latency back to the bandit (controller mode only)."""
        if self._controller is None:
            return
        entry = self._last_contexts.pop(request_id, None)
        if entry is not None:
            _, ctx = entry
            self._controller.update(ctx, observed_latency_ms)

    def get_controller_stats(self) -> Optional[dict]:
        """Return OnlineController statistics, or None if not in controller mode."""
        if self._controller is None:
            return None
        return self._controller.get_statistics()

    # ── Controller-mode internals ──────────────────────────────────────────

    _CONTEXT_TTL_S: float = 300.0  # drop contexts older than 5 minutes

    def _evict_stale_contexts(self) -> None:
        cutoff = time.monotonic() - self._CONTEXT_TTL_S
        stale = [rid for rid, (ts, _) in self._last_contexts.items() if ts < cutoff]
        for rid in stale:
            self._last_contexts.pop(rid)
        if stale:
            logger.warning("[KVCompressionAdapter] evicted %d stale controller "
                           "contexts (TTL=%.0fs)", len(stale), self._CONTEXT_TTL_S)

    def _compress_controller(
        self,
        stacked_6d: torch.Tensor,
        request_id: str,
        meta: dict,
    ) -> Optional["CompressedKVData"]:
        V_bytes = float(stacked_6d.numel() * stacked_6d.element_size())
        B_mbps = float(self._service_cfg.get("bandwidth_mbps", 10_000.0))
        T_model_ms = float(self._service_cfg.get("t_model_ms", 0.0))
        T_SLO_ms = float(self._service_cfg.get("slo_ms", float("inf")))
        acc_req = float(self._service_cfg.get("accuracy_requirement", 0.9))

        profile, context = self._controller.select_profile(
            V_bytes, B_mbps, T_model_ms, T_SLO_ms, acc_req)
        self._last_contexts[request_id] = (time.monotonic(), context)
        self._evict_stale_contexts()

        if profile is None:
            logger.debug("[KVCompressionAdapter] controller: no compression for %s "
                         "(reason=%s)", request_id, context.get("reason"))
            return None

        # Embed the selected profile's config so the consumer can decompress
        # without needing its own controller state.
        meta["ctrl_profile_cfg"] = profile.compression_config

        manager, cfg = self._get_cached_manager(profile.compression_config)
        logger.debug("[KVCompressionAdapter] controller: profile=%s for %s",
                     profile.profile_id, request_id)
        return manager.compress_all_layers(
            all_layers_data=stacked_6d,
            request_id=request_id,
            config=cfg,
            metadata=meta,
        )
