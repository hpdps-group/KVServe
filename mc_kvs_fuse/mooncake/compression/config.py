# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

QuantAxis = Literal["channel", "token", "tensor"]
SplitType = Literal["head", "layer"]


@dataclass(frozen=True)
class MooncakeCompressionConfig:
    enabled: bool = False
    transformer: Literal["hadamard"] = "hadamard"
    seed: int = 3237919422
    split_type: SplitType = "head"
    score_path: str | None = None
    hybrid_ratio: float = 0.8
    axis_key: QuantAxis = "channel"
    axis_value: QuantAxis = "token"
    high_key_num_levels: int = 12
    high_value_num_levels: int = 8
    low_key_num_levels: int = 6
    low_value_num_levels: int = 4
    codec: Literal["ans"] = "ans"
    slot_count: int = 2
    logical_blocks_per_chunk: int = 1
    codec_warn_ratio: float = 0.95

    @classmethod
    def from_extra_config(
        cls, extra_config: dict[str, Any]
    ) -> MooncakeCompressionConfig:
        raw = extra_config.get("compression", {})
        if raw is None:
            raw = {}
        if not isinstance(raw, dict):
            raise ValueError("Mooncake compression config must be a mapping.")
        aliases = {
            "high_key_max_value": "high_key_num_levels",
            "high_value_max_value": "high_value_num_levels",
            "low_key_max_value": "low_key_num_levels",
            "low_value_max_value": "low_value_num_levels",
        }
        normalized = {aliases.get(key, key): value for key, value in raw.items()}
        unknown = set(normalized) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(
                "Unknown Mooncake compression option(s): " + ", ".join(sorted(unknown))
            )
        config = cls(**normalized)
        config.validate()
        return config

    def validate(self) -> None:
        if not self.enabled:
            return
        if self.transformer != "hadamard":
            raise ValueError("Only the Hadamard transformer is supported.")
        if self.codec != "ans":
            raise ValueError("Only the nvCOMP ANS codec is supported.")
        if self.split_type not in ("head", "layer"):
            raise ValueError("split_type must be 'head' or 'layer'.")
        if self.split_type == "head" and not self.score_path:
            raise ValueError("score_path is required when split_type='head'.")
        if not 0.0 <= self.hybrid_ratio <= 1.0:
            raise ValueError("hybrid_ratio must be in [0, 1].")
        if self.axis_key not in ("channel", "token", "tensor"):
            raise ValueError(f"Invalid axis_key: {self.axis_key}")
        if self.axis_value not in ("channel", "token", "tensor"):
            raise ValueError(f"Invalid axis_value: {self.axis_value}")
        for name in (
            "high_key_num_levels",
            "high_value_num_levels",
            "low_key_num_levels",
            "low_value_num_levels",
        ):
            value = getattr(self, name)
            if not 2 <= value <= 256:
                raise ValueError(f"{name} must be in [2, 256], got {value}.")
        if self.slot_count != 2:
            raise ValueError("Mooncake compression v1 requires exactly two slots.")
        if (
            isinstance(self.logical_blocks_per_chunk, bool)
            or not isinstance(self.logical_blocks_per_chunk, int)
            or self.logical_blocks_per_chunk < 1
        ):
            raise ValueError("logical_blocks_per_chunk must be a positive integer.")
        if not 0.0 < self.codec_warn_ratio <= 1.0:
            raise ValueError("codec_warn_ratio must be in (0, 1].")

    def fingerprint_values(self) -> dict[str, Any]:
        return asdict(self)
