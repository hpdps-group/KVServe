"""Online controller: paper cost model + length-aware speeds + EWMA residual.

Does not use the frozen 1/B interval table. Speeds are looked up from the
measured CSV for (machine, input_length) at decision time.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from kvserve_v1.compression.controller.analytical_model import AnalyticalModel
from kvserve_v1.compression.controller.profile import Profile
from kvserve_v1.compression.controller.speed_table import SpeedTable


@dataclass
class Decision:
    profile: Optional[Profile]
    reason: str
    T_hat_ms: float
    T_eff_ms: float
    T_0_ms: float
    B_crit_mbps: Optional[float]
    S_mbps: Optional[float]
    n_feasible: int
    residual_ms: float = 0.0

    @property
    def profile_id(self) -> str:
        return "no_compression" if self.profile is None else self.profile.profile_id

    @property
    def cr(self) -> float:
        return 1.0 if self.profile is None else self.profile.compression_ratio

    @property
    def speedup(self) -> float:
        if self.T_eff_ms <= 0:
            return 1.0
        return self.T_0_ms / self.T_eff_ms


class ResidualBandit:
    """Per-profile EWMA residual, with cooldown after repeated SLO violations."""

    def __init__(self, eta: float = 0.3, violate_window: int = 10, violate_max: int = 3):
        self.eta = eta
        self.violate_window = violate_window
        self.violate_max = violate_max
        self.delta: Dict[str, float] = {}
        self.n: Dict[str, int] = {}
        self.violations: Dict[str, int] = {}
        self.cooldown_left: Dict[str, int] = {}

    def residual(self, profile_id: str) -> float:
        return self.delta.get(profile_id, 0.0)

    def in_cooldown(self, profile_id: str) -> bool:
        return self.cooldown_left.get(profile_id, 0) > 0

    def tick(self) -> None:
        for pid in list(self.cooldown_left):
            self.cooldown_left[pid] = max(0, self.cooldown_left[pid] - 1)

    def update(self, profile_id: str, T_obs_ms: float, T_hat_ms: float, slo_ms: float) -> None:
        d = T_obs_ms - T_hat_ms
        old = self.delta.get(profile_id, 0.0)
        self.delta[profile_id] = (1.0 - self.eta) * old + self.eta * d
        self.n[profile_id] = self.n.get(profile_id, 0) + 1
        if T_obs_ms > slo_ms:
            self.violations[profile_id] = self.violations.get(profile_id, 0) + 1
        if (
            self.violations.get(profile_id, 0) >= self.violate_max
            and self.n.get(profile_id, 0) >= self.violate_window
        ):
            self.cooldown_left[profile_id] = 8
            self.violations[profile_id] = 0


def _pipeline_components(profile: Profile) -> List[str]:
    comps = list((profile.metadata or {}).get("pipeline_components") or [])
    if comps:
        return comps
    return list((profile.compression_config or {}).get("pipeline") or [])


class DynamicOnlineController:
    """Accuracy → B* → SLO → argmin T_eff, with EWMA residual correction."""

    def __init__(
        self,
        profiles: List[Profile],
        speed: SpeedTable,
        eta: float = 0.3,
        epsilon: float = 0.0,
    ):
        self.profiles = profiles
        self.speed = speed
        self.epsilon = epsilon
        self.bandit = ResidualBandit(eta=eta)
        self.model = AnalyticalModel()

    @classmethod
    def from_library_json(
        cls,
        library_json: Path,
        machine: str,
        model: str,
        **kwargs: Any,
    ) -> "DynamicOnlineController":
        data = json.loads(Path(library_json).read_text())
        views: List[Profile] = []
        seen = set()
        for bucket in data["accuracy_buckets"]:
            for raw in bucket["pareto_profiles"]:
                pid = raw["profile_id"]
                if pid in seen:
                    continue
                seen.add(pid)
                views.append(Profile.from_dict(raw))
        return cls(views, SpeedTable(machine, model), **kwargs)

    def select(
        self,
        *,
        B_mbps: float,
        acc_req: float,
        slo_ms: float,
        V_bytes: float,
        T_model_ms: float,
        input_length: int,
        rng: Optional[Any] = None,
    ) -> Decision:
        self.bandit.tick()
        T0 = self.model.t0_ms(V_bytes, B_mbps, T_model_ms)
        feasible: List[Tuple[Profile, float, float, float, float]] = []
        n_acc = 0
        n_benefit = 0

        for p in self.profiles:
            if p.accuracy < acc_req:
                continue
            n_acc += 1
            if self.bandit.in_cooldown(p.profile_id):
                continue
            S = self.speed.pipeline_mbs(_pipeline_components(p), input_length)
            if S is None or S <= 0:
                continue
            if not self.model.benefits(p.compression_ratio, S, B_mbps):
                continue
            n_benefit += 1
            T_hat = self.model.tp_ms(V_bytes, B_mbps, T_model_ms, p.compression_ratio, S)
            delta = self.bandit.residual(p.profile_id)
            T_eff = T_hat + delta
            if T_eff > slo_ms:
                continue
            feasible.append((p, S, self.model.b_star_mbps(p.compression_ratio, S), T_hat, T_eff))

        if not feasible:
            if n_acc == 0:
                reason = "no_accuracy"
            elif n_benefit == 0:
                reason = "no_compression"
            else:
                reason = "slo_infeasible"
            return Decision(
                profile=None, reason=reason,
                T_hat_ms=T0, T_eff_ms=T0, T_0_ms=T0,
                B_crit_mbps=None, S_mbps=None, n_feasible=0,
            )

        if rng is not None and self.epsilon > 0 and rng.random() < self.epsilon:
            chosen = rng.choice(feasible)
            reason = "explore"
        else:
            chosen = min(feasible, key=lambda x: x[4])
            reason = "exploit"

        p, S, Bstar, T_hat, T_eff = chosen
        return Decision(
            profile=p, reason=reason,
            T_hat_ms=T_hat, T_eff_ms=T_eff, T_0_ms=T0,
            B_crit_mbps=Bstar, S_mbps=S, n_feasible=len(feasible),
            residual_ms=self.bandit.residual(p.profile_id),
        )

    def observe(self, decision: Decision, T_obs_ms: float, slo_ms: float) -> None:
        if decision.profile is None:
            return
        self.bandit.update(decision.profile.profile_id, T_obs_ms, decision.T_hat_ms, slo_ms)
