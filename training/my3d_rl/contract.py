"""Validation for the policy/runtime boundary.

Training code and the competition client both depend on this ordering. Failing
early here is cheaper than discovering a silent joint permutation in a match.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


class ContractError(ValueError):
    """Raised when a policy manifest cannot be deployed safely."""


@dataclass(frozen=True)
class PolicyContract:
    policy_name: str
    frequency_hz: int
    joint_order: tuple[str, ...]
    effector_order: tuple[str, ...]
    action_size: int
    observation_size: int
    observation_fields: tuple[tuple[str, int], ...]
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    action_clip: tuple[float, float]
    action_scale: float | None
    kp: float | None
    kd: float | None


def _require(mapping: dict[str, Any], key: str, context: str) -> Any:
    if key not in mapping:
        raise ContractError(f"missing {context}.{key}")
    return mapping[key]


def load_policy_contract(path: str | Path) -> PolicyContract:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ContractError("contract root must be a mapping")

    control = _require(raw, "control", "root")
    actor = _require(raw, "actor_observation", "root")
    deployment = _require(raw, "deployment", "root")
    joints = tuple(_require(raw, "joint_order", "root"))
    effectors = tuple(_require(raw, "effector_order", "root"))
    fields_raw = _require(actor, "fields", "actor_observation")
    fields = tuple((str(item["name"]), int(item["size"])) for item in fields_raw)

    action_size = int(_require(control, "action_size", "control"))
    observation_size = int(_require(actor, "size", "actor_observation"))
    input_shape = tuple(int(value) for value in deployment["input_shape"])
    output_shape = tuple(int(value) for value in deployment["output_shape"])
    action_clip = tuple(float(value) for value in control["action_clip"])
    action_scale = float(control["action_scale"]) if "action_scale" in control else None
    kp = float(control["kp"]) if "kp" in control else None
    kd = float(control["kd"]) if "kd" in control else None

    if len(joints) != action_size:
        raise ContractError(
            f"joint count {len(joints)} does not match action size {action_size}"
        )
    if len(set(joints)) != len(joints):
        raise ContractError("joint_order contains duplicates")
    if len(effectors) != action_size or len(set(effectors)) != len(effectors):
        raise ContractError("effector_order must contain one unique name per action")
    if sum(size for _, size in fields) != observation_size:
        raise ContractError("observation field sizes do not match declared size")
    if input_shape != (1, observation_size):
        raise ContractError("ONNX input shape does not match observation size")
    if output_shape != (1, action_size):
        raise ContractError("ONNX output shape does not match action size")
    if int(control["frequency_hz"]) != 50:
        raise ContractError("competition motion policies must run at 50 Hz")
    if len(action_clip) != 2 or action_clip[0] >= action_clip[1]:
        raise ContractError("control.action_clip must be an increasing pair")
    if action_scale is not None and not 0.0 < action_scale <= 1.0:
        raise ContractError("control.action_scale must be in (0, 1]")
    if (kp is None) != (kd is None):
        raise ContractError("control.kp and control.kd must be declared together")
    if kp is not None and kd is not None and (kp <= 0.0 or kd < 0.0):
        raise ContractError("control gains must satisfy kp > 0 and kd >= 0")

    return PolicyContract(
        policy_name=str(_require(raw, "policy_name", "root")),
        frequency_hz=int(control["frequency_hz"]),
        joint_order=joints,
        effector_order=effectors,
        action_size=action_size,
        observation_size=observation_size,
        observation_fields=fields,
        input_shape=input_shape,
        output_shape=output_shape,
        action_clip=action_clip,
        action_scale=action_scale,
        kp=kp,
        kd=kd,
    )
