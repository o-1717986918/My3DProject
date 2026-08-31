"""Guarded competition adapter for the experimental reference-run policy.

The accepted walk policy remains in control on every cycle.  This adapter can
blend a short, straight-line burst from the externally stored v4 actor and GMR
reference when (and only when) the operator explicitly enables the backend.
It is intentionally self-disabling: missing or unexpected assets, bad actor
outputs, or unsafe robot state all return control to the stable walk target in
the same cycle.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import logging
import os
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

from mujococodebase.utils.neural_network import load_network, run_network


logger = logging.getLogger(__name__)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class BlendedRunTarget:
    """One guarded run target expressed in the server's physical convention."""

    positions_rad: np.ndarray
    kp: float
    kd: float
    blend: float


class ReferenceRunController:
    """Blend a bounded v4 reference-policy burst into stable locomotion.

    This is a deployment adapter, not a release promotion.  The v4 policy did
    not pass the ten-second running gate, so the adapter is opt-in and limits a
    burst to less than the measured p10 fall horizon.  Stable walk targets are
    calculated independently and are supplied to :meth:`step` as the immediate
    fallback.
    """

    BACKEND_NAME = "reference_v4_burst"
    EXPECTED_MODEL_SHA256 = (
        "a107ffe6591ab89ad23018a6a65396fd3d2c59cd31924585fe482cf0606f4fa7"
    )
    EXPECTED_REFERENCE_SHA256 = (
        "02cd640919d81f0417246559bae491439e7afbfda614039d5ecae1293076c523"
    )
    TRAIN_TO_SERVER_SIGN = np.array(
        [
            1.0,
            -1.0,
            1.0,
            -1.0,
            -1.0,
            1.0,
            -1.0,
            -1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            -1.0,
            -1.0,
            1.0,
            1.0,
            -1.0,
            -1.0,
            -1.0,
            -1.0,
            -1.0,
            -1.0,
            -1.0,
        ],
        dtype=np.float64,
    )

    CONTROL_DT = 0.02
    CONTROL_FREQUENCY_HZ = 50.0
    COMMAND = np.array([1.8, 0.0, 0.0], dtype=np.float64)
    ACTION_SCALE = 0.15
    RUN_KP = 50.0
    RUN_KD = 1.2
    WALK_KP = 25.0
    WALK_KD = 0.6

    # 16 cycles = 0.32 s, below the measured 0.40 s p10 survival horizon of
    # the rejected ten-second v4 candidate.  The ends are smoothly blended.
    BURST_STEPS = 16
    BLEND_IN_STEPS = 4
    BLEND_OUT_STEPS = 6
    # The current candidate cannot safely own the full target.  Competition
    # integration is therefore a posture hint capped at ten percent while the
    # independently evaluated stable walk remains the dominant controller.
    MAX_POSE_BLEND = 0.10
    COOLDOWN_SECONDS = 2.0

    MIN_TARGET_DISTANCE_M = 3.5
    MAX_HEADING_ERROR_DEG = 6.0
    MAX_ENTRY_TILT_DEG = 5.0
    MAX_ACTIVE_TILT_DEG = 10.0
    MAX_ENTRY_GYRO_DEG_S = 35.0
    MAX_ACTIVE_GYRO_DEG_S = 120.0
    MIN_ENTRY_HEIGHT_M = 0.55
    MIN_ACTIVE_HEIGHT_M = 0.48
    MAX_REFERENCE_ERROR_RMS_RAD = 1.0
    MAX_RUN_TARGET_DELTA_RAD = 0.18

    def __init__(self, agent) -> None:
        self.agent = agent
        self.backend = os.environ.get("MY3D_RUN_BACKEND", "stable").strip().lower()
        self.available = False
        self.active = False
        self._model = None
        self._reference_position_physical = np.empty((0, 23), dtype=np.float64)
        self._reference_velocity_physical = np.empty((0, 23), dtype=np.float64)
        self._reference_forward_speed = 1.0
        self._nominal_frequency = 1.0
        self._phase = 0.0
        self._active_step = 0
        self._previous_action = np.zeros(23, dtype=np.float64)
        self._next_allowed_time = 0.0

        if self.backend != self.BACKEND_NAME:
            return
        self._load_assets()

    def _load_assets(self) -> None:
        model_value = os.environ.get("MY3D_RUN_MODEL", "").strip()
        reference_value = os.environ.get("MY3D_RUN_REFERENCE", "").strip()
        if not model_value or not reference_value:
            logger.warning(
                "reference-run disabled: MY3D_RUN_MODEL and "
                "MY3D_RUN_REFERENCE are both required"
            )
            return

        model_path = Path(model_value).expanduser().resolve()
        reference_path = Path(reference_value).expanduser().resolve()
        if not model_path.is_file() or not reference_path.is_file():
            logger.warning(
                "reference-run disabled: model/reference asset does not exist"
            )
            return
        try:
            if sha256_file(model_path) != self.EXPECTED_MODEL_SHA256:
                raise ValueError("model SHA-256 is not approved")
            if sha256_file(reference_path) != self.EXPECTED_REFERENCE_SHA256:
                raise ValueError("reference SHA-256 is not approved")
            with np.load(reference_path, allow_pickle=False) as archive:
                position = np.asarray(archive["joint_position"], dtype=np.float64)
                velocity = np.asarray(archive["joint_velocity"], dtype=np.float64)
                root_velocity = np.asarray(
                    archive["root_linear_velocity"], dtype=np.float64
                )
            if (
                position.shape != (34, 23)
                or velocity.shape != position.shape
                or root_velocity.shape != (34, 3)
                or not np.isfinite(position).all()
                or not np.isfinite(velocity).all()
                or not np.isfinite(root_velocity).all()
            ):
                raise ValueError("motion reference has an incompatible payload")
            reference_speed = float(np.mean(root_velocity[:, 0]))
            if not 1.0 <= reference_speed <= 4.5:
                raise ValueError("motion reference forward speed is outside its gate")

            model = load_network(str(model_path))
            self._validate_model_interface(model)
        except (KeyError, OSError, RuntimeError, ValueError) as exc:
            logger.exception("reference-run disabled: asset validation failed: %s", exc)
            return

        self._reference_position_physical = position
        self._reference_velocity_physical = velocity
        self._reference_forward_speed = reference_speed
        self._nominal_frequency = self.CONTROL_FREQUENCY_HZ / position.shape[0]
        self._model = model
        self.available = True
        logger.info(
            "reference-run ready: backend=%s frames=%d burst=%.2fs",
            self.backend,
            position.shape[0],
            self.BURST_STEPS * self.CONTROL_DT,
        )

    @staticmethod
    def _validate_model_interface(model) -> None:
        try:
            model_input = model["session"].get_inputs()[0]
            model_output = model["session"].get_outputs()[0]
            input_width = int(model_input.shape[-1])
            output_width = int(model_output.shape[-1])
        except (AttributeError, IndexError, KeyError, TypeError, ValueError) as exc:
            raise ValueError("cannot inspect ONNX input/output shapes") from exc
        if input_width != 80 or output_width != 23:
            raise ValueError(
                f"ONNX interface is {input_width}->{output_width}, expected 80->23"
            )

    def reset(self, server_time: float | None = None) -> None:
        """Cancel a burst when the parent Walk skill is re-entered."""
        if self.active:
            self._finish(float(server_time or 0.0), reason="walk-reset")
        self._previous_action.fill(0.0)

    def step(
        self,
        *,
        stable_positions_rad: np.ndarray,
        current_positions_rad: np.ndarray,
        current_velocities_rad_s: np.ndarray,
        local_target_delta_m: np.ndarray,
        heading_error_deg: float,
        is_target_absolute: bool,
    ) -> BlendedRunTarget | None:
        """Return a blended run target, or ``None`` for immediate stable walk."""
        if not self.available:
            return None

        now = float(self.agent.world.server_time or 0.0)
        if not self.active:
            if not self._entry_is_safe(
                now,
                local_target_delta_m,
                heading_error_deg,
                is_target_absolute,
            ):
                return None
            self._start(current_positions_rad)

        playmode_name = getattr(
            getattr(self.agent.world, "playmode", None), "name", None
        )
        if playmode_name != "PLAY_ON":
            self._finish(now, reason="playmode-guard")
            return None
        if not self._active_state_is_safe(current_positions_rad):
            self._finish(now, reason="posture-guard")
            return None

        try:
            reference_position, reference_velocity = self._reference_at_phase(
                self._phase
            )
            joint_position_training = current_positions_rad * self.TRAIN_TO_SERVER_SIGN
            joint_velocity_training = (
                current_velocities_rad_s * self.TRAIN_TO_SERVER_SIGN
            )
            reference_position_training = reference_position * self.TRAIN_TO_SERVER_SIGN
            velocity_scale = self.COMMAND[0] / self._reference_forward_speed
            reference_velocity_training = (
                velocity_scale * reference_velocity * self.TRAIN_TO_SERVER_SIGN
            )
            triplets = np.stack(
                [
                    (joint_position_training - reference_position_training) / 4.6,
                    (joint_velocity_training - reference_velocity_training) / 110.0,
                    self._previous_action / 10.0,
                ],
                axis=1,
            ).reshape(-1)
            robot = self.agent.robot
            angular_velocity = np.deg2rad(robot.gyroscope) / 50.0
            gravity = (
                R.from_quat(robot._global_cheat_orientation)
                .inv()
                .apply(np.array([0.0, 0.0, -1.0]))
            )
            phase_angle = 2.0 * np.pi * self._phase
            observation = np.concatenate(
                [
                    triplets,
                    angular_velocity,
                    self.COMMAND,
                    gravity,
                    np.array([np.cos(phase_angle), np.sin(phase_angle)]),
                ]
            )
            observation = np.nan_to_num(observation, nan=0.0, posinf=10.0, neginf=-10.0)
            observation = np.clip(observation, -10.0, 10.0)
            action = np.asarray(
                run_network(obs=observation, model=self._model), dtype=np.float64
            )
            if action.shape != (23,) or not np.isfinite(action).all():
                raise ValueError(f"actor returned invalid shape/values: {action.shape}")
        except Exception as exc:
            # Inference is an optional trust boundary. Any library/runtime
            # exception disables only this backend; stable walk still owns the
            # already computed target for the current cycle.
            logger.error("reference-run inference failed; disabling backend: %s", exc)
            self.available = False
            self._finish(now, reason="inference-error")
            return None

        action = np.clip(action, -1.0, 1.0)
        run_positions = (
            reference_position_training + self.ACTION_SCALE * action
        ) * self.TRAIN_TO_SERVER_SIGN
        run_positions = np.clip(
            run_positions,
            current_positions_rad - self.MAX_RUN_TARGET_DELTA_RAD,
            current_positions_rad + self.MAX_RUN_TARGET_DELTA_RAD,
        )
        blend = self._blend_for_step(self._active_step)
        positions = (1.0 - blend) * stable_positions_rad + blend * run_positions
        target = BlendedRunTarget(
            positions_rad=positions,
            kp=(1.0 - blend) * self.WALK_KP + blend * self.RUN_KP,
            kd=(1.0 - blend) * self.WALK_KD + blend * self.RUN_KD,
            blend=blend,
        )

        self._previous_action = action
        gait_frequency = (
            self._nominal_frequency
            * abs(float(self.COMMAND[0]))
            / self._reference_forward_speed
        )
        self._phase = (self._phase + self.CONTROL_DT * gait_frequency) % 1.0
        self._active_step += 1
        if self._active_step >= self.BURST_STEPS:
            self._finish(now, reason="completed")
        return target

    def _entry_is_safe(
        self,
        now: float,
        local_target_delta_m: np.ndarray,
        heading_error_deg: float,
        is_target_absolute: bool,
    ) -> bool:
        world = self.agent.world
        robot = self.agent.robot
        if (
            now < self._next_allowed_time
            or not is_target_absolute
            or getattr(world, "number", 1) == 1
            or getattr(getattr(world, "playmode", None), "name", None) != "PLAY_ON"
        ):
            return False
        delta = np.asarray(local_target_delta_m, dtype=np.float64)
        if delta.shape != (2,) or not np.isfinite(delta).all():
            return False
        if (
            np.linalg.norm(delta) < self.MIN_TARGET_DISTANCE_M
            or delta[0] <= 0.0
            or abs(float(heading_error_deg)) > self.MAX_HEADING_ERROR_DEG
        ):
            return False
        attitude = np.asarray(robot.global_orientation_euler, dtype=np.float64)
        gyro = np.asarray(robot.gyroscope, dtype=np.float64)
        height = float(world.global_position[2])
        if (
            attitude.shape != (3,)
            or gyro.shape != (3,)
            or not np.isfinite(attitude).all()
            or not np.isfinite(gyro).all()
            or not np.isfinite(height)
        ):
            return False
        roll, pitch = attitude[:2]
        if (
            max(abs(float(roll)), abs(float(pitch))) > self.MAX_ENTRY_TILT_DEG
            or np.max(np.abs(gyro)) > self.MAX_ENTRY_GYRO_DEG_S
            or height < self.MIN_ENTRY_HEIGHT_M
        ):
            return False
        return True

    def _active_state_is_safe(self, current_positions_rad: np.ndarray) -> bool:
        world = self.agent.world
        robot = self.agent.robot
        positions = np.asarray(current_positions_rad, dtype=np.float64)
        attitude = np.asarray(robot.global_orientation_euler, dtype=np.float64)
        gyro = np.asarray(robot.gyroscope, dtype=np.float64)
        height = float(world.global_position[2])
        if (
            positions.shape != (23,)
            or attitude.shape != (3,)
            or gyro.shape != (3,)
            or not np.isfinite(positions).all()
            or not np.isfinite(attitude).all()
            or not np.isfinite(gyro).all()
            or not np.isfinite(height)
        ):
            return False
        roll, pitch = attitude[:2]
        if (
            max(abs(float(roll)), abs(float(pitch))) > self.MAX_ACTIVE_TILT_DEG
            or np.max(np.abs(gyro)) > self.MAX_ACTIVE_GYRO_DEG_S
            or height < self.MIN_ACTIVE_HEIGHT_M
        ):
            return False
        reference_position, _ = self._reference_at_phase(self._phase)
        error = float(np.sqrt(np.mean(np.square(positions - reference_position))))
        return np.isfinite(error) and error <= self.MAX_REFERENCE_ERROR_RMS_RAD

    def _start(self, current_positions_rad: np.ndarray) -> None:
        squared_error = np.mean(
            np.square(
                self._reference_position_physical
                - np.asarray(current_positions_rad, dtype=np.float64)[None, :]
            ),
            axis=1,
        )
        nearest_frame = int(np.argmin(squared_error))
        self._phase = nearest_frame / self._reference_position_physical.shape[0]
        self._active_step = 0
        self._previous_action.fill(0.0)
        self.active = True
        logger.info(
            "reference-run burst activated: player=%d phase=%.3f",
            getattr(self.agent.world, "number", -1),
            self._phase,
        )

    def _finish(self, now: float, *, reason: str) -> None:
        was_active = self.active
        self.active = False
        self._active_step = 0
        self._previous_action.fill(0.0)
        self._next_allowed_time = max(
            self._next_allowed_time, now + self.COOLDOWN_SECONDS
        )
        if was_active:
            logger.info(
                "reference-run burst %s: player=%d",
                reason,
                getattr(self.agent.world, "number", -1),
            )

    def _reference_at_phase(self, phase: float) -> tuple[np.ndarray, np.ndarray]:
        frame = (float(phase) % 1.0) * self._reference_position_physical.shape[0]
        lower = int(np.floor(frame)) % self._reference_position_physical.shape[0]
        upper = (lower + 1) % self._reference_position_physical.shape[0]
        fraction = frame - np.floor(frame)
        position = (1.0 - fraction) * self._reference_position_physical[
            lower
        ] + fraction * self._reference_position_physical[upper]
        velocity = (1.0 - fraction) * self._reference_velocity_physical[
            lower
        ] + fraction * self._reference_velocity_physical[upper]
        return position, velocity

    @classmethod
    def _blend_for_step(cls, step: int) -> float:
        blend_in = min(1.0, (step + 1) / cls.BLEND_IN_STEPS)
        remaining = cls.BURST_STEPS - step
        blend_out = min(1.0, remaining / cls.BLEND_OUT_STEPS)
        return float(cls.MAX_POSE_BLEND * min(blend_in, blend_out))
