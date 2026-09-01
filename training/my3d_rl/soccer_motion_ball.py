"""Shared exact-physics geometry for motion-guided ball screening.

The PAiD soccer task places the ball at the final reference-anchor position.
Finite-phase restarts must preserve that geometry relative to the restarted
root, rather than teleporting the ball to a fixed robot-local offset.  This
module keeps that ownership rule and the strict contact classification shared
between evaluators and future K2 environments.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ReferenceStrike:
    """Kinematic closest-approach frame for the labelled kicking foot."""

    frame: int
    center_distance_m: float
    planar_speed_mps: float


@dataclass(frozen=True)
class BallContactSides:
    """Ball-contact classification for one exact MuJoCo substep."""

    any_robot: bool
    left_foot: bool
    right_foot: bool


@dataclass(frozen=True)
class BallPlacementPerturbation:
    """Deterministic ball endpoint perturbation with hashable seed provenance."""

    case_seed: int
    radius_offset_m: float
    arc_angle_rad: float


def deterministic_ball_placement_perturbation(
    *,
    base_seed: int,
    motion: int,
    start_frame: int,
    radius_noise_m: float,
    arc_noise_rad: float,
) -> BallPlacementPerturbation:
    """Sample a disjoint per-case ball perturbation using NumPy SeedSequence."""

    if min(base_seed, motion, start_frame) < 0:
        raise ValueError("ball perturbation seeds and coordinates must be non-negative")
    if min(radius_noise_m, arc_noise_rad) < 0.0 or not np.isfinite(
        [radius_noise_m, arc_noise_rad]
    ).all():
        raise ValueError("ball perturbation ranges must be finite and non-negative")
    sequence = np.random.SeedSequence(
        [int(base_seed), int(motion), int(start_frame), 0xB411]
    )
    case_seed = int(sequence.generate_state(1, dtype=np.uint64)[0]) & 0x7FFF_FFFF_FFFF_FFFF
    rng = np.random.default_rng(case_seed)
    return BallPlacementPerturbation(
        case_seed=case_seed,
        radius_offset_m=float(rng.uniform(-radius_noise_m, radius_noise_m)),
        arc_angle_rad=float(rng.uniform(-arc_noise_rad, arc_noise_rad)),
    )


def place_reference_ball_xy(
    root_positions_xy: np.ndarray,
    *,
    start_frame: int,
    model_root_xy: np.ndarray,
    yaw_rad: float = 0.0,
    radius_offset_m: float = 0.0,
    arc_angle_rad: float = 0.0,
) -> np.ndarray:
    """Place a ball at the remaining reference-anchor endpoint.

    The reference displacement from ``start_frame`` to the last frame is
    rotated by the reset yaw.  Radius and arc perturbations follow PAiD's
    endpoint-arc construction while remaining deterministic for an evaluator.
    """

    roots = np.asarray(root_positions_xy, dtype=np.float64)
    origin = np.asarray(model_root_xy, dtype=np.float64)
    values = np.asarray(
        [yaw_rad, radius_offset_m, arc_angle_rad], dtype=np.float64
    )
    if roots.ndim != 2 or roots.shape[1] != 2 or roots.shape[0] < 2:
        raise ValueError("root_positions_xy must have shape [T, 2], T >= 2")
    if origin.shape != (2,):
        raise ValueError("model_root_xy must have shape [2]")
    if not np.isfinite(roots).all() or not np.isfinite(origin).all():
        raise ValueError("ball-placement positions must be finite")
    if not np.isfinite(values).all():
        raise ValueError("ball-placement perturbations must be finite")
    if not 0 <= start_frame < roots.shape[0] - 1:
        raise ValueError("start_frame must leave at least one reference frame")

    remaining = roots[-1] - roots[start_frame]
    radius = float(np.linalg.norm(remaining)) + float(radius_offset_m)
    if radius < 0.0:
        raise ValueError("radius offset makes the reference radius negative")
    if np.linalg.norm(remaining) > 1.0e-12:
        base_angle = float(np.arctan2(remaining[1], remaining[0]))
    else:
        base_angle = 0.0
    angle = base_angle + float(arc_angle_rad) + float(yaw_rad)
    return origin + radius * np.array([np.cos(angle), np.sin(angle)])


def select_reference_strike(
    foot_center_positions: np.ndarray,
    ball_center_position: np.ndarray,
    *,
    frequency_hz: float,
) -> ReferenceStrike:
    """Select the earliest closest foot/ball-center approach in a reference."""

    positions = np.asarray(foot_center_positions, dtype=np.float64)
    ball = np.asarray(ball_center_position, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3 or positions.shape[0] < 2:
        raise ValueError("foot positions must have shape [T, 3], T >= 2")
    if ball.shape != (3,):
        raise ValueError("ball center must have shape [3]")
    if not np.isfinite(positions).all() or not np.isfinite(ball).all():
        raise ValueError("reference strike geometry must be finite")
    if not np.isfinite(frequency_hz) or frequency_hz <= 0.0:
        raise ValueError("frequency_hz must be positive and finite")

    distances = np.linalg.norm(positions - ball, axis=1)
    frame = int(np.argmin(distances))
    planar_velocity = np.gradient(positions[:, :2], axis=0) * frequency_hz
    return ReferenceStrike(
        frame=frame,
        center_distance_m=float(distances[frame]),
        planar_speed_mps=float(np.linalg.norm(planar_velocity[frame])),
    )


def classify_ball_contacts(
    geom_pairs: np.ndarray,
    *,
    ball_geom: int,
    left_foot_geom: int,
    right_foot_geom: int,
    robot_geoms: frozenset[int],
) -> BallContactSides:
    """Classify exact contact pairs without inferring contact from ball motion."""

    pairs = np.asarray(geom_pairs, dtype=np.int64)
    if pairs.size == 0:
        pairs = np.empty((0, 2), dtype=np.int64)
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError("geom_pairs must have shape [N, 2]")
    if len({ball_geom, left_foot_geom, right_foot_geom}) != 3:
        raise ValueError("ball and foot geom identifiers must be distinct")

    partners: set[int] = set()
    for first, second in pairs.tolist():
        if first == ball_geom:
            partners.add(int(second))
        elif second == ball_geom:
            partners.add(int(first))
    return BallContactSides(
        any_robot=bool(partners.intersection(robot_geoms)),
        left_foot=left_foot_geom in partners,
        right_foot=right_foot_geom in partners,
    )
