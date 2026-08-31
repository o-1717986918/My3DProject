from types import SimpleNamespace

import numpy as np
import pytest

from mujococodebase.skills.walk.reference_run import ReferenceRunController


class _OnnxValue:
    def __init__(self, width):
        self.shape = [None, width]


class _Session:
    def get_inputs(self):
        return [_OnnxValue(80)]

    def get_outputs(self):
        return [_OnnxValue(23)]


def _write_reference(path):
    frames = 34
    phase = np.linspace(0.0, 2.0 * np.pi, frames, endpoint=False)
    positions = np.zeros((frames, 23), dtype=np.float32)
    positions[:, 14] = 0.2 * np.sin(phase)
    positions[:, 20] = -0.2 * np.sin(phase)
    velocities = np.gradient(positions, 0.02, axis=0).astype(np.float32)
    root_velocity = np.zeros((frames, 3), dtype=np.float32)
    root_velocity[:, 0] = 2.45
    np.savez(
        path,
        joint_position=positions,
        joint_velocity=velocities,
        root_linear_velocity=root_velocity,
    )
    return positions


def _make_agent():
    robot = SimpleNamespace(
        global_orientation_euler=np.zeros(3),
        gyroscope=np.zeros(3),
        _global_cheat_orientation=np.array([0.0, 0.0, 0.0, 1.0]),
    )
    world = SimpleNamespace(
        server_time=1.0,
        number=4,
        playmode=SimpleNamespace(name="PLAY_ON"),
        global_position=np.array([0.0, 0.0, 0.65]),
    )
    return SimpleNamespace(robot=robot, world=world)


def _enable_controller(monkeypatch, tmp_path):
    model_path = tmp_path / "run.onnx"
    model_path.touch()
    reference_path = tmp_path / "reference.npz"
    positions = _write_reference(reference_path)
    monkeypatch.setenv("MY3D_RUN_BACKEND", "reference_v4_burst")
    monkeypatch.setenv("MY3D_RUN_MODEL", str(model_path))
    monkeypatch.setenv("MY3D_RUN_REFERENCE", str(reference_path))
    monkeypatch.setattr(
        "mujococodebase.skills.walk.reference_run.sha256_file",
        lambda path: (
            ReferenceRunController.EXPECTED_MODEL_SHA256
            if path.suffix == ".onnx"
            else ReferenceRunController.EXPECTED_REFERENCE_SHA256
        ),
    )
    monkeypatch.setattr(
        "mujococodebase.skills.walk.reference_run.load_network",
        lambda _: {"session": _Session()},
    )
    return positions


def test_reference_run_is_disabled_by_default(monkeypatch):
    monkeypatch.delenv("MY3D_RUN_BACKEND", raising=False)

    controller = ReferenceRunController(_make_agent())

    assert controller.backend == "stable"
    assert not controller.available
    assert not controller.active


def test_reference_run_blends_exact_assets_and_80_value_observation(
    monkeypatch, tmp_path
):
    positions = _enable_controller(monkeypatch, tmp_path)
    observed = {}

    def infer(obs, model):
        observed["obs"] = obs.copy()
        observed["model"] = model
        return np.zeros(23, dtype=np.float32)

    monkeypatch.setattr("mujococodebase.skills.walk.reference_run.run_network", infer)
    controller = ReferenceRunController(_make_agent())
    stable = np.full(23, 0.1)

    target = controller.step(
        stable_positions_rad=stable,
        current_positions_rad=positions[0].astype(float),
        current_velocities_rad_s=np.zeros(23),
        local_target_delta_m=np.array([5.0, 0.0]),
        heading_error_deg=0.0,
        is_target_absolute=True,
    )

    assert controller.available
    assert controller.active
    assert target is not None
    assert target.blend == pytest.approx(0.025)
    assert target.kp == pytest.approx(25.625)
    assert target.kd == pytest.approx(0.615)
    np.testing.assert_allclose(target.positions_rad, 0.975 * stable)
    assert observed["obs"].shape == (80,)
    np.testing.assert_allclose(observed["obs"][-2:], [1.0, 0.0], atol=1.0e-6)


def test_reference_run_bad_actor_output_disables_and_falls_back(monkeypatch, tmp_path):
    positions = _enable_controller(monkeypatch, tmp_path)
    monkeypatch.setattr(
        "mujococodebase.skills.walk.reference_run.run_network",
        lambda **_: np.zeros(22),
    )
    controller = ReferenceRunController(_make_agent())

    target = controller.step(
        stable_positions_rad=np.zeros(23),
        current_positions_rad=positions[0].astype(float),
        current_velocities_rad_s=np.zeros(23),
        local_target_delta_m=np.array([5.0, 0.0]),
        heading_error_deg=0.0,
        is_target_absolute=True,
    )

    assert target is None
    assert not controller.available
    assert not controller.active


@pytest.mark.parametrize(
    ("mutation", "local_target", "heading", "absolute"),
    [
        (lambda agent: setattr(agent.world, "number", 1), [5.0, 0.0], 0.0, True),
        (
            lambda agent: setattr(agent.world.playmode, "name", "OUR_FREE_KICK"),
            [5.0, 0.0],
            0.0,
            True,
        ),
        (lambda agent: None, [3.0, 0.0], 0.0, True),
        (lambda agent: None, [5.0, 1.0], 12.0, True),
        (lambda agent: None, [5.0, 0.0], 0.0, False),
        (
            lambda agent: agent.robot.global_orientation_euler.__setitem__(0, 6.0),
            [5.0, 0.0],
            0.0,
            True,
        ),
    ],
)
def test_reference_run_rejects_unsafe_entry(
    monkeypatch, tmp_path, mutation, local_target, heading, absolute
):
    positions = _enable_controller(monkeypatch, tmp_path)
    monkeypatch.setattr(
        "mujococodebase.skills.walk.reference_run.run_network",
        lambda **_: np.zeros(23),
    )
    agent = _make_agent()
    mutation(agent)
    controller = ReferenceRunController(agent)

    target = controller.step(
        stable_positions_rad=np.zeros(23),
        current_positions_rad=positions[0].astype(float),
        current_velocities_rad_s=np.zeros(23),
        local_target_delta_m=np.asarray(local_target),
        heading_error_deg=heading,
        is_target_absolute=absolute,
    )

    assert target is None
    assert not controller.active


def test_reference_run_rejects_wrong_hash(monkeypatch, tmp_path):
    _enable_controller(monkeypatch, tmp_path)
    monkeypatch.setattr(
        "mujococodebase.skills.walk.reference_run.sha256_file", lambda _: "wrong"
    )

    controller = ReferenceRunController(_make_agent())

    assert not controller.available


def test_reference_run_blend_window_is_bounded_and_symmetric():
    blends = [
        ReferenceRunController._blend_for_step(step)
        for step in range(ReferenceRunController.BURST_STEPS)
    ]

    assert blends[:4] == pytest.approx([0.025, 0.05, 0.075, 0.1])
    assert blends[-6:] == pytest.approx(
        [0.1, 0.1 * 5 / 6, 0.1 * 4 / 6, 0.1 * 3 / 6, 0.1 * 2 / 6, 0.1 / 6]
    )
    assert all(0.0 < value <= ReferenceRunController.MAX_POSE_BLEND for value in blends)
