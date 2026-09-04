"""RCSSServerMJ integration for asynchronous EGL rendering and pacing."""

from __future__ import annotations

import io
import logging
import time
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from threading import Condition, Thread
from typing import Any

import mujoco
import numpy as np
from PIL import Image
from rcsssmj.games.soccer.sim.soccer_commands import (
    DropBallCommand,
    KickOffCommand,
    PlaceBallCommand,
)
from rcsssmj.server.remote_monitor import SimMonitor
from rcsssmj.sim.state_info import SceneGraph, SimStateInformation

from mujococodebase.web_match_control import ControlMessage, MatchHub


logger = logging.getLogger(__name__)


def _display_value(value: Any) -> str:
    inner = getattr(value, "value", value)
    return str(inner)


@dataclass(frozen=True)
class _RenderSnapshot:
    model: Any
    frame_id: int
    time: float
    qpos: np.ndarray
    qvel: np.ndarray
    act: np.ndarray
    ctrl: np.ndarray
    mocap_pos: np.ndarray
    mocap_quat: np.ndarray
    status: dict[str, Any]


class WebMujocoMonitor(SimMonitor):
    """Copy poses quickly and render them on an independent EGL thread."""

    _CAMERA_ACTIONS = {
        "camera_cycle",
        "camera_mode",
        "camera_drag",
        "camera_zoom",
    }

    def __init__(
        self,
        model: Any,
        hub: MatchHub,
        *,
        render_interval: int = 4,
        width: int = 1280,
        height: int = 720,
        jpeg_quality: int = 82,
    ) -> None:
        super().__init__(update_interval=render_interval)
        self.hub = hub
        self.model = model
        self.width = width
        self.height = height
        self.jpeg_quality = jpeg_quality
        self.game_state: Any | None = None
        self._snapshot_ready = Condition()
        self._latest_snapshot: _RenderSnapshot | None = None
        self._snapshot_sequence = 0
        self._render_controls: deque[ControlMessage] = deque()
        self._render_shutdown = False
        self._render_thread = Thread(
            target=self._render_loop,
            name="egl_match_renderer",
            daemon=True,
        )
        self._render_thread.start()

    def _route_control(self, control: ControlMessage) -> bool:
        if control.action in self._CAMERA_ACTIONS:
            with self._snapshot_ready:
                self._render_controls.append(control)
                self._snapshot_ready.notify_all()
            return True
        if control.action == "kickoff":
            side = 0 if control.values["side"] == "left" else 1
            self.command_queue.put(KickOffCommand(side))
        elif control.action == "drop_ball":
            self.command_queue.put(DropBallCommand())
        elif control.action == "reset_ball":
            self.command_queue.put(PlaceBallCommand((0.0, 0.0, 0.11), (0, 0, 0)))
        return False

    def _game_status(self) -> dict[str, Any]:
        if self.game_state is None:
            return {}
        return {
            "left_team": self.game_state.left_team or "Left",
            "right_team": self.game_state.right_team or "Right",
            "left_score": self.game_state.left_score,
            "right_score": self.game_state.right_score,
            "play_time": round(float(self.game_state.play_time), 2),
            "play_mode": _display_value(self.game_state.play_mode),
        }

    def _publish_snapshot(self, scene_graph: SceneGraph, frame_id: int) -> None:
        data = scene_graph.mj_data
        snapshot = _RenderSnapshot(
            model=scene_graph.mj_model,
            frame_id=frame_id,
            time=float(data.time),
            qpos=np.array(data.qpos, copy=True),
            qvel=np.array(data.qvel, copy=True),
            act=np.array(data.act, copy=True),
            ctrl=np.array(data.ctrl, copy=True),
            mocap_pos=np.array(data.mocap_pos, copy=True),
            mocap_quat=np.array(data.mocap_quat, copy=True),
            status=self._game_status(),
        )
        with self._snapshot_ready:
            self._latest_snapshot = snapshot
            self._snapshot_sequence += 1
            self._snapshot_ready.notify_all()

    def update(
        self,
        state_info: Sequence[SimStateInformation],
        frame_id: int,
        last_recompilation: int,
    ) -> None:
        del last_recompilation
        scene_graph = None
        for info in state_info:
            if isinstance(info, SceneGraph):
                scene_graph = info
            elif info.__class__.__name__ == "SoccerGameInformation":
                self.game_state = info

        force_snapshot = False
        for control in self.hub.poll_controls():
            force_snapshot |= self._route_control(control)

        if scene_graph is not None and (
            force_snapshot or frame_id % self.update_interval == 0
        ):
            self._publish_snapshot(scene_graph, frame_id)

    def _configure_camera(self, model: Any, camera: Any, mode: str) -> str:
        if mode == "static":
            camera.fixedcamid = -1
            camera.type = mujoco.mjtCamera.mjCAMERA_FREE
            camera.trackbodyid = -1
            camera.distance = 15.0
            camera.elevation = -45.0
            camera.azimuth = 90.0
        else:
            camera.fixedcamid = -1
            camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            camera.trackbodyid = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_BODY, "ball"
            )
            camera.distance = 3.5
            camera.elevation = -12.0
            camera.azimuth = 90.0
        self.hub.update_status({"camera_mode": mode})
        return mode

    def _apply_camera_control(
        self,
        control: ControlMessage,
        model: Any,
        renderer: mujoco.Renderer,
        camera: Any,
        mode: str,
    ) -> str:
        action = control.action
        if action == "camera_cycle":
            return self._configure_camera(
                model, camera, "follow" if mode == "static" else "static"
            )
        if action == "camera_mode":
            return self._configure_camera(
                model, camera, str(control.values["mode"])
            )
        if action == "camera_drag":
            mouse_action = (
                mujoco.mjtMouse.mjMOUSE_ROTATE_V
                if control.values["mode"] == "orbit"
                else mujoco.mjtMouse.mjMOUSE_MOVE_V
            )
            mujoco.mjv_moveCamera(
                model,
                mouse_action,
                float(control.values["dx"]) / self.width,
                float(control.values["dy"]) / self.height,
                renderer.scene,
                camera,
            )
        elif action == "camera_zoom":
            mujoco.mjv_moveCamera(
                model,
                mujoco.mjtMouse.mjMOUSE_ZOOM,
                0,
                0.05 * float(control.values["delta"]),
                renderer.scene,
                camera,
            )
        return mode

    def _create_renderer(self, model: Any) -> tuple[mujoco.Renderer, Any, Any]:
        model.vis.global_.offwidth = max(model.vis.global_.offwidth, self.width)
        model.vis.global_.offheight = max(model.vis.global_.offheight, self.height)
        renderer = mujoco.Renderer(
            model,
            height=self.height,
            width=self.width,
            max_geom=2000,
        )
        data = mujoco.MjData(model)
        camera = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(model, camera)
        return renderer, data, camera

    @staticmethod
    def _load_snapshot(data: Any, snapshot: _RenderSnapshot) -> None:
        data.time = snapshot.time
        data.qpos[:] = snapshot.qpos
        data.qvel[:] = snapshot.qvel
        if data.act.size:
            data.act[:] = snapshot.act
        data.ctrl[:] = snapshot.ctrl
        if data.mocap_pos.size:
            data.mocap_pos[:] = snapshot.mocap_pos
            data.mocap_quat[:] = snapshot.mocap_quat

    def _render_loop(self) -> None:
        renderer: mujoco.Renderer | None = None
        render_model = None
        render_data = None
        camera = None
        camera_mode = "static"
        handled_sequence = 0
        last_frame_at = time.monotonic()
        render_fps = 0.0
        scene_option = mujoco.MjvOption()

        try:
            while True:
                with self._snapshot_ready:
                    self._snapshot_ready.wait_for(
                        lambda: (
                            self._render_shutdown
                            or self._snapshot_sequence > handled_sequence
                            or bool(self._render_controls)
                        )
                    )
                    if self._render_shutdown:
                        break
                    snapshot = self._latest_snapshot
                    handled_sequence = self._snapshot_sequence
                    controls = list(self._render_controls)
                    self._render_controls.clear()

                if snapshot is None:
                    continue
                if render_model is not snapshot.model:
                    if renderer is not None:
                        renderer.close()
                    render_model = snapshot.model
                    renderer, render_data, camera = self._create_renderer(
                        render_model
                    )
                    camera_mode = self._configure_camera(
                        render_model, camera, camera_mode
                    )

                assert renderer is not None
                assert render_data is not None
                assert camera is not None
                self._load_snapshot(render_data, snapshot)
                mujoco.mj_forward(render_model, render_data)
                for control in controls:
                    camera_mode = self._apply_camera_control(
                        control,
                        render_model,
                        renderer,
                        camera,
                        camera_mode,
                    )

                renderer.update_scene(
                    render_data,
                    camera=camera,
                    scene_option=scene_option,
                )
                renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
                renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0
                pixels = renderer.render()
                output = io.BytesIO()
                Image.fromarray(pixels).save(
                    output,
                    format="JPEG",
                    quality=self.jpeg_quality,
                    subsampling=1,
                )

                now = time.monotonic()
                instant_fps = 1.0 / max(now - last_frame_at, 1e-6)
                last_frame_at = now
                render_fps = (
                    instant_fps
                    if render_fps == 0
                    else render_fps * 0.85 + instant_fps * 0.15
                )
                self.hub.publish_frame(
                    output.getvalue(),
                    {
                        **snapshot.status,
                        "server_state": "running",
                        "frame_id": snapshot.frame_id,
                        "camera_mode": camera_mode,
                        "render_fps": round(render_fps, 1),
                    },
                )
        except Exception as error:  # render failure must not stop match physics
            logger.exception("EGL match renderer stopped unexpectedly")
            self.hub.update_status(
                {"server_state": "render_error", "render_error": str(error)}
            )
        finally:
            if renderer is not None:
                renderer.close()

    def shutdown(self) -> None:
        super().shutdown()
        with self._snapshot_ready:
            self._render_shutdown = True
            self._snapshot_ready.notify_all()
        self._render_thread.join(timeout=10)


class WebPacedServerMixin:
    """Add browser-controlled pause, step, and safe bounded speed to a server."""

    hub: MatchHub

    def _parallel_update_loop(self) -> None:
        sim_timestep = self.sim.timestep
        cycle_start = time.monotonic() - sim_timestep

        while not self._shutdown:
            turn = self.hub.wait_for_turn()
            if turn is None:
                break

            if turn == "control":
                active_monitors, monitors_to_remove = self._filter_monitors()
                self._update_monitors(active_monitors)
                self._remove_monitors(monitors_to_remove)
                continue

            manual_step = turn == "step"
            _, ready_agents, active_agents, disconnected_agents = (
                self._filter_agents()
            )
            active_monitors, monitors_to_remove = self._filter_monitors()
            self._deactivate_agents(disconnected_agents)
            activated_agents = self._activate_agents(ready_agents)
            self.sim.generate_perceptions()

            if self.real_time and not manual_step:
                target_interval = sim_timestep / self.hub.speed
                delay = max(0.0, target_interval - (time.monotonic() - cycle_start))
                if not self.hub.wait_delay(delay):
                    cycle_start = time.monotonic()
                    continue
            cycle_start = time.monotonic()

            self._collect_actions(active_agents, block=self.sync_mode)
            self._send_perceptions(*active_agents, *activated_agents)
            monitor_commands = self._collect_commands(active_monitors)
            monitor_commands.extend(self._collect_commands(monitors_to_remove))
            self.sim.step(monitor_commands)
            self._update_monitors(active_monitors)
            self._remove_agents(disconnected_agents)
            self._remove_monitors(monitors_to_remove)

    def shutdown(self) -> None:
        self.hub.stop()
        super().shutdown()
