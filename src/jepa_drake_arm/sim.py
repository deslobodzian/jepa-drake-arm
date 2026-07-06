import os

import numpy as np

from pydrake.geometry import (
    Box,
    ClippingRange,
    ColorRenderCamera,
    Cylinder,
    DepthRange,
    DepthRenderCamera,
    MakeRenderEngineVtk,
    RenderCameraCore,
    RenderEngineVtkParams,
)
from pydrake.math import RigidTransform, RotationMatrix
from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph, MultibodyPlant
from pydrake.systems.analysis import Simulator
from pydrake.systems.controllers import InverseDynamicsController
from pydrake.systems.framework import DiagramBuilder
from pydrake.systems.primitives import Multiplexer, TrajectorySource
from pydrake.systems.sensors import CameraInfo, RgbdSensor

from .motions import NUM_JOINTS, MotionParams, make_joint_trajectory

IIWA_URL = "package://drake_models/iiwa_description/urdf/iiwa14_primitive_collision.urdf"
RENDERER_NAME = "vjepa_vtk"

# Supported arms. V-JEPA2-AC was trained on Franka data (DROID), so the panda
# is the right embodiment for action-conditioned experiments; the iiwa is the
# default for the encoder-only experiments.
ARMS = {
    "iiwa": dict(
        url=IIWA_URL,
        base_frame="base",
        ee_frames=("iiwa_link_ee", "iiwa_link_7"),
        q_home=np.array([0.0, 0.5, 0.0, -1.7, 0.0, 1.0, 0.0]),
    ),
    "panda": dict(
        url="package://drake_models/franka_description/urdf/panda_arm.urdf",
        base_frame="panda_link0",
        ee_frames=("panda_link8",),
        q_home=np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]),
    ),
}


def _look_at(eye, target, up=(0.0, 0.0, 1.0)) -> RigidTransform:
    """Camera pose with +Z looking at `target` (Drake camera convention: +Z
    forward, +X right, +Y down in the image)."""
    eye = np.asarray(eye, dtype=float)
    z = np.asarray(target, dtype=float) - eye
    z /= np.linalg.norm(z)
    x = np.cross(-np.asarray(up, dtype=float), z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = RotationMatrix(np.column_stack([x, y, z]))
    return RigidTransform(R, eye)


def _add_static_scenery(plant: MultibodyPlant) -> None:
    """Visual-only ground plane and landmarks so the video has spatial context."""
    world = plant.world_body()
    plant.RegisterVisualGeometry(
        world, RigidTransform([0.0, 0.0, -0.01]), Box(3.5, 3.5, 0.02),
        "ground", [0.82, 0.82, 0.85, 1.0])
    plant.RegisterVisualGeometry(
        world, RigidTransform([0.85, 0.7, 0.1]), Box(0.2, 0.2, 0.2),
        "landmark_box", [0.16, 0.47, 0.84, 1.0])
    plant.RegisterVisualGeometry(
        world, RigidTransform([0.6, -0.85, 0.15]), Cylinder(0.08, 0.3),
        "landmark_cylinder", [0.89, 0.29, 0.28, 1.0])


def _make_render_params() -> RenderEngineVtkParams:
    from pydrake.geometry import LightParameter

    params = RenderEngineVtkParams()
    params.default_clear_color = [0.62, 0.72, 0.85]
    # Key + fill directional lights read much better than the default headlight.
    params.lights = [
        LightParameter(type="directional", direction=[-0.4, 0.3, -1.0],
                       intensity=1.0, frame="world"),
        LightParameter(type="directional", direction=[0.6, -0.5, -0.4],
                       intensity=0.45, frame="world"),
    ]
    # EGL renders offscreen on NVIDIA without an X display.
    if hasattr(params, "backend") and not os.environ.get("DISPLAY"):
        params.backend = "EGL"
    return params


def _make_controller_plant(arm: str = "iiwa") -> MultibodyPlant:
    plant = MultibodyPlant(time_step=0.0)
    (model,) = Parser(plant).AddModelsFromUrl(ARMS[arm]["url"])
    plant.WeldFrames(plant.world_frame(),
                     plant.GetFrameByName(ARMS[arm]["base_frame"]))
    plant.Finalize()
    return plant


def _ee_frame(plant: MultibodyPlant):
    for name in ("iiwa_link_ee", "iiwa_link_7", "panda_link8"):
        if plant.HasFrameNamed(name):
            return plant.GetFrameByName(name)
    raise RuntimeError("no known end-effector frame in plant")


def ee_pose7(plant: MultibodyPlant, plant_context) -> np.ndarray:
    """End-effector pose as [xyz, euler-xyz, gripper] (V-JEPA2-AC convention)."""
    from scipy.spatial.transform import Rotation

    X_WE = _ee_frame(plant).CalcPoseInWorld(plant_context)
    euler = Rotation.from_matrix(X_WE.rotation().matrix()).as_euler("xyz")
    return np.concatenate([X_WE.translation(), euler, [0.0]])


def run_episode(
    motion,
    params: MotionParams,
    duration: float = 4.0,
    fps: float = 16.0,
    image_size: int = 768,
    meshcat=None,
    return_ee_poses: bool = False,
) -> np.ndarray:
    """Simulate one episode and return rendered frames (T, H, W, 3) uint8.

    With `return_ee_poses=True`, returns (frames, poses) where poses is
    (T, 7): end-effector [xyz, euler-xyz, gripper] at each captured frame.
    """
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=2e-3)
    (iiwa,) = Parser(plant).AddModelsFromUrl(IIWA_URL)
    plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("base"))
    _add_static_scenery(plant)
    plant.Finalize()

    scene_graph.AddRenderer(RENDERER_NAME, MakeRenderEngineVtk(_make_render_params()))

    q_traj = make_joint_trajectory(motion, duration, params)
    v_traj = q_traj.MakeDerivative()

    controller = builder.AddSystem(InverseDynamicsController(
        _make_controller_plant(),
        kp=[400.0] * NUM_JOINTS,
        ki=[1.0] * NUM_JOINTS,
        kd=[40.0] * NUM_JOINTS,
        has_reference_acceleration=False,
    ))
    builder.Connect(plant.get_state_output_port(iiwa),
                    controller.get_input_port_estimated_state())
    builder.Connect(controller.get_output_port_control(),
                    plant.get_actuation_input_port(iiwa))

    src_q = builder.AddSystem(TrajectorySource(q_traj))
    src_v = builder.AddSystem(TrajectorySource(v_traj))
    mux = builder.AddSystem(Multiplexer([NUM_JOINTS, NUM_JOINTS]))
    builder.Connect(src_q.get_output_port(), mux.get_input_port(0))
    builder.Connect(src_v.get_output_port(), mux.get_input_port(1))
    builder.Connect(mux.get_output_port(),
                    controller.get_input_port_desired_state())

    core = RenderCameraCore(
        RENDERER_NAME,
        CameraInfo(width=image_size, height=image_size, fov_y=0.85),
        ClippingRange(0.1, 10.0),
        RigidTransform(),
    )
    camera = builder.AddSystem(RgbdSensor(
        scene_graph.world_frame_id(),
        _look_at(eye=[1.7, -1.35, 1.0], target=[0.0, 0.0, 0.45]),
        ColorRenderCamera(core, False),
        DepthRenderCamera(core, DepthRange(0.1, 10.0)),
    ))
    builder.Connect(scene_graph.get_query_output_port(),
                    camera.query_object_input_port())

    if meshcat is not None:
        from pydrake.visualization import AddDefaultVisualization
        AddDefaultVisualization(builder, meshcat)

    diagram = builder.Build()
    simulator = Simulator(diagram)
    context = simulator.get_mutable_context()

    plant_context = plant.GetMyMutableContextFromRoot(context)
    plant.SetPositions(plant_context, iiwa, q_traj.value(0.0).ravel())

    camera_context = camera.GetMyContextFromRoot(context)
    color_port = camera.color_image_output_port()

    frames, poses = [], []
    n_frames = int(round(duration * fps))
    for i in range(n_frames):
        simulator.AdvanceTo(i / fps)
        image = color_port.Eval(camera_context)
        frames.append(np.array(image.data, copy=True)[:, :, :3])
        if return_ee_poses:
            poses.append(ee_pose7(plant, plant_context))
    if return_ee_poses:
        return np.stack(frames), np.stack(poses)
    return np.stack(frames)


class InteractiveArmSim:
    """Persistent iiwa sim driven by end-effector deltas (for MPC loops).

    Same plant/controller/camera as `run_episode`, but the controller's
    desired state is a fixed input port that `move_ee_delta` retargets via
    differential IK, advancing the dynamics between control steps.
    """

    def __init__(self, q0: np.ndarray | None = None, image_size: int = 768,
                 arm: str = "iiwa"):
        from pydrake.systems.primitives import ConstantVectorSource

        if q0 is None:
            q0 = ARMS[arm]["q_home"]
        builder = DiagramBuilder()
        plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=2e-3)
        (self._iiwa,) = Parser(plant).AddModelsFromUrl(ARMS[arm]["url"])
        plant.WeldFrames(plant.world_frame(),
                         plant.GetFrameByName(ARMS[arm]["base_frame"]))
        _add_static_scenery(plant)
        plant.Finalize()
        scene_graph.AddRenderer(RENDERER_NAME,
                                MakeRenderEngineVtk(_make_render_params()))

        controller = builder.AddSystem(InverseDynamicsController(
            _make_controller_plant(arm),
            kp=[400.0] * NUM_JOINTS, ki=[1.0] * NUM_JOINTS,
            kd=[40.0] * NUM_JOINTS, has_reference_acceleration=False))
        builder.Connect(plant.get_state_output_port(self._iiwa),
                        controller.get_input_port_estimated_state())
        builder.Connect(controller.get_output_port_control(),
                        plant.get_actuation_input_port(self._iiwa))
        desired = builder.AddSystem(
            ConstantVectorSource(np.concatenate([q0, np.zeros(NUM_JOINTS)])))
        builder.Connect(desired.get_output_port(),
                        controller.get_input_port_desired_state())

        core = RenderCameraCore(
            RENDERER_NAME,
            CameraInfo(width=image_size, height=image_size, fov_y=0.85),
            ClippingRange(0.1, 10.0), RigidTransform())
        camera = builder.AddSystem(RgbdSensor(
            scene_graph.world_frame_id(),
            _look_at(eye=[1.7, -1.35, 1.0], target=[0.0, 0.0, 0.45]),
            ColorRenderCamera(core, False),
            DepthRenderCamera(core, DepthRange(0.1, 10.0))))
        builder.Connect(scene_graph.get_query_output_port(),
                        camera.query_object_input_port())

        self._diagram = builder.Build()
        self._simulator = Simulator(self._diagram)
        context = self._simulator.get_mutable_context()
        self._plant, self._camera = plant, camera
        self._plant_context = plant.GetMyMutableContextFromRoot(context)
        self._desired_context = desired.GetMyMutableContextFromRoot(context)
        self._desired = desired
        self._camera_context = camera.GetMyContextFromRoot(context)
        plant.SetPositions(self._plant_context, self._iiwa, q0)
        self._t = 0.0

    def _set_desired(self, q_target: np.ndarray) -> None:
        self._desired.get_mutable_source_value(self._desired_context).set_value(
            np.concatenate([q_target, np.zeros(NUM_JOINTS)]))

    def q(self) -> np.ndarray:
        return self._plant.GetPositions(self._plant_context, self._iiwa)

    def ee_pose(self) -> np.ndarray:
        return ee_pose7(self._plant, self._plant_context)

    def render(self) -> np.ndarray:
        image = self._camera.color_image_output_port().Eval(self._camera_context)
        return np.array(image.data, copy=True)[:, :, :3]

    def advance(self, dt: float) -> None:
        self._t += dt
        self._simulator.AdvanceTo(self._t)

    def move_ee_delta(self, dxyz: np.ndarray, duration: float = 0.4,
                      substeps: int = 4) -> None:
        """Translate the end effector by dxyz (world frame) via differential
        IK on a scratch context, then track the resulting joint target."""
        from pydrake.multibody.tree import JacobianWrtVariable

        scratch = self._plant.CreateDefaultContext()
        q_target = self.q().copy()
        ee = _ee_frame(self._plant)
        for _ in range(substeps):
            self._plant.SetPositions(scratch, self._iiwa, q_target)
            J = self._plant.CalcJacobianTranslationalVelocity(
                scratch, JacobianWrtVariable.kQDot, ee, [0, 0, 0],
                self._plant.world_frame(), self._plant.world_frame())
            dq = np.linalg.pinv(J, rcond=1e-3) @ (np.asarray(dxyz) / substeps)
            q_target = q_target + dq
        lo = self._plant.GetPositionLowerLimits()
        hi = self._plant.GetPositionUpperLimits()
        q_target = np.clip(q_target, lo + 0.02, hi - 0.02)
        self._set_desired(q_target)
        self.advance(duration)
