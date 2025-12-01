import numpy as np
import genesis as gs

def rot_x(deg):
    rad = np.deg2rad(deg)
    c, s = np.cos(rad), np.sin(rad)
    return np.array([
        [1, 0,  0],
        [0, c, -s],
        [0, s,  c],
    ], dtype=np.float32)

def rot_y(deg):
    rad = np.deg2rad(deg)
    c, s = np.cos(rad), np.sin(rad)
    return np.array([
        [ c, 0,  s],
        [ 0, 1,  0],
        [-s, 0,  c],
    ], dtype=np.float32)

def rot_z(deg):
    rad = np.deg2rad(deg)
    c, s = np.cos(rad), np.sin(rad)
    return np.array([
        [ c, -s, 0],
        [ s,  c, 0],
        [ 0,  0, 1],
    ], dtype=np.float32)

gs.init(
    seed=42,
    precision="64",
    eps=1e-15,
    logging_level=None,
    backend=gs.cuda,
)

scene = gs.Scene(
    show_viewer=True,
    viewer_options=gs.options.ViewerOptions(
        res=(1280, 720),
        camera_pos=(5.5, 2.0, 4.5),
        camera_lookat=(0.0, 0.0, 0.5),
        camera_fov=40,
        max_FPS=60,
    ),
    vis_options=gs.options.VisOptions(
        show_world_frame=True,
        world_frame_size=0.2,
        show_link_frame=False,
        link_frame_size=0.2,
        show_cameras=False,
        plane_reflection=False,
        ambient_light=(0.1, 0.1, 0.1)
    ),
    sim_options=gs.options.SimOptions(
         dt=1e-2,
        gravity=(0.0, 0.0, -9.81),
    ),
    renderer=gs.renderers.Rasterizer(),
    # renderer=gs.renderers.RayTracer(),
)

plane = scene.add_entity(
    morph=gs.morphs.Plane()
)

franka = scene.add_entity(
    morph=gs.morphs.MJCF(
        file="xml/franka_emika_panda/panda.xml",
        pos = (0, 0, 0),
        euler = (0, 0, 90),
        # quat = (1.0, 0.0, 0.0, 0.0), # w-x-y-z
        scale = 1.0,
        visualization = True,
        collision = True,
    ),
    visualize_contact=False,
)

dofs_idx = [joint.idx_local for joint in franka.joints]

cube = scene.add_entity(
    morph=gs.morphs.Box(
        pos=(0.65, 0, 0),
        euler=(0, 0, 0),
        size=(0.1, 0.1, 0.1),
        visualization = True,
        collision = True,
    )
)

cam = scene.add_camera(
    res=(640, 640),
    fov=40,
    up=(0,0,1),
    GUI=True,
)

R =  rot_y(-90) @ rot_x(0) @ rot_z(-90)
offset_T = np.eye(4, dtype=np.float32)
# offset_T[:3, :3] = R 
# offset_T[:3, 3] = [0.35, 0.0, 0.2]

cam.attach(
    rigid_link=franka.get_link('hand'),
    offset_T=offset_T
)

scene.build(n_envs=0)

############ Optional: set control gains ############
# set positional gains
franka.set_dofs_kp(
    # kp             = np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]),
    kp             = np.array([1500, 1500, 1000, 1000, 500, 500, 500, 10, 10]),
    dofs_idx_local = dofs_idx,
)
# set velocity gains
franka.set_dofs_kv(
    kv             = np.array([450, 450, 350, 350, 200, 200, 200, 10, 10]),
    dofs_idx_local = dofs_idx,
)
# set force range for safety
franka.set_dofs_force_range(
    # lower          = np.array([-87, -87, -87, -87, -12, -12, -12, -100, -100]),
    # upper          = np.array([ 87,  87,  87,  87,  12,  12,  12,  100,  100]),
    lower          = np.array([-77, -77, -77, -77, -12, -12, -12, -100, -100]),
    upper          = np.array([ 77,  77,  77,  77,  12,  12,  12,  100,  100]),
    dofs_idx_local = dofs_idx,
)


# Retrieve some commonly used handles
rigid_solver = scene.sim.rigid_solver # low-level rigid body solver
end_effector = franka.get_link("hand") # Franka gripper frame
cube_link = cube.get_link("box_baselink") # the link we want to pick

################ Reach straight pose ################
franka.control_dofs_position(
    np.array([0, 0, 0, 0, 0, 0, 0, 0.04, 0.04]),
    dofs_idx,
)

for i in range(200):
    scene.step()
    # cam.render()


################ Reach pre-grasp pose ################
q_pregrasp = franka.inverse_kinematics(
    link=end_effector,
    pos=np.array([0.65, 0.0, 0.5]), # just above the cube
    quat=np.array([0, 1, 0, 0]), # down-facing orientation
)
print(f"q_pregrasp: {q_pregrasp}")

franka.control_dofs_position(
    q_pregrasp[:-2], # arm joints only
    dofs_idx[:-2],
    # np.arange(7),
)

for i in range(200):
    scene.step()
    # cam.render()

q_pregrasp = franka.inverse_kinematics(
    link=end_effector,
    pos=np.array([0.65, 0.0, 0.22]), # just above the cube
    quat=np.array([0, 1, 0, 0]), # down-facing orientation
)
print(f"q_pregrasp: {q_pregrasp}")

franka.control_dofs_position(
    q_pregrasp[:-2],
    dofs_idx[:-2],
)

for i in range(200):
    scene.step()
    # cam.render()

################ Attach (activate suction) ################
weld_links_target = np.array([cube_link.idx], dtype=gs.np_int)
weld_links_handle = np.array([end_effector.idx], dtype=gs.np_int)
rigid_solver.add_weld_constraint(weld_links_target, weld_links_handle)


################ Lift and transport ################
q_lift = franka.inverse_kinematics(
    link=end_effector,
    pos=np.array([0.65, 0.0, 0.7]), # just above the cube
    quat=np.array([0, 1, 0, 0]), # down-facing orientation
)

franka.control_dofs_position(
    q_lift[:-2],
    dofs_idx[:-2]
)

for i in range(200):
    scene.step()
    # cam.render()

q_place = franka.inverse_kinematics(
    link=end_effector,
    pos=np.array([0.65, 0.65, 0.2]), # target place pose
    quat = np.array([0, 1, 0, 0]),
)

franka.control_dofs_position(
    q_place[:-2],
    dofs_idx[:-2],
)

for i in range(200):
    scene.step()
    # cam.render()


################ Detach (release suction) ################
rigid_solver.delete_weld_constraint(weld_links_target, weld_links_handle)

for i in range(400):
    scene.step()
    # cam.render()
