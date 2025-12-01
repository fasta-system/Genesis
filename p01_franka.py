import numpy as np
import genesis as gs
# gs.init(backend=gs.cpu)
# gs.init(backend=gs.cuda)
gs.init(
    seed                = None,
    precision           = '32',
    debug               = False,
    eps                 = 1e-12,
    logging_level       = None,
    backend             = gs.cuda,
    theme               = 'dark',
    logger_verbose_time = False
)

scene = gs.Scene(
    show_viewer=True,
    sim_options=gs.options.SimOptions(
        dt=1e-2,
        gravity=(0.0, 0.0, -9.81),
    ),
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
        plane_reflection=True,
        ambient_light=(0.1, 0.1, 0.1)
    ),
    renderer=gs.renderers.Rasterizer(),
)

plane = scene.add_entity(gs.morphs.Plane())


franka = scene.add_entity(
    morph=gs.morphs.MJCF(
        file='xml/franka_emika_panda/panda.xml',
        pos = (0, 0, 0),
        euler = (0, 0, 90),
        # quat = (1.0, 0.0, 0.0, 0.0), # w-x-y-z
        scale = 1.0,
        visualization = True,
        collision = True,
    ),
    visualize_contact=False,
)

cam1 = scene.add_camera(
    res=(640, 480),
    pos=(0, 0, 0),
    lookat=(0, 0, 0),
    fov=30,
    GUI=True,
)

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

# R = rot_z(yaw_deg) @ rot_y(pitch_deg) @ rot_x(roll_deg)
R =  rot_y(-90) @ rot_x(0) @ rot_z(-90)
# R =  rot_z(0) @ rot_y(0) @ rot_x(0)

offset_T = np.eye(4, dtype=np.float32)

# Try one of these; if it still looks down, flip the sign.
# offset_T[:3, :3] = R   # or rot_x(90)

# Optional: move the camera a bit forward/up in the head frame
# offset_T[:3, 3] = [0.35, 0.0, 0.2]  # x,y,z offset in meters, e.g. 35cm forward, 20cm up in link frame

cam1.attach(
    # rigid_link=franka,
    # rigid_link=franka.base_link,
    rigid_link=franka.get_link('hand'),
    offset_T=offset_T,
)

cam2 = scene.add_camera(
    res=(640, 480),
    pos=(3.5, 0.0, 2.5),
    lookat=(0, 0, 0.5),
    fov=30,
    GUI=True,
)

cam2.follow_entity(franka)

scene.build(n_envs=0)

print(franka.links)
# print(franka.links[0])
# print(franka.links[0].name)
# print(franka.base_link)
links_name = [link.name for link in franka.links]
links_idx = [link.idx for link in franka.links]
links_name_idx = {link.name: link.idx for link in franka.links}
joints_name = [joint.name for joint in franka.joints]
joints_idx = [joint.idx for joint in franka.joints]
dofs_idx = [joint.idx_local for joint in franka.joints]
# dofs_idx = [franka.get_joint(name).dof_idx_local for name in joints_name]
joints_name_idx = {joint.name: joint.idx for joint in franka.joints}

print(links_name_idx)
print(joints_name)
print(joints_name_idx)

############ Optional: set control gains ############

# set positional gains
franka.set_dofs_kp(
    kp=np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]),
    dofs_idx_local = dofs_idx,
)

# set velocity gains
franka.set_dofs_kv(
    kv=np.array([450, 450, 350, 350, 200, 200, 200, 10, 10]),
    dofs_idx_local=dofs_idx
)

# set force range for safety
franka.set_dofs_force_range(
    lower          = np.array([-87, -87, -87, -87, -12, -12, -12, -100, -100]),
    upper          = np.array([ 87,  87,  87,  87,  12,  12,  12,  100,  100]),
    dofs_idx_local = dofs_idx,
)

for i in range(1000):
    if i < 300:
        franka.set_dofs_position(
            np.zeros((len(dofs_idx),)),
            dofs_idx,
        )

    elif i < 600:
        franka.set_dofs_position(
            np.array([0, 1, 0, 0, 0, 0, 0, 0.04, 0.04])[1:],
            dofs_idx[1:],
        )
        franka.set_dofs_velocity(
            np.array([1.0, 0, 0, 0, 0, 0, 0, 0, 0])[:1],
            dofs_idx[:1],
        )
    elif i < 1000:
        franka.set_dofs_position(
            np.array([0, 0, 0, 0, 0, 0, 0, 0.04, 0.04])[:],
            dofs_idx[:],
        )
    
    scene.step()

    
    if i == 600:
        print("""
                End
                        OF
                            SET
                START
                        OF
                            CONTROL
            """)


for i in range(1500):
    if i == 0:
        franka.control_dofs_position(
            np.array([1, 1, 0, 0, 0, 0, 0, 0.04, 0.04]),
            dofs_idx,
        )
    elif i == 500:
        franka.control_dofs_position(
            np.array([-1, 0.8, 1, -2, 1, 0.5, -0.5, 0.04, 0.04]),
            dofs_idx,
        )
    elif i == 1000:
        franka.control_dofs_position(
            np.array([0, 0, 0, 0, 0, 0, 0, 0, 0]),
            dofs_idx
        )

    scene.step()

    # cam1.render(rgb=True, depth=False, segmentation=False, normal=False)
    # cam2.render(rgb=True, depth=False, segmentation=False, normal=False)