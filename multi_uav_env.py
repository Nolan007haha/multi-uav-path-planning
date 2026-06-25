import numpy as np
import pybullet as p
import pybullet_data
from gym_pybullet_drones.envs import CtrlAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
import time

NUM_DRONES = 3

INIT_XYZS = np.array([
    [-2.0, -2.0, 0.1],
    [ 0.0, -2.0, 0.1],
    [ 2.0, -2.0, 0.1],
])
GOALS = np.array([
    [-2.0, 2.0, 1.0],
    [ 0.0, 2.0, 1.0],
    [ 2.0, 2.0, 1.0],
])
HOME = np.array([[INIT_XYZS[i][0], INIT_XYZS[i][1], 1.0] for i in range(NUM_DRONES)])
OBSTACLES = [
    (-2.0,  0.0, 'high'),
    ( 0.0, -0.7, 'low'),
    ( 0.0,  0.7, 'tall'),
    ( 2.0,  0.0, 'tall'),
]

env = CtrlAviary(num_drones=NUM_DRONES, initial_xyzs=INIT_XYZS, gui=True)
client = env.CLIENT
ctrl = [DSLPIDControl(drone_model=env.DRONE_MODEL) for _ in range(NUM_DRONES)]

obs, _ = env.reset()

p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0, physicsClientId=client)
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=client)
p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0, physicsClientId=client)
p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0, physicsClientId=client)
p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0, physicsClientId=client)

p.resetDebugVisualizerCamera(
    cameraDistance=9.0, cameraYaw=50, cameraPitch=-45,
    cameraTargetPosition=[0.0, 0.0, 0.8], physicsClientId=client)

p.setAdditionalSearchPath(pybullet_data.getDataPath())

for ox, oy, typ in OBSTACLES:
    if typ == 'tall':
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.3,0.3,1.2], physicsClientId=client)
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.3,0.3,1.2], rgbaColor=[1.0,0.1,0.1,1], physicsClientId=client)
        p.createMultiBody(0, col, vis, [ox, oy, 1.2], physicsClientId=client)
    elif typ == 'low':
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.4,0.4,0.65], physicsClientId=client)
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.4,0.4,0.65], rgbaColor=[1.0,0.55,0.0,1], physicsClientId=client)
        p.createMultiBody(0, col, vis, [ox, oy, 0.65], physicsClientId=client)
    else:
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.5,0.5,0.55], physicsClientId=client)
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.5,0.5,0.55], rgbaColor=[0.5,0.2,0.8,1], physicsClientId=client)
        p.createMultiBody(0, col, vis, [ox, oy, 1.45], physicsClientId=client)

colors = [[1,0,0],[0,0.3,1],[0,0.8,0]]

# Warehouse = raised bright-yellow pad at each start
for i in range(NUM_DRONES):
    vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.42,0.42,0.1], rgbaColor=[1.0,0.85,0.0,1], physicsClientId=client)
    p.createMultiBody(0, -1, vis, [float(INIT_XYZS[i][0]), float(INIT_XYZS[i][1]), 0.05], physicsClientId=client)
    p.addUserDebugText('WAREHOUSE '+str(i+1), [float(INIT_XYZS[i][0]), float(INIT_XYZS[i][1]), 0.6], [0.85,0.65,0.0], 1.4, physicsClientId=client)

# Delivery / customer points
for i, (g, c) in enumerate(zip(GOALS, colors)):
    vis = p.createVisualShape(p.GEOM_CYLINDER, radius=0.3, length=0.05, rgbaColor=c+[0.8], physicsClientId=client)
    p.createMultiBody(0, -1, vis, [float(g[0]), float(g[1]), 0.01], physicsClientId=client)
    p.addUserDebugText('CUSTOMER '+str(i+1), [float(g[0]), float(g[1]), float(g[2])+0.4], c, 1.2, physicsClientId=client)

# Package = big bright-orange cube, one per drone
package_ids = []
for i in range(NUM_DRONES):
    vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.14,0.14,0.14], rgbaColor=[1.0,0.45,0.0,1], physicsClientId=client)
    pid = p.createMultiBody(0, -1, vis, [float(INIT_XYZS[i][0]), float(INIT_XYZS[i][1]), 0.25], physicsClientId=client)
    package_ids.append(pid)
carrying = [True]*NUM_DRONES

print("=== PICKUP-DELIVERY VERSION ===")
print("Created", len(package_ids), "orange packages and", NUM_DRONES, "yellow warehouses")

SENSOR_RANGE = 1.6
N_RING = 16
ring_angles = np.linspace(0, 2*np.pi, N_RING, endpoint=False)
ring_dirs = np.array([[np.cos(a), np.sin(a), 0.0] for a in ring_angles])
ray_line_ids = [[p.addUserDebugLine([0,0,0],[0,0,0.001],[0,1,0], physicsClientId=client)
                 for _ in range(N_RING)] for _ in range(NUM_DRONES)]

PLANE_ID = env.PLANE_ID
DRONE_IDS = list(env.DRONE_IDS)
EXCLUDE = set(DRONE_IDS) | {PLANE_ID} | set(package_ids)
SAFE = 1.0
GOAL_LOCK = 0.6

TAKEOFF_STEPS = 200
CRUISE_SPEED = 0.006
DT = 1/240.0
SENSE_EVERY = 4

action = np.zeros((NUM_DRONES, 4))
prev_pos = [None]*NUM_DRONES
moving_target = np.array([[INIT_XYZS[i][0], INIT_XYZS[i][1], 1.0] for i in range(NUM_DRONES)], dtype=float)
phase = ['takeoff'] * NUM_DRONES
cached_dir = [np.zeros(3) for _ in range(NUM_DRONES)]

def cast(pos, direction, own_id):
    d = direction / (np.linalg.norm(direction)+1e-9)
    res = p.rayTest(pos.tolist(), (pos + d*SENSOR_RANGE).tolist(), physicsClientId=client)[0]
    hid = res[0]
    if hid < 0 or hid == own_id or hid in EXCLUDE:
        return SENSOR_RANGE
    return res[2] * SENSOR_RANGE

def sense_steer(i, pos, goal, own_id):
    to_goal = goal - pos
    gh = np.array([to_goal[0], to_goal[1], 0.0])
    gh = gh / (np.linalg.norm(gh) + 1e-9)
    up = np.array([0,0,1.0])
    left = np.cross(up, gh); left /= (np.linalg.norm(left)+1e-9)

    d_fwd = cast(pos, gh, own_id)
    d_fu  = cast(pos, gh + 0.8*up, own_id)
    d_fd  = cast(pos, gh - 0.8*up, own_id)
    d_up  = cast(pos, up, own_id)
    d_l   = cast(pos, left, own_id)
    d_r   = cast(pos, -left, own_id)

    if d_fwd > SAFE:
        direction = gh.copy()
        dz = goal[2] - pos[2]
        if dz > 0 and d_up < SAFE:
            dz = 0
        direction[2] = np.clip(dz, -1, 1) * 0.5
    elif d_fu > SAFE:
        direction = np.array([gh[0]*0.25, gh[1]*0.25, 1.0])
    elif d_fd > SAFE:
        direction = np.array([gh[0]*0.25, gh[1]*0.25, -1.0])
    else:
        side = left if d_l >= d_r else -left
        direction = side.copy()
    direction = direction / (np.linalg.norm(direction)+1e-9)

    ring_hit = [False]*N_RING
    froms = np.tile(pos, (N_RING,1))
    tos = pos + ring_dirs*SENSOR_RANGE
    results = p.rayTestBatch(froms.tolist(), tos.tolist(), physicsClientId=client)
    for k, res in enumerate(results):
        hid = res[0]
        if hid<0 or hid==own_id or hid in EXCLUDE: continue
        if res[2]*SENSOR_RANGE < SAFE: ring_hit[k]=True
    return direction, ring_hit

def steer_to(i, pos, goal, own_id, step):
    if step % SENSE_EVERY == 0:
        dirc, ring_hit = sense_steer(i, pos, goal, own_id)
        cached_dir[i] = dirc
        for k in range(N_RING):
            end = pos + ring_dirs[k]*SENSOR_RANGE
            col = [1,0,0] if ring_hit[k] else [0,1,0.3]
            p.addUserDebugLine(pos.tolist(), end.tolist(), col, 2,
                               replaceItemUniqueId=ray_line_ids[i][k], physicsClientId=client)
    moving_target[i] += cached_dir[i] * CRUISE_SPEED
    moving_target[i][2] = np.clip(moving_target[i][2], 0.35, 2.1)

print("Mission: warehouse -> pickup -> deliver -> return home")

try:
    for step in range(999999):
        for i in range(NUM_DRONES):
            pos, _ = p.getBasePositionAndOrientation(env.DRONE_IDS[i], physicsClientId=client)
            pos = np.array(pos)

            if phase[i] == 'takeoff':
                moving_target[i] = np.array([INIT_XYZS[i][0], INIT_XYZS[i][1], 1.0])
                if step >= TAKEOFF_STEPS:
                    phase[i] = 'deliver'
                    print("Drone "+str(i+1)+": package loaded, delivering...")

            elif phase[i] == 'deliver':
                horiz = np.linalg.norm(pos[:2] - GOALS[i][:2])
                if horiz < GOAL_LOCK:
                    d = GOALS[i] - moving_target[i]; nd = np.linalg.norm(d)
                    if nd > CRUISE_SPEED:
                        moving_target[i] += d/nd * CRUISE_SPEED
                    else:
                        moving_target[i] = GOALS[i].copy()
                        phase[i] = 'drop'
                else:
                    steer_to(i, pos, GOALS[i], env.DRONE_IDS[i], step)

            elif phase[i] == 'drop':
                moving_target[i][2] -= 0.004
                if moving_target[i][2] <= 0.5:
                    p.resetBasePositionAndOrientation(package_ids[i], [float(GOALS[i][0]), float(GOALS[i][1]), 0.14], [0,0,0,1])
                    carrying[i] = False
                    phase[i] = 'return'
                    print("Drone "+str(i+1)+": delivered! returning home...")

            elif phase[i] == 'return':
                horiz = np.linalg.norm(pos[:2] - HOME[i][:2])
                if horiz < GOAL_LOCK:
                    d = HOME[i] - moving_target[i]; nd = np.linalg.norm(d)
                    if nd > CRUISE_SPEED:
                        moving_target[i] += d/nd * CRUISE_SPEED
                    else:
                        moving_target[i] = HOME[i].copy()
                        phase[i] = 'land'
                else:
                    steer_to(i, pos, HOME[i], env.DRONE_IDS[i], step)

            elif phase[i] == 'land':
                moving_target[i][2] -= 0.004
                if moving_target[i][2] <= 0.12:
                    moving_target[i][2] = 0.12
                    phase[i] = 'done'
                    print("Drone "+str(i+1)+": back at warehouse!")
            else:
                moving_target[i] = np.array([INIT_XYZS[i][0], INIT_XYZS[i][1], 0.12])

            if carrying[i]:
                p.resetBasePositionAndOrientation(package_ids[i],
                    [float(pos[0]), float(pos[1]), float(max(pos[2]-0.16, 0.14))], [0,0,0,1])

            action[i], _, _ = ctrl[i].computeControlFromState(
                control_timestep=env.CTRL_TIMESTEP,
                state=obs[i],
                target_pos=moving_target[i],
                target_rpy=np.zeros(3))

        obs, _, _, _, _ = env.step(action)

        if step % 6 == 0:
            for i in range(NUM_DRONES):
                pos, _ = p.getBasePositionAndOrientation(env.DRONE_IDS[i], physicsClientId=client)
                pos = list(pos)
                if prev_pos[i] is not None:
                    p.addUserDebugLine(prev_pos[i], pos, colors[i], 2, lifeTime=20.0, physicsClientId=client)
                prev_pos[i] = pos

        time.sleep(DT)

except (p.error, KeyboardInterrupt):
    print("\nSimulation stopped.")
    env.close()