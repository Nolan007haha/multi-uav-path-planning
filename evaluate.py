import numpy as np
import pybullet as p
import heapq
from gym_pybullet_drones.envs import CtrlAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl

NUM = 3
N_TRIALS = 40
MAX_STEPS = 4000
DT = 1.0/240.0
TAKEOFF_STEPS = 200
SPEED = 0.008
SENSOR_RANGE = 1.6
SAFE = 1.0
REACH = 0.3
COLLIDE_DIST = 0.03
STUCK_LIMIT = 500
GRID_RES = 0.2
GRID_MIN = -3.6
GRID_N = 36

def make_scenario(seed, dynamic):
    rng = np.random.default_rng(seed)
    init  = np.array([[-2,-2,0.1],[0,-2,0.1],[2,-2,0.1]], float)
    goals = np.array([[-2, 2,1.0],[0, 2,1.0],[2, 2,1.0]], float)
    types = ['high','low','tall']
    known = []
    unknown = []
    for lane_x in (-2.0, 0.0, 2.0):
        t  = types[rng.integers(0,3)]
        oy = float(rng.uniform(-0.4, 0.4))
        ox = lane_x + float(rng.uniform(-0.2, 0.2))
        known.append((ox, oy, t))
        if dynamic and rng.random() < 0.7:
            ut = types[rng.integers(0,3)]
            uy = float(rng.uniform(1.0, 1.6))
            ux = lane_x + float(rng.uniform(-0.3, 0.3))
            unknown.append((ux, uy, ut))
    return init, goals, known, unknown

def create_obstacles(obstacles, client):
    ids = []
    for ox, oy, typ in obstacles:
        if typ == 'tall':
            he, cz = [0.3,0.3,1.2], 1.2
        elif typ == 'low':
            he, cz = [0.4,0.4,0.65], 0.65
        else:
            he, cz = [0.5,0.5,0.55], 1.45
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=he, physicsClientId=client)
        bid = p.createMultiBody(0, col, -1, [ox, oy, cz], physicsClientId=client)
        ids.append(bid)
    return ids

def cast(pos, direction, own_id, drone_ids, plane_id, client):
    d = direction / (np.linalg.norm(direction) + 1e-9)
    res = p.rayTest(pos.tolist(), (pos + d*SENSOR_RANGE).tolist(), physicsClientId=client)[0]
    hid = res[0]
    if hid < 0 or hid == own_id or hid == plane_id or hid in drone_ids:
        return SENSOR_RANGE
    return res[2] * SENSOR_RANGE

def reactive_steer(pos, goal, own_id, drone_ids, plane_id, client):
    to_goal = goal - pos
    gh = np.array([to_goal[0], to_goal[1], 0.0])
    gh = gh / (np.linalg.norm(gh) + 1e-9)
    up = np.array([0,0,1.0])
    left = np.cross(up, gh); left /= (np.linalg.norm(left) + 1e-9)
    d_fwd = cast(pos, gh, own_id, drone_ids, plane_id, client)
    d_fu  = cast(pos, gh + 0.8*up, own_id, drone_ids, plane_id, client)
    d_fd  = cast(pos, gh - 0.8*up, own_id, drone_ids, plane_id, client)
    d_up  = cast(pos, up, own_id, drone_ids, plane_id, client)
    d_l   = cast(pos, left, own_id, drone_ids, plane_id, client)
    d_r   = cast(pos, -left, own_id, drone_ids, plane_id, client)
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
    return direction / (np.linalg.norm(direction) + 1e-9)

def world_to_cell(x, y):
    return (int((x-GRID_MIN)/GRID_RES), int((y-GRID_MIN)/GRID_RES))

def cell_to_world(cx, cy):
    return (GRID_MIN + (cx+0.5)*GRID_RES, GRID_MIN + (cy+0.5)*GRID_RES)

def build_grid(obstacles, inflate=0.3):
    grid = np.zeros((GRID_N, GRID_N), dtype=bool)
    for ox, oy, typ in obstacles:
        half = 0.3 if typ=='tall' else (0.4 if typ=='low' else 0.5)
        r = half + inflate
        for cx in range(GRID_N):
            for cy in range(GRID_N):
                wx, wy = cell_to_world(cx, cy)
                if abs(wx-ox) <= r and abs(wy-oy) <= r:
                    grid[cx, cy] = True
    return grid

def astar(grid, start, goal):
    s = world_to_cell(start[0], start[1])
    g = world_to_cell(goal[0], goal[1])
    def h(a,b): return ((a[0]-b[0])**2+(a[1]-b[1])**2)**0.5
    openset = [(0.0, s)]
    came = {}
    gsc = {s:0.0}
    nbrs = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
    while openset:
        _, cur = heapq.heappop(openset)
        if cur == g:
            path = [cur]
            while cur in came:
                cur = came[cur]; path.append(cur)
            path.reverse()
            return [cell_to_world(cx,cy) for cx,cy in path]
        for dx,dy in nbrs:
            nx,ny = cur[0]+dx, cur[1]+dy
            if nx<0 or ny<0 or nx>=GRID_N or ny>=GRID_N: continue
            if grid[nx,ny]: continue
            ng = gsc[cur] + (dx*dx+dy*dy)**0.5
            if (nx,ny) not in gsc or ng < gsc[(nx,ny)]:
                gsc[(nx,ny)] = ng
                came[(nx,ny)] = cur
                heapq.heappush(openset, (ng + h((nx,ny), g), (nx,ny)))
    return None

def plan_waypoints(known, init, goals):
    grid = build_grid(known)
    wps = []
    for i in range(NUM):
        path = astar(grid, init[i][:2], goals[i][:2])
        if path is None:
            path = [(goals[i][0], goals[i][1])]
        pts = path[::3]
        pts.append((goals[i][0], goals[i][1]))
        wps.append(pts)
    return wps

def run_episode(method, seed, dynamic):
    init, goals, known, unknown = make_scenario(seed, dynamic)
    env = CtrlAviary(num_drones=NUM, initial_xyzs=init, gui=False)
    client = env.CLIENT
    ctrl = [DSLPIDControl(drone_model=env.DRONE_MODEL) for _ in range(NUM)]
    obs, _ = env.reset()
    obstacle_ids = create_obstacles(known + unknown, client)
    drone_ids = list(env.DRONE_IDS)
    plane_id = env.PLANE_ID

    use_astar = method in ('astar_only', 'astar')
    if use_astar:
        waypoints = plan_waypoints(known, init, goals)
        wp_idx = [0]*NUM
        stuck = [0]*NUM

    moving_target = init.copy().astype(float)
    moving_target[:,2] = 1.0
    phase = ['takeoff']*NUM
    reached_step = [None]*NUM
    collided = [False]*NUM
    prev = [init[i].copy() for i in range(NUM)]
    action = np.zeros((NUM,4))

    for step in range(MAX_STEPS):
        for i in range(NUM):
            pos = np.array(p.getBasePositionAndOrientation(drone_ids[i], physicsClientId=client)[0])
            if reached_step[i] is not None:
                moving_target[i] = goals[i]
            elif phase[i] == 'takeoff':
                moving_target[i] = np.array([init[i][0], init[i][1], 1.0])
                if step >= TAKEOFF_STEPS:
                    phase[i] = 'cruise'
            else:
                if method == 'straight':
                    d = goals[i] - moving_target[i]; n = np.linalg.norm(d)
                    moving_target[i] += d/n*SPEED if n > SPEED else (goals[i]-moving_target[i])
                elif method == 'reactive':
                    dirc = reactive_steer(pos, goals[i], drone_ids[i], drone_ids, plane_id, client)
                    moving_target[i] += dirc * SPEED
                    moving_target[i][2] = np.clip(moving_target[i][2], 0.35, 2.1)
                elif method == 'astar_only':
                    wx, wy = waypoints[i][wp_idx[i]]
                    wp = np.array([wx, wy, 1.0])
                    to_wp = wp - moving_target[i]; n = np.linalg.norm(to_wp)
                    if n > SPEED:
                        moving_target[i] += to_wp/n*SPEED
                    else:
                        moving_target[i] = wp.copy()
                        if wp_idx[i] < len(waypoints[i])-1:
                            wp_idx[i] += 1
                else:
                    wx, wy = waypoints[i][wp_idx[i]]
                    subgoal = np.array([wx, wy, 1.0])
                    dirc = reactive_steer(pos, subgoal, drone_ids[i], drone_ids, plane_id, client)
                    moving_target[i] += dirc * SPEED
                    moving_target[i][2] = np.clip(moving_target[i][2], 0.35, 2.1)
                    stuck[i] += 1
                    if wp_idx[i] < len(waypoints[i])-1 and (np.linalg.norm(pos[:2]-np.array([wx,wy])) < 0.4 or stuck[i] > STUCK_LIMIT):
                        wp_idx[i] += 1
                        stuck[i] = 0
                if np.linalg.norm(pos[:2] - goals[i][:2]) < REACH:
                    reached_step[i] = step

            action[i], _, _ = ctrl[i].computeControlFromState(
                control_timestep=env.CTRL_TIMESTEP, state=obs[i],
                target_pos=moving_target[i], target_rpy=np.zeros(3))

        obs, _, _, _, _ = env.step(action)

        if step % 5 == 0:
            for i in range(NUM):
                for oid in obstacle_ids:
                    cps = p.getClosestPoints(drone_ids[i], oid, 0.1, physicsClientId=client)
                    if any(c[8] < COLLIDE_DIST for c in cps):
                        collided[i] = True
                for j in range(i+1, NUM):
                    cps = p.getClosestPoints(drone_ids[i], drone_ids[j], 0.1, physicsClientId=client)
                    if any(c[8] < COLLIDE_DIST for c in cps):
                        collided[i] = True; collided[j] = True

        if all(r is not None for r in reached_step):
            break

    env.close()
    return [{'collided': collided[i], 'success': (reached_step[i] is not None) and not collided[i]} for i in range(NUM)]

def main():
    methods = ['straight', 'reactive', 'astar_only', 'astar']
    label = {'straight':'Direct','reactive':'Reactive','astar_only':'A* only','astar':'A*+Reactive'}
    modes = [('Static', False), ('Dynamic', True)]
    res = {}
    for mode_name, dyn in modes:
        for m in methods:
            recs = []
            for trial in range(N_TRIALS):
                recs.extend(run_episode(m, trial, dyn))
                print("  [" + mode_name + "] " + label[m] + ": " + str(trial+1) + "/" + str(N_TRIALS))
            n = len(recs)
            res[(mode_name, m)] = {'succ': 100.0*sum(r['success'] for r in recs)/n, 'coll': sum(r['collided'] for r in recs)}
    print("")
    print("="*64)
    print("TRADE-OFF: Static (known map) vs Dynamic (unknown obstacles)")
    print(str(N_TRIALS) + " trials x " + str(NUM) + " drones per cell")
    print("="*64)
    print("{:<14}{:>11}{:>11}{:>11}{:>11}".format("Method","Static%","StaticCol","Dynamic%","DynCol"))
    print("-"*64)
    for m in methods:
        s = res[('Static', m)]; d = res[('Dynamic', m)]
        print("{:<14}{:>10.1f}%{:>11}{:>10.1f}%{:>11}".format(label[m], s['succ'], s['coll'], d['succ'], d['coll']))
    print("="*64)
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        labels = [label[m] for m in methods]
        x = np.arange(len(methods)); w = 0.38
        plt.figure(figsize=(10,5))
        plt.bar(x-w/2, [res[('Static',m)]['succ'] for m in methods], w, label='Static map', color='#5cb85c')
        plt.bar(x+w/2, [res[('Dynamic',m)]['succ'] for m in methods], w, label='Dynamic map', color='#f0ad4e')
        plt.xticks(x, labels); plt.ylabel('Success rate (%)'); plt.ylim(0,100)
        plt.title('Global vs local trade-off: success rate by environment')
        plt.legend(); plt.grid(axis='y', alpha=0.3); plt.tight_layout()
        plt.savefig('/Users/nah/Desktop/eval_results.png', dpi=130)
        print("Da luu bieu do: ~/Desktop/eval_results.png")
    except Exception as e:
        print("Khong ve duoc bieu do: " + str(e))

if __name__ == "__main__":
    main()
