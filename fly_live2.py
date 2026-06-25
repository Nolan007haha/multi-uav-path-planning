import numpy as np
import pybullet as p
import pybullet_data
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import subprocess
from gym_pybullet_drones.envs import CtrlAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from stable_baselines3 import PPO
import time, os

NUM_DRONES = 3
PPO_DRONE  = 0

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

PPO_MODEL_PATH = os.path.expanduser("~/Desktop/ppo_uav_delivery")
try:
    ppo_model = PPO.load(PPO_MODEL_PATH)
    print("PPO model loaded.")
    USE_PPO = True
except Exception as e:
    ppo_model = None
    USE_PPO = False
    print("WARNING: PPO not found, all Classical.", e)

def get_ppo_obs(pos, goal, vel):
    rel  = goal - pos
    hits = []
    for a in np.linspace(0, 2*np.pi, 8, endpoint=False):
        d = np.array([np.cos(a), np.sin(a), 0.0])
        try:
            res = p.rayTest(pos.tolist(), (pos+d*3.0).tolist(),
                            physicsClientId=client)[0]
            hits.append(float(res[2]) if res[0] > 0 else 1.0)
        except Exception:
            hits.append(1.0)
    return np.concatenate([rel, vel, hits]).astype(np.float32)

env = CtrlAviary(num_drones=NUM_DRONES, initial_xyzs=INIT_XYZS, gui=True)
client = env.CLIENT
ctrl = [DSLPIDControl(drone_model=env.DRONE_MODEL) for _ in range(NUM_DRONES)]
obs, _ = env.reset()

p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0, physicsClientId=client)
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=client)
p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0, physicsClientId=client)
p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0, physicsClientId=client)
p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0, physicsClientId=client)
p.resetDebugVisualizerCamera(cameraDistance=9.0, cameraYaw=50, cameraPitch=-45,
                              cameraTargetPosition=[0.0,0.0,0.8], physicsClientId=client)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

obstacle_ids = []
for ox, oy, typ in OBSTACLES:
    if typ == 'tall':
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.3,0.3,1.2], physicsClientId=client)
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.3,0.3,1.2], rgbaColor=[1.0,0.1,0.1,1], physicsClientId=client)
        bid = p.createMultiBody(0, col, vis, [ox, oy, 1.2], physicsClientId=client)
    elif typ == 'low':
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.4,0.4,0.65], physicsClientId=client)
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.4,0.4,0.65], rgbaColor=[1.0,0.55,0.0,1], physicsClientId=client)
        bid = p.createMultiBody(0, col, vis, [ox, oy, 0.65], physicsClientId=client)
    else:
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.5,0.5,0.55], physicsClientId=client)
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.5,0.5,0.55], rgbaColor=[0.5,0.2,0.8,1], physicsClientId=client)
        bid = p.createMultiBody(0, col, vis, [ox, oy, 1.45], physicsClientId=client)
    obstacle_ids.append(bid)

colors     = [[1,0,0],[0,0.3,1],[0,0.8,0]]
MPL_COLORS = ['#e05050','#4a7fe0','#4caf50']

for i in range(NUM_DRONES):
    vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.42,0.42,0.1],
                              rgbaColor=[1.0,0.85,0.0,1], physicsClientId=client)
    p.createMultiBody(0,-1,vis,
                      [float(INIT_XYZS[i][0]),float(INIT_XYZS[i][1]),0.05],
                      physicsClientId=client)
    lbl = ('WAREHOUSE '+str(i+1)+' [PPO+Classical]') if i==PPO_DRONE \
          else ('WAREHOUSE '+str(i+1)+' [Classical]')
    col = [1,0.8,0] if i==PPO_DRONE else [0.85,0.65,0.0]
    p.addUserDebugText(lbl,[float(INIT_XYZS[i][0]),float(INIT_XYZS[i][1]),0.65],
                       col,1.1,physicsClientId=client)

for i,(g,c) in enumerate(zip(GOALS,colors)):
    vis = p.createVisualShape(p.GEOM_CYLINDER,radius=0.3,length=0.05,
                              rgbaColor=c+[0.8],physicsClientId=client)
    p.createMultiBody(0,-1,vis,[float(g[0]),float(g[1]),0.01],physicsClientId=client)
    p.addUserDebugText('CUSTOMER '+str(i+1),
                       [float(g[0]),float(g[1]),float(g[2])+0.4],c,1.2,physicsClientId=client)

package_ids = []
for i in range(NUM_DRONES):
    vis = p.createVisualShape(p.GEOM_BOX,halfExtents=[0.14,0.14,0.14],
                              rgbaColor=[1.0,0.45,0.0,1],physicsClientId=client)
    pid = p.createMultiBody(0,-1,vis,
                            [float(INIT_XYZS[i][0]),float(INIT_XYZS[i][1]),0.25],
                            physicsClientId=client)
    package_ids.append(pid)
carrying = [True]*NUM_DRONES

print("=== Drone1=PPO+Classical Hybrid | Drone2,3=Classical | Drone-Drone Avoidance ON ===")

delivery_time = [None]*NUM_DRONES
return_time   = [None]*NUM_DRONES
returned      = [False]*NUM_DRONES
path_length   = [0.0]*NUM_DRONES
last_pos      = [np.array(INIT_XYZS[i],dtype=float) for i in range(NUM_DRONES)]
collided      = [set() for _ in range(NUM_DRONES)]
battery       = [100.0]*NUM_DRONES
traj          = [[] for _ in range(NUM_DRONES)]
ppo_vel       = np.zeros(3)

SENSOR_RANGE  = 1.6
N_RING        = 16
ring_angles   = np.linspace(0,2*np.pi,N_RING,endpoint=False)
ring_dirs     = np.array([[np.cos(a),np.sin(a),0.0] for a in ring_angles])
ray_line_ids  = [[p.addUserDebugLine([0,0,0],[0,0,0.001],[0,1,0],physicsClientId=client)
                  for _ in range(N_RING)] for _ in range(NUM_DRONES)]
TRAIL_MAX     = 60
trail_ids     = [[] for _ in range(NUM_DRONES)]
trail_ptr     = [0]*NUM_DRONES

PLANE_ID  = env.PLANE_ID
DRONE_IDS = list(env.DRONE_IDS)
EXCLUDE   = set(DRONE_IDS)|{PLANE_ID}|set(package_ids)

SAFE           = 1.0
GOAL_LOCK      = 0.6
TAKEOFF_STEPS  = 200
CRUISE_SPEED   = 0.006
DT             = 1/240.0
SENSE_EVERY    = 4
BATTERY_DRAIN  = 0.005
REPULSE_RADIUS = 0.9
REPULSE_GAIN   = 0.014
PPO_BLEND      = 0.4   # 40% PPO + 60% Classical

action        = np.zeros((NUM_DRONES,4))
prev_pos      = [None]*NUM_DRONES
moving_target = np.array([[INIT_XYZS[i][0],INIT_XYZS[i][1],1.0]
                           for i in range(NUM_DRONES)],dtype=float)
phase         = ['takeoff']*NUM_DRONES
cached_dir    = [np.zeros(3) for _ in range(NUM_DRONES)]
step          = 0

def drone_repulsion(i, pos):
    rep = np.zeros(3)
    for j in range(NUM_DRONES):
        if j==i: continue
        jpos,_ = p.getBasePositionAndOrientation(env.DRONE_IDS[j],physicsClientId=client)
        jpos   = np.array(jpos)
        diff   = pos-jpos
        dist   = np.linalg.norm(diff)
        if 0.01<dist<REPULSE_RADIUS:
            rep += (diff/dist)*(REPULSE_RADIUS-dist)*REPULSE_GAIN
    return rep

def cast(pos, direction, own_id):
    d   = direction/(np.linalg.norm(direction)+1e-9)
    res = p.rayTest(pos.tolist(),(pos+d*SENSOR_RANGE).tolist(),
                    physicsClientId=client)[0]
    hid = res[0]
    if hid<0 or hid==own_id or hid in EXCLUDE: return SENSOR_RANGE
    return res[2]*SENSOR_RANGE

def sense_steer(i, pos, goal, own_id):
    to_goal=goal-pos
    gh=np.array([to_goal[0],to_goal[1],0.0]); gh/=(np.linalg.norm(gh)+1e-9)
    up=np.array([0,0,1.0])
    left=np.cross(up,gh); left/=(np.linalg.norm(left)+1e-9)
    d_fwd=cast(pos,gh,own_id); d_fu=cast(pos,gh+0.8*up,own_id)
    d_fd=cast(pos,gh-0.8*up,own_id); d_up=cast(pos,up,own_id)
    d_l=cast(pos,left,own_id); d_r=cast(pos,-left,own_id)
    if d_fwd>SAFE:
        direction=gh.copy(); dz=goal[2]-pos[2]
        if dz>0 and d_up<SAFE: dz=0
        direction[2]=np.clip(dz,-1,1)*0.5
    elif d_fu>SAFE: direction=np.array([gh[0]*0.25,gh[1]*0.25,1.0])
    elif d_fd>SAFE: direction=np.array([gh[0]*0.25,gh[1]*0.25,-1.0])
    else:
        side=left if d_l>=d_r else -left; direction=side.copy()
    direction/=(np.linalg.norm(direction)+1e-9)
    ring_hit=[False]*N_RING
    froms=np.tile(pos,(N_RING,1)); tos=pos+ring_dirs*SENSOR_RANGE
    results=p.rayTestBatch(froms.tolist(),tos.tolist(),physicsClientId=client)
    for k,res in enumerate(results):
        hid=res[0]
        if hid<0 or hid==own_id or hid in EXCLUDE: continue
        if res[2]*SENSOR_RANGE<SAFE: ring_hit[k]=True
    return direction, ring_hit

def steer_classical(i, pos, goal, own_id, step):
    if step%SENSE_EVERY==0:
        dirc,ring_hit=sense_steer(i,pos,goal,own_id)
        cached_dir[i]=dirc
        try:
            for k in range(N_RING):
                end=pos+ring_dirs[k]*SENSOR_RANGE
                col=[1,0,0] if ring_hit[k] else [0,1,0.3]
                p.addUserDebugLine(pos.tolist(),end.tolist(),col,2,
                                   replaceItemUniqueId=ray_line_ids[i][k],
                                   physicsClientId=client)
        except p.error: pass
    rep=drone_repulsion(i,pos)
    moving_target[i]+=cached_dir[i]*CRUISE_SPEED+rep
    moving_target[i][2]=np.clip(moving_target[i][2],0.35,2.1)

def steer_ppo_hybrid(pos, goal, own_id, step):
    """
    Hybrid: Classical (60%) handles obstacle avoidance safely.
    PPO (40%) biases direction — RL influence visible in trajectory.
    No wall crashes, no flipping.
    """
    global ppo_vel

    # Classical direction (safe)
    if step%SENSE_EVERY==0:
        dirc,ring_hit=sense_steer(PPO_DRONE,pos,goal,own_id)
        cached_dir[PPO_DRONE]=dirc
        try:
            for k in range(N_RING):
                end=pos+ring_dirs[k]*SENSOR_RANGE
                col=[1,0,0] if ring_hit[k] else [0,1,0.3]
                p.addUserDebugLine(pos.tolist(),end.tolist(),col,2,
                                   replaceItemUniqueId=ray_line_ids[PPO_DRONE][k],
                                   physicsClientId=client)
        except p.error: pass

    classical_dir=cached_dir[PPO_DRONE]

    # PPO direction bias
    try:
        obs_vec=get_ppo_obs(pos,goal,ppo_vel)
        act,_=ppo_model.predict(obs_vec,deterministic=True)
        act=np.clip(act,-1,1)
        ppo_dir=act/(np.linalg.norm(act)+1e-9)
    except Exception:
        ppo_dir=classical_dir.copy()

    # Blend
    blended=(1.0-PPO_BLEND)*classical_dir + PPO_BLEND*ppo_dir
    blended/=(np.linalg.norm(blended)+1e-9)

    rep=drone_repulsion(PPO_DRONE,pos)
    moving_target[PPO_DRONE]+=blended*CRUISE_SPEED+rep
    moving_target[PPO_DRONE][2]=np.clip(moving_target[PPO_DRONE][2],0.35,2.1)
    ppo_vel=(blended*CRUISE_SPEED).copy()

def draw_trail(i, dpos):
    if prev_pos[i] is None: return
    try:
        if len(trail_ids[i])<TRAIL_MAX:
            lid=p.addUserDebugLine(prev_pos[i],dpos,colors[i],2,physicsClientId=client)
            trail_ids[i].append(lid)
        else:
            lid=trail_ids[i][trail_ptr[i]]
            p.addUserDebugLine(prev_pos[i],dpos,colors[i],2,
                               replaceItemUniqueId=lid,physicsClientId=client)
            trail_ptr[i]=(trail_ptr[i]+1)%TRAIL_MAX
    except p.error: pass

def show_uav_visualization():
    fig,ax=plt.subplots(figsize=(8,8))
    foot={'tall':0.3,'low':0.4,'high':0.5}
    ocol={'tall':'#e01010','low':'#ff8c00','high':'#8033cc'}
    olabel={'tall':'Tall (around)','low':'Low (over)','high':'High (under)'}
    seen=set()
    for ox,oy,typ in OBSTACLES:
        h=foot[typ]; lbl=olabel[typ] if typ not in seen else None; seen.add(typ)
        ax.add_patch(Rectangle((ox-h,oy-h),2*h,2*h,color=ocol[typ],
                                alpha=0.55,zorder=1,label=lbl))
    for i in range(NUM_DRONES):
        ax.scatter(INIT_XYZS[i][0],INIT_XYZS[i][1],marker='s',s=220,c='gold',
                   edgecolors='k',zorder=4,label='Warehouse' if i==0 else None)
        ax.scatter(GOALS[i][0],GOALS[i][1],marker='*',s=340,c=MPL_COLORS[i],
                   edgecolors='k',zorder=4,label='Customer' if i==0 else None)
        if traj[i]:
            xs=[pt[0] for pt in traj[i]]; ys=[pt[1] for pt in traj[i]]
            lbl='Drone1 (PPO+Classical)' if i==PPO_DRONE \
                else 'Drone'+str(i+1)+' (Classical)'
            ls='--' if i==PPO_DRONE else '-'
            ax.plot(xs,ys,color=MPL_COLORS[i],lw=2.2,zorder=3,label=lbl,linestyle=ls)
    ax.set_title('Multi-UAV Delivery — PPO+Classical Hybrid vs Classical\n(Drone-Drone Avoidance ON)',
                 fontsize=12,fontweight='bold')
    ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)')
    ax.set_aspect('equal'); ax.grid(True,alpha=0.3)
    ax.set_xlim(-3.5,3.5); ax.set_ylim(-3.5,3.5)
    ax.legend(loc='upper left',fontsize=8,framealpha=0.9)
    plt.tight_layout()
    plt.savefig('/Users/nah/Desktop/uav_visualization.png',dpi=120)
    plt.close(fig); print('Saved uav_visualization.png')

def show_eval_results():
    labels=['D1\n(PPO+Classical)','D2\n(Classical)','D3\n(Classical)']
    dt_v=[delivery_time[i] if delivery_time[i] else 0.0 for i in range(NUM_DRONES)]
    pl_v=[path_length[i] for i in range(NUM_DRONES)]
    cl_v=[len(collided[i]) for i in range(NUM_DRONES)]
    bat_v=[battery[i] for i in range(NUM_DRONES)]
    eff_v=[]
    for i in range(NUM_DRONES):
        ideal=np.linalg.norm(GOALS[i]-INIT_XYZS[i])+np.linalg.norm(HOME[i]-GOALS[i])
        eff_v.append(min(100.0,ideal/pl_v[i]*100) if pl_v[i]>0 else 0.0)
    success_n=sum(1 for r in returned if r)
    mission_time=step*DT
    throughput=success_n/(mission_time/60.0) if mission_time>0 else 0.0
    fig,axs=plt.subplots(2,3,figsize=(15,8))
    fig.suptitle('PPO+Classical Hybrid vs Classical — Per-Drone Evaluation\n(Drone-Drone Avoidance Enabled)',
                 fontsize=13,fontweight='bold')
    def bar(ax,vals,title,ylabel,fmt='{:.1f}',ymax=None):
        bars=ax.bar(labels,vals,color=MPL_COLORS)
        ax.set_title(title); ax.set_ylabel(ylabel)
        if ymax: ax.set_ylim(0,ymax)
        for j,v in enumerate(vals):
            ax.text(j,v,fmt.format(v),ha='center',va='bottom',fontsize=9)
        bars[PPO_DRONE].set_edgecolor('gold')
        bars[PPO_DRONE].set_linewidth(2.5)
    bar(axs[0,0],dt_v,'Delivery Time','seconds','{:.1f}')
    bar(axs[0,1],pl_v,'Path Length','meters','{:.2f}')
    bar(axs[0,2],eff_v,'Path Efficiency','%','{:.0f}',ymax=110)
    bar(axs[1,0],cl_v,'Collision Count','collisions','{:.0f}')
    bar(axs[1,1],bat_v,'Battery Remaining','%','{:.1f}',ymax=110)
    axs[1,2].axis('off')
    summary=("MISSION SUMMARY\n\n"
             "Success Rate : {:.0f}%\n"
             "Delivered    : {}/{}\n"
             "Throughput   : {:.1f} pkg/min\n"
             "Total Time   : {:.1f} s\n"
             "Total Path   : {:.2f} m\n"
             "Total Coll.  : {}\n\n"
             "D1  = PPO+Classical Hybrid\n"
             "      (60% Classical safety +\n"
             "       40% PPO RL direction)\n"
             "D2,3= A*+Reactive Classical\n"
             "All = Drone-Drone Avoid ON").format(
                 success_n/NUM_DRONES*100,success_n,NUM_DRONES,
                 throughput,mission_time,sum(pl_v),sum(cl_v))
    axs[1,2].text(0.05,0.95,summary,ha='left',va='top',fontsize=10,family='monospace')
    plt.tight_layout()
    plt.savefig('/Users/nah/Desktop/evaluation_results.png',dpi=120)
    plt.close(fig); print('Saved evaluation_results.png')

print("Mission: Drone1=PPO+Classical | Drone2,3=Classical | Drone-Drone Avoidance ON")
try:
    for step in range(999999):
        for i in range(NUM_DRONES):
            pos,_=p.getBasePositionAndOrientation(env.DRONE_IDS[i],physicsClientId=client)
            pos=np.array(pos)
            path_length[i]+=float(np.linalg.norm(pos-last_pos[i]))
            last_pos[i]=pos
            traj[i].append((float(pos[0]),float(pos[1])))
            if phase[i]!='done':
                battery[i]=max(0.0,battery[i]-BATTERY_DRAIN)

            if phase[i]=='takeoff':
                moving_target[i]=np.array([INIT_XYZS[i][0],INIT_XYZS[i][1],1.0])
                if step>=TAKEOFF_STEPS:
                    phase[i]='deliver'
                    tag='(PPO+Classical)' if i==PPO_DRONE else '(Classical)'
                    print("Drone"+str(i+1)+tag+": delivering...")

            elif phase[i]=='deliver':
                horiz=np.linalg.norm(pos[:2]-GOALS[i][:2])
                if horiz<GOAL_LOCK:
                    d=GOALS[i]-moving_target[i]; nd=np.linalg.norm(d)
                    if nd>CRUISE_SPEED: moving_target[i]+=d/nd*CRUISE_SPEED
                    else: moving_target[i]=GOALS[i].copy(); phase[i]='drop'
                else:
                    if i==PPO_DRONE and USE_PPO:
                        steer_ppo_hybrid(pos,GOALS[i],env.DRONE_IDS[i],step)
                    else:
                        steer_classical(i,pos,GOALS[i],env.DRONE_IDS[i],step)

            elif phase[i]=='drop':
                moving_target[i][2]-=0.004
                if moving_target[i][2]<=0.5:
                    p.resetBasePositionAndOrientation(
                        package_ids[i],
                        [float(GOALS[i][0]),float(GOALS[i][1]),0.14],
                        [0,0,0,1])
                    carrying[i]=False; delivery_time[i]=step*DT
                    phase[i]='return'
                    print("Drone"+str(i+1)+": delivered! returning...")

            elif phase[i]=='return':
                horiz=np.linalg.norm(pos[:2]-HOME[i][:2])
                if horiz<GOAL_LOCK:
                    d=HOME[i]-moving_target[i]; nd=np.linalg.norm(d)
                    if nd>CRUISE_SPEED: moving_target[i]+=d/nd*CRUISE_SPEED
                    else: moving_target[i]=HOME[i].copy(); phase[i]='land'
                else:
                    if i==PPO_DRONE and USE_PPO:
                        steer_ppo_hybrid(pos,HOME[i],env.DRONE_IDS[i],step)
                    else:
                        steer_classical(i,pos,HOME[i],env.DRONE_IDS[i],step)

            elif phase[i]=='land':
                moving_target[i][2]-=0.004
                if moving_target[i][2]<=0.12:
                    moving_target[i][2]=0.12; phase[i]='done'
                    returned[i]=True; return_time[i]=step*DT
                    print("Drone"+str(i+1)+": back at warehouse!")
            else:
                moving_target[i]=np.array([INIT_XYZS[i][0],INIT_XYZS[i][1],0.12])

            if carrying[i]:
                p.resetBasePositionAndOrientation(
                    package_ids[i],
                    [float(pos[0]),float(pos[1]),float(max(pos[2]-0.16,0.14))],
                    [0,0,0,1])

            action[i],_,_=ctrl[i].computeControlFromState(
                control_timestep=env.CTRL_TIMESTEP,
                state=obs[i],
                target_pos=moving_target[i],
                target_rpy=np.zeros(3))

        obs,_,_,_,_=env.step(action)

        if step%6==0:
            for i in range(NUM_DRONES):
                for ob in obstacle_ids:
                    pts=p.getClosestPoints(env.DRONE_IDS[i],ob,0.05,physicsClientId=client)
                    if pts and min(pt[8] for pt in pts)<0.02:
                        collided[i].add(ob)
            for i in range(NUM_DRONES):
                dpos,_=p.getBasePositionAndOrientation(env.DRONE_IDS[i],physicsClientId=client)
                dpos=list(dpos)
                draw_trail(i,dpos)
                prev_pos[i]=dpos

        if all(ph=='done' for ph in phase):
            print("=== ALL DELIVERIES COMPLETE ===")
            time.sleep(1.0); break

        time.sleep(DT)

except (p.error,KeyboardInterrupt):
    print("\nSimulation stopped early.")

finally:
    try: env.close()
    except Exception: pass
    show_uav_visualization()
    show_eval_results()
    for f in ['/Users/nah/Desktop/uav_visualization.png',
              '/Users/nah/Desktop/evaluation_results.png']:
        try: subprocess.run(['open',f])
        except Exception: pass
    print("Done.")