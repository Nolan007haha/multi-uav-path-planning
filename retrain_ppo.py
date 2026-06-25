import numpy as np, os, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt, subprocess
import gymnasium as gym
from gymnasium import spaces
import pybullet as p, pybullet_data
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

class UAVDeliveryEnv(gym.Env):
    metadata = {"render_modes": []}
    def __init__(self):
        super().__init__()
        self.max_steps = 400
        self.observation_space = spaces.Box(-np.inf, np.inf, (14,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (3,), dtype=np.float32)
        self.client = p.connect(p.DIRECT)
        p.setGravity(0,0,-9.81,physicsClientId=self.client)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self._step=0; self.drone_id=None; self.obstacle_ids=[]
        self.goal=np.array([0.0,4.0,1.0]); self.prev_dist=None; self.vel=np.zeros(3)

    def _build_world(self):
        p.resetSimulation(physicsClientId=self.client)
        p.setGravity(0,0,-9.81,physicsClientId=self.client)
        p.loadURDF("plane.urdf",physicsClientId=self.client)
        col=p.createCollisionShape(p.GEOM_SPHERE,radius=0.2,physicsClientId=self.client)
        vis=p.createVisualShape(p.GEOM_SPHERE,radius=0.2,rgbaColor=[0,0.5,1,1],physicsClientId=self.client)
        self.drone_id=p.createMultiBody(0.5,col,vis,[0,0,1.0],physicsClientId=self.client)
        self.obstacle_ids=[]
        for ox,oy,oz in [(0,1.5,1.0),(0,2.8,0.6),(-0.5,3.5,1.3)]:
            c=p.createCollisionShape(p.GEOM_BOX,halfExtents=[0.3,0.3,oz],physicsClientId=self.client)
            v=p.createVisualShape(p.GEOM_BOX,halfExtents=[0.3,0.3,oz],rgbaColor=[1,0.2,0.2,1],physicsClientId=self.client)
            self.obstacle_ids.append(p.createMultiBody(0,c,v,[ox,oy,oz],physicsClientId=self.client))

    def _get_lidar(self,pos):
        hits=[]
        for a in np.linspace(0,2*np.pi,8,endpoint=False):
            d=np.array([np.cos(a),np.sin(a),0.0])
            res=p.rayTest(pos.tolist(),(pos+d*3.0).tolist(),physicsClientId=self.client)[0]
            hits.append(res[2] if res[0]>0 else 1.0)
        return np.array(hits,dtype=np.float32)

    def _get_obs(self):
        pos,_=p.getBasePositionAndOrientation(self.drone_id,physicsClientId=self.client)
        pos=np.array(pos)
        return np.concatenate([self.goal-pos,self.vel,self._get_lidar(pos)]).astype(np.float32)

    def reset(self,seed=None,options=None):
        super().reset(seed=seed); self._build_world()
        self._step=0; self.vel=np.zeros(3)
        start=np.array([np.random.uniform(-0.5,0.5),0.0,1.0])
        p.resetBasePositionAndOrientation(self.drone_id,start.tolist(),[0,0,0,1],physicsClientId=self.client)
        self.prev_dist=np.linalg.norm(self.goal-start)
        return self._get_obs(),{}

    def step(self,action):
        self._step+=1
        pos,_=p.getBasePositionAndOrientation(self.drone_id,physicsClientId=self.client)
        pos=np.array(pos); self.vel=np.clip(action,-1,1)*0.15
        new_pos=np.clip(pos+self.vel,[-3,-1,0.3],[3,6,2.5])
        p.resetBasePositionAndOrientation(self.drone_id,new_pos.tolist(),[0,0,0,1],physicsClientId=self.client)
        p.stepSimulation(physicsClientId=self.client)
        dist=np.linalg.norm(self.goal-new_pos)
        reward=(self.prev_dist-dist)*5.0-0.05; self.prev_dist=dist
        terminated=False; truncated=False
        if dist<0.4: reward+=100.0; terminated=True
        for oid in self.obstacle_ids:
            pts=p.getClosestPoints(self.drone_id,oid,0.15,physicsClientId=self.client)
            if pts: reward-=50.0; terminated=True; break
        if new_pos[2]<0.3 or self._step>=self.max_steps: truncated=True
        return self._get_obs(),float(reward),terminated,truncated,{}

    def close(self): p.disconnect(self.client)

class MetricsCallback(BaseCallback):
    def __init__(self,check_freq=2000):
        super().__init__(); self.check_freq=check_freq
        self.ep_rewards=[]; self.successes=[]
        self.log_rewards=[]; self.log_success=[]; self.log_steps=[]; self._cur=0.0
    def _on_step(self):
        self._cur+=self.locals["rewards"][0]
        if self.locals["dones"][0]:
            self.ep_rewards.append(self._cur)
            self.successes.append(1 if self._cur>50 else 0)
            self._cur=0.0
        if self.num_timesteps%self.check_freq==0 and self.ep_rewards:
            w=min(30,len(self.ep_rewards))
            self.log_rewards.append(np.mean(self.ep_rewards[-w:]))
            self.log_success.append(np.mean(self.successes[-w:])*100)
            self.log_steps.append(self.num_timesteps)
            print("Step {:>7d} | Reward {:.1f} | Success {:.0f}%".format(
                self.num_timesteps,self.log_rewards[-1],self.log_success[-1]))
        return True

print("=== Continuing PPO: 80k -> 200k ===")
env = UAVDeliveryEnv()
cb  = MetricsCallback(check_freq=2000)
model_path = os.path.expanduser("~/Desktop/ppo_uav_delivery")
model = PPO.load(model_path, env=env)
print("Loaded model, continuing from 80k...")
model.learn(total_timesteps=120000, callback=cb, reset_num_timesteps=False)
model.save(model_path)
env.close()
print("Model saved (200k total).")

fig,axs=plt.subplots(1,2,figsize=(12,4))
fig.suptitle("PPO Training (200k steps) — Multi-UAV Delivery",fontsize=13,fontweight='bold')
axs[0].plot(cb.log_steps,cb.log_rewards,color='#4a7fe0',lw=2)
axs[0].fill_between(cb.log_steps,cb.log_rewards,alpha=0.15,color='#4a7fe0')
axs[0].set_title("Avg Episode Reward (Steps 80k-200k)")
axs[0].set_xlabel("Timesteps"); axs[0].set_ylabel("Reward")
axs[0].axhline(0,color='gray',lw=0.8,ls='--')
axs[1].plot(cb.log_steps,cb.log_success,color='#4caf50',lw=2)
axs[1].fill_between(cb.log_steps,cb.log_success,alpha=0.15,color='#4caf50')
axs[1].set_title("Success Rate (Steps 80k-200k)")
axs[1].set_xlabel("Timesteps"); axs[1].set_ylabel("%")
axs[1].set_ylim(0,105)
plt.tight_layout()
out=os.path.expanduser("~/Desktop/ppo_training_200k.png")
plt.savefig(out,dpi=120); plt.close()
subprocess.run(["open",out])
print("Saved",out)