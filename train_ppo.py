import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pybullet as p
import pybullet_data
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

# ── Lightweight single-drone env (no GUI, fast reset) ──────────────────────
class UAVDeliveryEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()
        self.max_steps = 400
        self.dt = 1/60.0
        # obs: rel_pos(3) + velocity(3) + 8 lidar rays = 14
        self.observation_space = spaces.Box(-np.inf, np.inf, (14,), dtype=np.float32)
        # action: dx dy dz in [-1,1]
        self.action_space = spaces.Box(-1.0, 1.0, (3,), dtype=np.float32)

        self.client = p.connect(p.DIRECT)
        p.setGravity(0, 0, -9.81, physicsClientId=self.client)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self._step = 0
        self.drone_id = None
        self.obstacle_ids = []
        self.goal = np.array([0.0, 4.0, 1.0])
        self.prev_dist = None
        self.vel = np.zeros(3)

    def _build_world(self):
        p.resetSimulation(physicsClientId=self.client)
        p.setGravity(0, 0, -9.81, physicsClientId=self.client)
        p.loadURDF("plane.urdf", physicsClientId=self.client)
        # drone as small sphere
        col = p.createCollisionShape(p.GEOM_SPHERE, radius=0.2, physicsClientId=self.client)
        vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.2, rgbaColor=[0,0.5,1,1], physicsClientId=self.client)
        self.drone_id = p.createMultiBody(0.5, col, vis, [0,0,1.0], physicsClientId=self.client)
        # 3 obstacles
        obs_defs = [(0,1.5,1.0,'tall'),(0,2.8,0.6,'low'),(-0.5,3.5,1.3,'tall')]
        self.obstacle_ids = []
        for ox,oy,oz,_ in obs_defs:
            c = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.3,0.3,oz], physicsClientId=self.client)
            v = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.3,0.3,oz], rgbaColor=[1,0.2,0.2,1], physicsClientId=self.client)
            self.obstacle_ids.append(p.createMultiBody(0, c, v, [ox,oy,oz], physicsClientId=self.client))

    def _get_lidar(self, pos):
        angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
        hits = []
        for a in angles:
            d = np.array([np.cos(a), np.sin(a), 0.0])
            end = pos + d*3.0
            res = p.rayTest(pos.tolist(), end.tolist(), physicsClientId=self.client)[0]
            hits.append(res[2] if res[0] > 0 else 1.0)
        return np.array(hits, dtype=np.float32)

    def _get_obs(self):
        pos, _ = p.getBasePositionAndOrientation(self.drone_id, physicsClientId=self.client)
        pos = np.array(pos)
        rel = self.goal - pos
        lidar = self._get_lidar(pos)
        return np.concatenate([rel, self.vel, lidar]).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._build_world()
        self._step = 0
        self.vel = np.zeros(3)
        start = np.array([np.random.uniform(-0.5,0.5), 0.0, 1.0])
        p.resetBasePositionAndOrientation(self.drone_id, start.tolist(), [0,0,0,1], physicsClientId=self.client)
        self.prev_dist = np.linalg.norm(self.goal - start)
        return self._get_obs(), {}

    def step(self, action):
        self._step += 1
        pos, _ = p.getBasePositionAndOrientation(self.drone_id, physicsClientId=self.client)
        pos = np.array(pos)
        self.vel = np.clip(action, -1, 1) * 0.15
        new_pos = np.clip(pos + self.vel, [-3,-1,0.3],[3,6,2.5])
        p.resetBasePositionAndOrientation(self.drone_id, new_pos.tolist(), [0,0,0,1], physicsClientId=self.client)
        p.stepSimulation(physicsClientId=self.client)

        dist = np.linalg.norm(self.goal - new_pos)
        # reward shaping
        reward = (self.prev_dist - dist) * 5.0   # progress
        reward -= 0.05                             # time penalty
        self.prev_dist = dist

        terminated = False
        truncated = False

        # goal reached
        if dist < 0.4:
            reward += 100.0
            terminated = True

        # collision check
        for oid in self.obstacle_ids:
            pts = p.getClosestPoints(self.drone_id, oid, 0.15, physicsClientId=self.client)
            if pts:
                reward -= 50.0
                terminated = True
                break

        # out of bounds / too long
        if new_pos[2] < 0.3 or self._step >= self.max_steps:
            truncated = True

        return self._get_obs(), float(reward), terminated, truncated, {}

    def close(self):
        p.disconnect(self.client)


# ── Callback: log reward + success rate every N steps ──────────────────────
class MetricsCallback(BaseCallback):
    def __init__(self, check_freq=1000):
        super().__init__()
        self.check_freq = check_freq
        self.ep_rewards = []
        self.ep_lengths = []
        self.successes = []
        self.log_rewards = []
        self.log_success = []
        self.log_steps = []
        self._cur_reward = 0.0
        self._cur_len = 0

    def _on_step(self):
        self._cur_reward += self.locals["rewards"][0]
        self._cur_len += 1
        info = self.locals["infos"][0]
        if self.locals["dones"][0]:
            self.ep_rewards.append(self._cur_reward)
            self.ep_lengths.append(self._cur_len)
            # success = episode ended with reward spike (goal reached ~+100)
            self.successes.append(1 if self._cur_reward > 50 else 0)
            self._cur_reward = 0.0
            self._cur_len = 0

        if self.num_timesteps % self.check_freq == 0 and self.ep_rewards:
            window = min(20, len(self.ep_rewards))
            self.log_rewards.append(np.mean(self.ep_rewards[-window:]))
            self.log_success.append(np.mean(self.successes[-window:]) * 100)
            self.log_steps.append(self.num_timesteps)
            print("Step {:>7d} | Avg Reward {:.1f} | Success {:.0f}%".format(
                self.num_timesteps,
                self.log_rewards[-1],
                self.log_success[-1]))
        return True


# ── Train ──────────────────────────────────────────────────────────────────
def train():
    print("=== PPO Training: Multi-UAV Delivery (single-drone env) ===")
    env = UAVDeliveryEnv()
    cb = MetricsCallback(check_freq=1000)

    model = PPO(
        "MlpPolicy", env,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=128,
        n_epochs=5,
        gamma=0.99,
        ent_coef=0.01,
        verbose=0,
        device="cpu",   # M1 MPS can be unstable with SB3; cpu is safe
    )

    TOTAL = 80_000   # ~25-40 min on M1; bump to 150_000 if you have time
    model.learn(total_timesteps=TOTAL, callback=cb, progress_bar=False)
    model.save(os.path.expanduser("~/Desktop/ppo_uav_delivery"))
    env.close()
    print("Model saved.")
    return cb

# ── Plot ───────────────────────────────────────────────────────────────────
def plot(cb):
    steps = cb.log_steps

    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("PPO Training — Multi-UAV Delivery", fontsize=14, fontweight='bold')

    # reward curve
    axs[0].plot(steps, cb.log_rewards, color='#4a7fe0', lw=2)
    axs[0].fill_between(steps, cb.log_rewards, alpha=0.15, color='#4a7fe0')
    axs[0].set_title("Avg Episode Reward"); axs[0].set_xlabel("Timesteps"); axs[0].set_ylabel("Reward")
    axs[0].axhline(0, color='gray', lw=0.8, ls='--')

    # success rate
    axs[1].plot(steps, cb.log_success, color='#4caf50', lw=2)
    axs[1].fill_between(steps, cb.log_success, alpha=0.15, color='#4caf50')
    axs[1].set_title("Success Rate (%)"); axs[1].set_xlabel("Timesteps"); axs[1].set_ylabel("%")
    axs[1].set_ylim(0, 105)

    # episode count
    counts = list(range(1, len(cb.ep_rewards)+1))
    window = 20
    smoothed = [np.mean(cb.ep_rewards[max(0,i-window):i+1]) for i in range(len(cb.ep_rewards))]
    axs[2].plot(counts, cb.ep_rewards, color='#e0e0e0', lw=0.8, alpha=0.6, label='raw')
    axs[2].plot(counts, smoothed, color='#e05050', lw=2, label='smoothed (20-ep)')
    axs[2].set_title("Reward per Episode"); axs[2].set_xlabel("Episode"); axs[2].set_ylabel("Reward")
    axs[2].legend(fontsize=8)

    plt.tight_layout()
    out = os.path.expanduser("~/Desktop/ppo_training_results.png")
    plt.savefig(out, dpi=120)
    plt.close()
    print("Saved", out)
    import subprocess; subprocess.run(['open', out])

if __name__ == "__main__":
    cb = train()
    plot(cb)
    print("=== DONE ===")