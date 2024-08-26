from gymnasium.core import Env
import numpy as np
import gymnasium
from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer
from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE


DEFAULT_CAMERA_CONFIG = {
    "distance": 1.25,
    "azimuth": 145,
    "elevation": -25.0,
    "lookat": np.array([0, 0.65, 0.0]),
    }
DEFAULT_SIZE=224


class VierWrapper(gymnasium.Wrapper):
    def __init__(self, env: Env, seed, env_name):
        super().__init__(env)

        self.unwrapped.model.vis.global_.offwidth = DEFAULT_SIZE
        self.unwrapped.model.vis.global_.offheight = DEFAULT_SIZE
        self.unwrapped.mujoco_renderer = MujocoRenderer(env.model, env.data, DEFAULT_CAMERA_CONFIG, DEFAULT_SIZE, DEFAULT_SIZE)

        # Hack: enable random reset
        self.unwrapped._freeze_rand_vec = False
        self.unwrapped.seed(seed)
        
        self.env_name = env_name

    def reset(self):
        obs, info = super().reset()
        self.obs = obs
        return obs, info

    def step(self, action):
        next_obs, reward, done, truncate, info = self.env.step(action) 
        self.obs = next_obs
        return next_obs, reward, done, truncate, info

def setup_reach_metaworld_env(seed):
    env_cls = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE['reach-v2-goal-observable']
    env = VierWrapper(env_cls(render_mode="rgb_array"), seed, 'reach-v2-goal-observable')
    return env

def setup_drawer_close_metaworld_env(seed):
    env_cls = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE['drawer-close-v2-goal-observable']
    env = VierWrapper(env_cls(render_mode="rgb_array"), seed, 'drawer-close-v2-goal-observable')
    return env

def setup_window_open_metaworld_env(seed):
    env_cls = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE['window-open-v2-goal-observable']
    env = VierWrapper(env_cls(render_mode="rgb_array"), seed, 'window-open-v2-goal-observable')
    return env

def setup_metaworld_env(task, seed, aaa):
    env_cls = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[task]
    env = VierWrapper(env_cls(render_mode="rgb_array"), seed, task)
    return env