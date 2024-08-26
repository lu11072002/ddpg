import ray
import numpy as np
from env import setup_metaworld_env
import logging
from agent import DDPG_agent
from experience_replay import ReplayMemory
import torch
import random
from PIL import Image
import json
import os


@ray.remote
class Worker:
    def __init__(self, task_name: str, seed: int, worker_id: int) -> None:
        self.worker_id = worker_id
        self.seed = seed
        self.task_name = task_name
        # Initialize the environment with the given task and seed
        self.env = setup_metaworld_env(task_name, seed, False)
        
    def getActionSpace(self):
        return self.env.action_space
    
    def getObservationSpace(self):
        return self.env.observation_space

    def reset(self):
        # Reset the environment and return the initial observation and info
        obs, info = self.env.reset()
        return obs, info, self.worker_id

    def step(self, action, save=False):
        # Take a step in the environment with the given action
        obs, reward, done, truncate, info = self.env.step(np.array(action))
        return obs, reward, done, truncate, info, self.worker_id
    
    def render(self):
        return self.env.render()


def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

def create_vector(size, i):
    vector = np.zeros(size)
    vector[i] = 1
    return vector

def concatenate_obs(obs, size, i):
    return np.concatenate([create_vector(size, i), obs])

def run_episode(workers, agent, rpm):
    nb_workers = len(workers)
    action_space_env = ray.get(workers[0].getActionSpace.remote())
    
    batch_data = ray.get([worker.reset.remote() for worker in workers])
    obs_batch = []
    for obs, _, worker_id in batch_data:
        obs_batch.append(concatenate_obs(obs, nb_workers, worker_id))
    
    step = 0
    total_reward = np.zeros(nb_workers)
    while True:
        batch_action = []
        for obs in obs_batch:
            action = agent.predict(obs)
            action += opt["NOISE"] * np.random.randn(action_space_env.shape[0])
            action = np.clip(action, action_space_env.low,action_space_env.high)
            batch_action.append(action)
        
        batch_data = ray.get([worker.step.remote(action) for worker, action in zip(workers, batch_action)])
        next_obs_batch = []
        reward_batch = []
        done_batch = []

        for (next_obs, reward, done, _, info, worker_id) in batch_data:
            next_obs = concatenate_obs(next_obs, nb_workers, worker_id)
            next_obs_batch.append(next_obs)
            reward_batch.append(reward)
            done_batch.append(done)
            rpm.append((obs_batch[worker_id], action, opt["REWARD_SCALE"] * reward, next_obs, done))

        if len(rpm) > opt["MEMORY_WARMUP_SIZE"] and (step % opt["LEARN_FREQ"]) == 0:
            (batch_obs, batch_action, batch_reward, batch_next_obs,
             batch_done) = rpm.sample(opt["BATCH_SIZE"])
            agent.learn(batch_obs, batch_action, batch_reward, batch_next_obs,
                        batch_done)

        obs = next_obs
        total_reward += np.array(reward_batch)
        step += 1
        if done or step >= 200:
            break
    return step, total_reward

def evaluate(time, worker, agent, vector = None, display = False):
    eval_reward = []
    for i in range(time):
        frames = []
        obs, _, _ = ray.get(worker.reset.remote())
        obs = np.concatenate([vector, obs])
        
        episode_reward = 0
        episode_success = 0
        step = 0
        while True:
            step += 1
            action = agent.predict(obs)
            obs, reward, isOver, _, info, _ = ray.get(worker.step.remote(action))
            obs = np.concatenate([vector, obs])
            
            if display:
                frames.append(Image.fromarray(ray.get(worker.render.remote(action)), 'RGB'))
            episode_reward += reward
            episode_success += info["success"]
            
            if isOver or step >= 200:
                #print(f"Success : {episode_success/200}")
                break
        eval_reward.append(episode_reward)
        if display:
            frames[0].save(f'images/display{i}.gif', save_all=True, append_images=frames[1:], optimize=False, duration=40, loop=0)
    mean_reward = np.mean(eval_reward)
    

    print("evaluating on {} episodes with mean reward {}.".format(time, mean_reward))
    logging.warning("evaluating on {} episodes with mean reward {}.".format(time, mean_reward))
    return mean_reward

def train_ddpg(workers, env_name, agent, episodes, rpm):
    nb_workers = len(workers)
    max_reward = np.ones(nb_workers) * -1e10
    
    for i in range(episodes):
        step, total_reward = run_episode(workers, agent, rpm)
        if i % 10 == 0:
            print("Episode {}, step {} Reward Sum {}.".format(i, step, total_reward))
            logging.warning("Episode {}, step {} Reward Sum {}.".format(i, step, total_reward))
        
        if (i + 1) % 100 == 0:
            for i in range(nb_workers):
                vector = np.zeros(nb_workers)
                vector[i] = 1
                total_reward = evaluate(10, workers[i], agent, vector) 
                if total_reward >= max_reward[i]:
                    max_reward[i] = total_reward
                    agent.save(env_name)
            

opt = {
    "ACTOR_LR" : 3e-4,  # Actor learning rate
    "CRITIC_LR" : 3e-4,  # Critic learning rate
    "GAMMA" : 0.99,      # reward
    "TAU" : 0.005,
    "MEMORY_SIZE" : int(1e6),
    "MEMORY_WARMUP_SIZE" : 500,
    "BATCH_SIZE" : 64,
    "REWARD_SCALE" : 1,
    "NOISE" : 0.01,
    "LEARN_FREQ" : 5,
    "TRAIN_EPISODE" : 2000 # episode
}


def open_opt(name):
    global opt
    try :
        opt = json.load(open(f'data/{name}/param.json'))
    except FileNotFoundError:
        json.dump(opt, open(f'data/{name}/param.json', 'w'))


def train_OneForAll_ray(name, envs):
    if not os.path.exists(f"data/{name}"):
        os.makedirs(f"data/{name}")
        
    ray.init()
    seed = 0
    setup_seed(20)
    open_opt(name)
    logging.basicConfig(filename="data/{}/logger.log".format(name))
    
    workers = [Worker.remote(envs[i], seed + i, i) for i in range(len(envs))]
    
    act_dim = ray.get(workers[0].getActionSpace.remote()).shape[0]
    obs_dim = ray.get(workers[0].getObservationSpace.remote()).shape[0] + len(envs)
    rpm = ReplayMemory(opt["MEMORY_SIZE"])
    agent = DDPG_agent(obs_dim = obs_dim, act_dim = act_dim, actor_lr = opt["ACTOR_LR"], critic_lr = opt["CRITIC_LR"], tau = opt["TAU"], gamma = opt["GAMMA"])

    train_ddpg(workers, name, agent, opt["TRAIN_EPISODE"], rpm)
    
    
# Display never tested
def test_OneForAll_ray(name, env, vector):
    if not os.path.exists(f"data/{name}"):
        print("env never trained")
        return
    
    ray.init()
    seed = 0
    open_opt(name)
        
    worker = Worker.remote(env, seed, 0)
    
    act_dim = ray.get(worker.getActionSpace.remote()).shape[0]
    obs_dim = ray.get(worker.getObservationSpace.remote()).shape[0] + len(vector)
    agent = DDPG_agent(obs_dim = obs_dim, act_dim = act_dim, actor_lr = opt["ACTOR_LR"], critic_lr = opt["CRITIC_LR"], tau = opt["TAU"], gamma = opt["GAMMA"])
    
    agent.load("data/" + name + "/model.pth")
    evaluate(10, worker, agent, vector, display=True)