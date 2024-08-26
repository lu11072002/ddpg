import logging
import numpy as np
from agent import DDPG_agent
from experience_replay import ReplayMemory
import torch
import random
from PIL import Image
import json
import time
import os

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True


def run_episode(env, agent, rpm, vector):
    obs, _ = env.reset()
    obs = np.concatenate([vector, obs])
    
    step = 0
    total_reward = 0
    while True:
        action = agent.predict(obs)
        action += opt["NOISE"] * np.random.randn(env.action_space.shape[0])
        action = np.clip(action, env.action_space.low,env.action_space.high)
        next_obs, reward, done, _, _ = env.step(action)
        next_obs = np.concatenate([vector, next_obs])
        
        rpm.append((obs, action, opt["REWARD_SCALE"] * reward, next_obs, done))

        if len(rpm) > opt["MEMORY_WARMUP_SIZE"] and (step % opt["LEARN_FREQ"]) == 0:
            (batch_obs, batch_action, batch_reward, batch_next_obs,
             batch_done) = rpm.sample(opt["BATCH_SIZE"])
            agent.learn(batch_obs, batch_action, batch_reward, batch_next_obs,
                        batch_done)

        obs = next_obs
        total_reward += reward
        step += 1
        if done or step >= 200:
            break
    return step, total_reward

def evaluate(time, env, agent, vector, display = False, name = None):
    eval_reward = []
    for i in range(time):
        frames = []
        obs = env.reset()[0]
        obs = np.concatenate([vector, obs])
        
        episode_reward = 0
        episode_success = 0
        step = 0
        while True:
            step += 1
            action = agent.predict(obs)
            obs, reward, isOver, _, info = env.step(action)
            obs = np.concatenate([vector, obs])
            
            if display:
                frames.append(Image.fromarray(env.render(), 'RGB'))
            episode_reward += reward
            episode_success += info["success"]
            
            if isOver or step >= 200:
                #print(f"Success : {episode_success/200}")
                break
        eval_reward.append(episode_reward)
        if display:
            frames[0].save(f'images/{name}/display{i}.gif', save_all=True, append_images=frames[1:], optimize=False, duration=40, loop=0)
    mean_reward = np.mean(eval_reward)
    

    print("evaluating on {} episodes with mean reward {}.".format(time, mean_reward))
    logging.warning("evaluating on {} episodes with mean reward {}.".format(time, mean_reward))
    return mean_reward

def train_ddpg(envs, env_name, agent, episodes, rpm):
    max_reward = np.ones(len(envs)) * -1e10
    
    index = 0
    env = envs[index]
    vector = np.zeros(len(envs))
    vector[index] = 1
    for i in range(episodes):
        
        step, total_reward = run_episode(env, agent, rpm, vector)
        if i % 10 == 0:
            print("Episode {}, step {} Reward Sum {} for {}.".format(i, step, total_reward, env.env_name))
            logging.warning("Episode {}, step {} Reward Sum {} for {}.".format(i, step, total_reward, env.env_name))

        if (i + 1) % 100 == 0:
            total_reward = evaluate(10, env, agent, vector) 
            if total_reward >= max_reward[index]:
                max_reward[index] = total_reward
                agent.save(env_name)
            
            index += 1
            if index == len(envs):
                index = 0
            env = envs[index]
            vector = np.zeros(len(envs))
            vector[index] = 1

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


def train_OneForAll(name, envs):
    """Training function for multitask

    Args:
        name (str): name of the folder saved into data
        envs (list): list of the metaworld environement to train
    """
    if not os.path.exists(f"data/{name}"):
        os.makedirs(f"data/{name}")
        
    setup_seed(20)
    open_opt(name)
    logging.basicConfig(filename="data/{}/logger.log".format(name))
    
    act_dim = envs[0].action_space.shape[0]
    obs_dim = envs[0].observation_space.shape[0] + len(envs)
    rpm = ReplayMemory(opt["MEMORY_SIZE"])
    agent = DDPG_agent(obs_dim = obs_dim, act_dim = act_dim, actor_lr = opt["ACTOR_LR"], critic_lr = opt["CRITIC_LR"], tau = opt["TAU"], gamma = opt["GAMMA"])
    
    random.seed(time.time()) # not necessary
    train_ddpg(envs, name, agent, opt["TRAIN_EPISODE"], rpm)


def test_OneForAll(name, env, vector):
    """Training function for multitask

    Args:
        name (str): name of the folder saved into data
        env (environement): metaworld environement to test
        vector(list): vector with all zeros and a number one at the position of the environement (list for the training)
    """
    if not os.path.exists(f"data/{name}"):
        print("env never trained")
        return
    
    if not os.path.exists(f"images/{name}"):
        os.makedirs(f"images/{name}")
    
    open_opt(name)
    act_dim = env.action_space.shape[0]
    obs_dim = env.observation_space.shape[0] + len(vector)
    agent = DDPG_agent(obs_dim = obs_dim, act_dim = act_dim, actor_lr = opt["ACTOR_LR"], critic_lr = opt["CRITIC_LR"], tau = opt["TAU"], gamma = opt["GAMMA"])
    
    agent.load("data/" + name + "/model.pth")
    evaluate(10, env, agent, vector, display=True, name=name)

