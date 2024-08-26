import logging
import numpy as np
from agent import DDPG_agent
from experience_replay import ReplayMemory
import torch
import random
from PIL import Image
import json
import os

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True


def run_episode(env, agent, rpm):
    obs, _ = env.reset()
    step = 0
    total_reward = 0
    while True:
        action = agent.predict(obs)
        action += opt["NOISE"] * np.random.randn(env.action_space.shape[0])
        action = np.clip(action, env.action_space.low,env.action_space.high)
        next_obs, reward, done, _, _ = env.step(action)
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

def evaluate(time, env, agent, display = False):
    eval_reward = []
    for i in range(time):
        frames = []
        obs = env.reset()[0]
        episode_reward = 0
        episode_success = 0
        step = 0
        while True:
            step += 1
            action = agent.predict(obs)
            obs, reward, isOver, _, info = env.step(action)
            if display:
                frames.append(Image.fromarray(env.render(), 'RGB'))
            episode_reward += reward
            episode_success += info["success"]
            
            if isOver or step >= 200:
                #print(f"Success : {episode_success/200}")
                break
        eval_reward.append(episode_reward)
        if display:
            frames[0].save(f'images/{env.env_name}/display{i}.gif', save_all=True, append_images=frames[1:], optimize=False, duration=40, loop=0)
    mean_reward = np.mean(eval_reward)
    

    print("evaluating on {} episodes with mean reward {}.".format(time, mean_reward))
    logging.warning("evaluating on {} episodes with mean reward {}.".format(time, mean_reward))
    return mean_reward

def train_ddpg(env, env_name, agent, episodes, rpm):
    max_reward = -1e10
    while len(rpm) < opt["MEMORY_WARMUP_SIZE"]:
        run_episode(env, agent, rpm)
    for i in range(episodes):
        step, total_reward = run_episode(env, agent, rpm)
        if i % 10 == 0:
            print("Episode {}, step {} Reward Sum {}.".format(i, step, total_reward))
            logging.warning("Episode {}, step {} Reward Sum {}.".format(i, step, total_reward))

        if (i + 1) % 100 == 0:
            total_reward = evaluate(10, env, agent) 
            if total_reward >= max_reward:
                max_reward = total_reward
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


def train_OneForEach(env):
    """Training function for monotask

    Args:
        env (environement): give the metaworld environement to train
    """
    name = env.env_name
    if not os.path.exists(f"data/{name}"):
        os.makedirs(f"data/{name}")
        
    setup_seed(20)
    open_opt(name)
    logging.basicConfig(filename="data/{}/logger.log".format(name))
    
    act_dim = env.action_space.shape[0]
    obs_dim = env.observation_space.shape[0]
    rpm = ReplayMemory(opt["MEMORY_SIZE"])
    agent = DDPG_agent(obs_dim = obs_dim, act_dim = act_dim, actor_lr = opt["ACTOR_LR"], critic_lr = opt["CRITIC_LR"], tau = opt["TAU"], gamma = opt["GAMMA"])

    train_ddpg(env, name, agent, opt["TRAIN_EPISODE"], rpm)


def test_OneForEach(env):
    """Testing function for monotask

    Args:
        env (environement): give the metaworld environement to test
    """
    name = env.env_name
    if not os.path.exists(f"data/{name}"):
        print("env never trained")
        return
    
    if not os.path.exists(f"images/{name}"):
        os.makedirs(f"images/{name}")
    
    open_opt(name)
    act_dim = env.action_space.shape[0]
    obs_dim = env.observation_space.shape[0]
    agent = DDPG_agent(obs_dim = obs_dim, act_dim = act_dim, actor_lr = opt["ACTOR_LR"], critic_lr = opt["CRITIC_LR"], tau = opt["TAU"], gamma = opt["GAMMA"])
    
    agent.load("data/" + name + "/model.pth")
    evaluate(10, env, agent, display=True)
