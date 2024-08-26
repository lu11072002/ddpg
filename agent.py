from network import DDPGnetwork
import copy
import torch
import numpy as np
from torch import nn, optim
from torch.distributions.categorical import Categorical


class DDPG_agent():
    def __init__(self, obs_dim, act_dim, actor_lr, critic_lr, tau, gamma):
        self.gamma = gamma
        self.tau = tau
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
        self.model = DDPGnetwork(obs_dim, act_dim)
        self.target_model = DDPGnetwork(obs_dim, act_dim)
        self.target_model.load_state_dict(copy.deepcopy(self.model.state_dict()))
        self.actor_optim = optim.Adam(self.model.actor.parameters(), self.actor_lr)
        self.critic_optim = optim.Adam(self.model.critic.parameters(), self.critic_lr)
        
    def predict(self, obs):
        with torch.no_grad():
            self.model.to(self.device)
            obs = np.expand_dims(obs, axis = 0)
            obs = torch.tensor(obs, dtype=torch.float32).to(self.device)
            return self.model(obs).detach().cpu().numpy()[0]
        
            m = Categorical(self.model(obs))
            return m.sample().item()

    def learn(self, obs, action, reward, next_obs, terminal):
        self._critic_learn(obs, action, reward, next_obs,terminal)
        self._actor_learn(obs)
        self.sync_target()

    def _actor_learn(self, obs):
        self.model.to(self.device)
        self.model.train()
        obs = torch.tensor(obs, dtype=torch.float32).to(self.device)
        action = self.model(obs)
        obs_and_act = torch.cat([obs, action], dim = -1)
        Q = self.target_model.value(obs_and_act)
        loss = torch.mean(-1.0 * Q)
        self.actor_optim.zero_grad()
        loss.backward()
        self.actor_optim.step()

    def _critic_learn(self, obs, act, reward, next_obs, terminal):
        self.model.to(self.device)
        self.model.train()
        terminal = np.expand_dims(terminal, axis = -1)
        reward = np.expand_dims(reward, axis = -1)
        obs, act, reward, next_obs, terminal = torch.tensor(obs, dtype = torch.float32), torch.tensor(act, dtype = torch.float32), torch.tensor(reward, dtype = torch.float32), torch.tensor(next_obs, dtype = torch.float32), torch.tensor(terminal, dtype = torch.float32)
        obs, act, reward, next_obs, terminal = obs.to(self.device), act.to(self.device), reward.to(self.device), next_obs.to(self.device), terminal.to(self.device)
        
        self.target_model.to(self.device)
        self.target_model.eval()
        with torch.no_grad():
            next_action = self.target_model(next_obs)
            obs_and_act = torch.cat([next_obs, next_action.detach()], dim = -1)
            next_Q = self.target_model.value(obs_and_act)
            target_Q = reward + (1.0 - terminal) * self.gamma * next_Q

        obs_and_act2 = torch.cat([obs, act], dim = -1) 
        Q = self.model.value(obs_and_act2)
        loss = nn.MSELoss()(Q, target_Q.detach())
        self.critic_optim.zero_grad()
        loss.backward()
        self.critic_optim.step()

    def sync_target(self, decay=None):
        self.target_model.to("cpu")
        self.model.to("cpu")
        if decay is None:
            decay = self.tau
        for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - decay) +
                param.data * decay
            )
            
    def save(self, name):
        torch.save(self.model, "data/" + name + "/model.pth")
        
    def load(self, path):
        self.model = torch.load(path, map_location="cuda:0" if torch.cuda.is_available() else "cpu")
        self.sync_target()
