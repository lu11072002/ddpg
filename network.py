from torch import nn


HIDDEN_NETWORK_SIZE = 32

class DDPGnetwork(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super(DDPGnetwork, self).__init__()
        self.actor = actorNet(obs_dim, act_dim)
        self.critic = criticNet(obs_dim + act_dim, 1)
        
    def forward(self, obs):
        return self.actor(obs)
    
    def value(self, obs_and_action):
        return self.critic(obs_and_action)

#add more layer
class actorNet(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super(actorNet, self).__init__()
        
        self.fc1 = nn.Linear(obs_dim, HIDDEN_NETWORK_SIZE, True)
        self.relu1 = nn.ReLU()
        #nn.init.xavier_uniform_(self.fc1.weight)
        #nn.init.zeros_(self.fc1.bias)
        
        self.fc2 = nn.Linear(HIDDEN_NETWORK_SIZE, HIDDEN_NETWORK_SIZE, True)
        self.relu2 = nn.ReLU()
        #nn.init.xavier_uniform_(self.fc2.weight)
            
        self.fc3 = nn.Linear(HIDDEN_NETWORK_SIZE, act_dim, True)
        self.tanh3 = nn.Tanh()
        #nn.init.uniform_(self.fc3.weight, -3e-3, 3e-3)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        
        x = self.fc2(x)
        x = self.relu2(x)
        
        x = self.fc3(x)
        x = self.tanh3(x)
        return x


class criticNet(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super(criticNet, self).__init__()
        
        self.fc1 = nn.Linear(obs_dim, HIDDEN_NETWORK_SIZE, True)
        self.relu1 = nn.ReLU()
        #nn.init.xavier_uniform_(self.fc1.weight)
        #nn.init.zeros_(self.fc1.bias)
        
        self.fc2 = nn.Linear(HIDDEN_NETWORK_SIZE, HIDDEN_NETWORK_SIZE, True)
        self.relu2 = nn.ReLU()
        #nn.init.xavier_uniform_(self.fc2.weight)
            
        self.fc3 = nn.Linear(HIDDEN_NETWORK_SIZE, act_dim, True)
        #nn.init.uniform_(self.fc3.weight, -3e-3, 3e-3)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        
        x = self.fc2(x)
        x = self.relu2(x)
        
        x = self.fc3(x)
        return x
        


    
    
