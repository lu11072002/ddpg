# DDPG : Monotask or multitask

Implementation of DDPG (Deep Deterministic Policy Gradient) for the monotask and multitask.

![Reach environement](https://github.com/lu11072002/ddpg/blob/main/readme-images/reach.gif "Reach environement")
![Drawer close environement](https://github.com/lu11072002/ddpg/blob/main/readme-images/drawer-close.gif "Drawer close environement")
![window open environement](https://github.com/lu11072002/ddpg/blob/main/readme-images/window-open.gif "window open environement")

## Training

During the session, the agent save his model in `data/[env]/model.pth`. He save too the log and the parameters used.

For all the environement, a pramaters file was created for a better customization of the pramater for each environement.

### Monotask

![Monotask](https://github.com/lu11072002/ddpg/blob/main/readme-images/graph_OneForEach.jpg "Monotask")

### Multitask

![Multitask](https://github.com/lu11072002/ddpg/blob/main/readme-images/graph_OneForAll.jpg "Multitask")

### Example script

```python
from ddpg_one_for_each import train_OneForEach
from ddpg_one_for_all import train_OneForAll
from env import setup_drawer_close_metaworld_env, setup_reach_metaworld_env, setup_window_open_metaworld_env


# Training monotask agent
env = setup_window_open_metaworld_env(0)
train_OneForEach(env)

# Training multitask agent
envs = [setup_drawer_close_metaworld_env(0),setup_reach_metaworld_env(0),setup_window_open_metaworld_env(0)]
train_OneForAll("OneForAll", envs)
```

## Testing

During the test session, the agent laud the model and param created before, and save the display images on the folder `image/[env]/display[X].gif`

### Example script

```python
from ddpg_one_for_each import test_OneForEach
from ddpg_one_for_all import test_OneForAll
from env import setup_drawer_close_metaworld_env, setup_reach_metaworld_env, setup_window_open_metaworld_env

# Setup environement to test
env = setup_drawer_close_metaworld_env(0)

# Testing monotask agent
test_OneForEach(env)

# Testing multitask agent
test_OneForAll("OneForAll", env, [1,0,0])
```
