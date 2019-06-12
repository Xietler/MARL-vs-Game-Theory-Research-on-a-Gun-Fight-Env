# GunFight


### Dependencies：
* **Operating Systems**：Windows10 / Linux in Docker
* **Language**:Python 3.6.x
* **Packages**:gym,numpy,scipy,pandas,tensorboardX,tqdm,argparse,matplotlib,Pytorch 0.4.0,torchvision,pygame


---




### Structure

#### 1.DDPG part

```Python
ddpg.py
param_noise.py
ounoise.py
OptDDPG.py
replay_memory.py
normalized_actions
```

#### 2.PPO part
```Python
PPO.py
```

#### 3.Env part
```Python
Shoot.py
Soldier.py
env.py
utils.pyd
```

#### 4.Test part
```Python
OptDDPGSelf.py
PPOself.py
```
---

### Experiment

#### PPO training and test with Nash
```python
def testppo(train,turn,battle,random_seed,nash_lower = 20,nash_upper = 20)
```
* **train**:training episode
* **turn**:testing turn
* **battle**:battle num per turn
* **random_seed**:change the seed to get avg result
* **nash**:set the distance range for Nash Policy

```Shell
Example:
python PPOself.py 5 20 100 10 20 20
```

#### DDPG training
```python
def OptDDPG_TrainSelf(episode,battle):
```
* **episode**:train episode
* **battle**:battles per episode
  
```Example:
python OptDDPGSelf.py train 150 2000
```

#### DDPG test with Nash Using Model
```python
def OptDDPGtestmodelNash(turn,battle,actor,critic,random_seed,nash_lower = 20,nash_upper = 20):
```
* **turn**:testing turns
* **battle**:battles per turn
* **actor**:actor model path
* **critic**:critic model path
* **random_seed**:change the seed to get avg result
* **nash**:like PPO settings
```Example:
python OptDDPGSelf.py exec 20 100 models/ddpg_actor_SelfB_300000 models/ddpg_Critic_SelfB_300000 10 20 20
```
---
### Environment

* **GUI**
  a demo for auto battle for approximate Nash Solution:game.py

* **Interface**
  ```python
  Env.reset():return to the origin config
  Env.update(actionR,actionB):update the config for env
  Env.stepshoot(actionR,actionB):like gym interface,it will return state,reward,done and info(not implemented)

  Code Example:
       while Env.done == false:
           actionR = agentR.act()
           actionB = agentB.act()
           stateR,stateB,RR,RB,done,_ = Env.stepshoot(actionR,actionB)
           if done:
               break


* **DDPG**
    ```python
    #Create a DDPG agent:
    #observation_space:self position,enemy position,self bullet,enemy bullet
    #action_space:gym.spaces[0,4]and[0,1]
    #[0,4] to int represent the move action
    #[0,1]:above 0.5 means shoot,below means not shoot
    agentR = DDPG(gamma,tau,hidden_size,observation_space,action_space)
    agentB = DDPG(gamma,tau,hidden_size,observation_space,action_space)
    memoryR = Memory()# or OptMemory()
    memoryB = Memory()# or OptMemory()

    #Load a model:
    def load_model(self, actor_path, critic_path):

    #save a model:
    def save_model(self, env_name, suffix="", actor_path=None,critic_path=None):
    ```

* **PPO**
  ```python
    #num_inputs represent observation space:4 dim info
    #num_actions represent discrete space(10)
    networkR = ActorCritic(num_inputs, num_actions, layer_norm=args.layer_norm)
    networkB = ActorCritic(num_inputs, num_actions, layer_norm=args.layer_norm)
    optimizerR = opt.Adam(networkR.parameters(), lr=args.lr)
    optimizerB = opt.Adam(networkB.parameters(), lr=args.lr)
    running_state = ZFilter((num_inputs,), clip=5.0)
  ```
---
###References
* 莫烦强化学习、Pytorch、Tensorflow系列教程 https://morvanzhou.github.io/about/
* OpenAI强化学习的baseline implementation：https://github.com/openai/baselines
* DeepMind官方github：https://github.com/deepmind
* MARL论文集：https://github.com/LantaoYu/MARL-Papers
* RL平台工具：OpenAI gym,ML-Agents@Unity
