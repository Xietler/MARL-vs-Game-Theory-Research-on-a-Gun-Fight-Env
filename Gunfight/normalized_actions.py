import gym
import torch


#using gym.ActionWrapper to normailize origin Network input and output
class NormalizedActions(gym.ActionWrapper):

    def _action(self, action):
        action = (action + 1) / 2  # [-1, 1] => [0, 1]
        action *= (self.action_space.high - self.action_space.low)
        action += self.action_space.low
        return action

    def _reverse_action(self, action):
        action -= self.action_space.low
        action /= (self.action_space.high - self.action_space.low)
        action = action * 2 - 1
        return action

#normalize the action according to the tensor
def normalize(mu,bullet):
    tensor0 = mu.numpy()[0]
    tensor1 = mu.numpy()[1]
    run = (tensor0+1)/2
    run = run*5
    shoot = (tensor1+1)/2
    if bullet == 0:
        shoot = 0
    return round(shoot)*5+(int)(run)

#convert to the action back to tensor
def R_normalize(action):
    run = action%5
    shoot = action//5
    return torch.Tensor([run/5*2-1,shoot])

#change the action when shoot without bullet to avoid misleading state and action
def bullet_normalize(action,bullet):
    tensor0 = action.numpy()[0]
    tensor1 = action.numpy()[1]
    if tensor1 >= 0.5 and bullet == 0:
        tensor1 = 1 - tensor1
    return torch.Tensor([tensor0,tensor1])
