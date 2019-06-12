import random
from collections import namedtuple
import queue
import math

# Taken from
# https://github.com/pytorch/tutorials/blob/master/Reinforcement%20(Q-)Learning%20with%20PyTorch.ipynb

Transition = namedtuple(
    'Transition', ('state', 'action', 'mask', 'next_state', 'reward'))

Transitionp = namedtuple('Transition',('priority','state','action','mask','next_state','reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)



#change the replay policy according to the reward
class OptReplayMemory(object):

    def __init__(self,capacity):
        self.capacity = 2*capacity
        self.heap = []
        self.profit_memory = []
        self.p_pos = 0
        self.loss_memory = []
        self.loss_record = []
        self.alpha = 0.5

    def pushP(self,*args):
        if len(self.profit_memory) < self.capacity:
            self.profit_memory.append(Transition(*args))
            self.p_pos = (self.p_pos + 1)%self.capacity
            return self.p_pos

    def pushL(self,*args):
        if len(self.loss_memory) < self.capacity:
            self.loss_memory.append(Transition(*args))

    def sampleP(self,batch_size):
        return random.sample(self.profit_memory,(int)(self.alpha*batch_size))

    def sampleL(self,batch_size):
        return random.sample(self.loss_memory,batch_size-(int)(self.alpha*batch_size))

    def sample(self,batch_size):
        return self.sampleP(batch_size)+self.sampleL(batch_size)

    def pushLR(self,state):
        self.loss_record.append(state)

    def judge(self,state_batch):
        size = len(state_batch)
        count = 0
        for s in state_batch:
            if s in self.loss_record:
                count += 1
        rate = float(count)/size
        if rate < 0.1 and self.alpha > rate:
            self.alpha -= rate
        elif self.alpha + rate < 1:
            self.alpha += rate
        else:
            pass


    def __len__(self):
        return len(self.profit_memory)+len(self.loss_memory)


def buildMaxHeap(arr):
    import math
    for i in range(int(math.floor(len(arr)/2)),-1,-1):
        heapify(arr,i)

def heapify(arr, i):
    left = 2*i+1
    right = 2*i+2
    largest = i
    if left < arrLen and arr[left] > arr[largest]:
        largest = left
    if right < arrLen and arr[right] > arr[largest]:
        largest = right

    if largest != i:
        swap(arr, i, largest)
        heapify(arr, largest)

def swap(arr, i, j):
    arr[i], arr[j] = arr[j], arr[i]

def heapSort(arr):
    global arrLen
    arrLen = len(arr)
    buildMaxHeap(arr)
    for i in range(len(arr)-1,0,-1):
        swap(arr,0,i)
        arrLen -=1
        heapify(arr, 0)
    return arr
