import Shoot
import random
import os
import sys
import time

class Soldier(object):
    bullet_count = 1
    bullettype = 1
    PJL = [0,0,0]
    pos = []
    team = 'R'
    size = 10
    alive = False
    reward = 0


    def __init__(self,bullet = 1,team = 'R',size = 10):
        if team == 'R':
            self.pos = [size,1]
            self.bullettype = 1
        else:
            self.pos = [1,size]
            self.bullettype = 2
            self.team = 'B'
        self.bullet_count = bullet
        self.alive = True

    def __del__(self):
        return

    def shoot(self):
        if self.bullet_count > 0 :
            self.bullet_count -= 1
            return True
        else:
            return False

    def move(self,command):
        if command == 2 and self.pos[1] < self.size:#Right
             self.pos[1] += 1
             return True
        if command == 0 and self.pos[1] > 1:#Left
             self.pos[1] -= 1
             return True
        if command == 1 and self.pos[0] > 1:#Up
             self.pos[0] -= 1
             return True
        if command == 3 and self.pos[0] < self.size:#Down
             self.pos[0] += 1
             return True
        if command == 4:#Still
             return True
        return False

    def dst(self,distance):
        Arr = Shoot.distance(10)
        if distance <= Arr[20] and distance >= Arr[20] and self.bullet_count == 1:
            self.shoot()
            return 1



    def killed(self):
        self.alive = False

    def inspect(self,i_command):
        self.PJL = i_command

    def random_move(self):
        return random.randint(0,4)


    def ou_distance(self,enemy):
        return abs(enemy[0]-self.pos[0])+abs(enemy[1]-self.pos[1])

    def act(self,dis,lower=20,upper=20):
        Arr = Shoot.distance(10)
        if( self.bullet_count == 1 and dis <= Arr[lower] and dis >= Arr[upper]):
            action = random.randint(5,9)
            return action
        else:
            action = random.randint(0,4)
            return action












