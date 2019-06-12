import Soldier
import Shoot
import numpy as np
import math
import random

class env(object):
    R_Soldier = []
    B_Soldier = []
    vsmode = 1
    size = 10
    done = False
    win = 0
    turn = -1
    state = np.zeros((size,size))
    R_r = 0
    B_r = 0

    def __init__(self,vsmode = 1,size = 10):
        self.vsmode = vsmode
        self.size = size
        for i in range(vsmode):
            tempR = Soldier.Soldier(1,'R',10)
            tempB = Soldier.Soldier(1,'B',10)
            self.R_Soldier.append(tempR)
            self.B_Soldier.append(tempB)
        self.done = False
        self.win = 0
        self.turn = -1
        self.R_r = 0
        self.B_r = 0


    def __del__(self):
        return

    def reset(self):
        self.vsmode = 1
        self.size = 10
        self.R_Soldier.clear()
        self.B_Soldier.clear()
        for i in range(self.vsmode):
            tempR = Soldier.Soldier(1, 'R', 10)
            tempB = Soldier.Soldier(1, 'B', 10)
            self.R_Soldier.append(tempR)
            self.B_Soldier.append(tempB)
        self.done = False
        self.win = -1
        self.turn = -1
        self.R_r = 0
        self.B_r = 0
        return self.stepshoot(4, 4)

    def update(self,actionR,actionB,dis):
        self.R_r = 0
        self.B_r = 0
        self.turn = self.turn + 1
        #dis = self.calcdist(self.R_Soldier[0],self.B_Soldier[0])
        if( actionR[1] == 1 and self.R_Soldier[0].bullet_count > 0 ):
            prob1 = Shoot.f1(dis)
            r = random.randint(1,10000)
            if float(r)/10000 < prob1:
                self.B_Soldier[0].alive = False
                self.R_Soldier[0].bullet_count -= 1
                self.done = True
                self.R_r += 5
                self.B_r -= 3
                #print("R kill B")
            else:
                self.R_r -= 1
                self.R_Soldier[0].bullet_count -= 1
                #print("R fail to kill B")
        if actionB[1] == 1 and self.B_Soldier[0].bullet_count > 0 :
            prob2 = Shoot.f2(dis)
            r = random.randint(1,10000)
            if float(r)/10000 < prob2:
                self.R_Soldier[0].alive = False
                self.B_Soldier[0].bullet_count -= 1
                self.done = True
                self.B_r += 5
                self.R_r -= 3
                #print("B kill R")
            else:
                self.B_r -= 1
                self.B_Soldier[0].bullet_count -= 1
                #print("B fail to kill R")
        if( self.R_Soldier[0].alive == False ):
            self.R_Soldier.clear()
        if( self.B_Soldier[0].alive == False ):
            self.B_Soldier.clear()
        if( self.gameover() ):
            self.done = True
            return
        else:
            self.R_Soldier[0].move(actionR[0])
            self.B_Soldier[0].move(actionB[0])
            self.R_r = self.R_r + 0.05 - 0.04*self.R_Soldier[0].bullet_count
            self.B_r = self.B_r + 0.05 - 0.04*self.B_Soldier[0].bullet_count

        self.state = np.zeros((self.size,self.size))
        for R in self.R_Soldier:
          self.state[R.pos[0]-1][R.pos[1]-1] += 1
        for B in self.B_Soldier:
          self.state[B.pos[0]-1][B.pos[1]-1] += 10

    def gameover(self):
        if len(self.R_Soldier) == 0 and len(self.B_Soldier) > 0:
            self.done = True
            self.win = -1
            return True
        elif len(self.R_Soldier) > 0 and len(self.B_Soldier) == 0:
            self.done = True
            self.win = 1
            return True
        elif len(self.R_Soldier) == 0 and len(self.B_Soldier) == 0:
            self.done = True
            self.win = 0
            return True
        elif self.R_Soldier[0].bullet_count == 0 and self.B_Soldier[0].bullet_count == 0:
            self.done = True
            self.win = 0
            return True
        elif self.turn >=1000:
            self.win = 0
            self.done = True
            return True
        else:
            return False

    def config(self,player):
        qualify = [0,0,0,0,1]
        if player.pos[1] > 1:
            qualify[0] = 1
        if player.pos[1] < self.size:
            qualify[2] = 1
        if player.pos[0] < self.size:
            qualify[3] = 1
        if player.pos[0] > 1:
            qualify[1] = 1
        if player.bullet_count == 0:
            qualify[4] = 0
        return qualify

    def calcdist(self,player1,player2):
        return math.sqrt((pow(abs(player1.pos[0]-player2.pos[0]),2)+pow(abs(player1.pos[1]-player2.pos[1]),2)))

    def step(self,action1,action2):
        return np.zeros((100,)),5,False,0


    def stepshoot(self,action1,action2):
        actionR = [0,0]
        actionB = [0,0]
        actionR[0] = action1%5
        actionR[1] = action1//5
        actionB[0] = action2%5
        actionB[1] = action2//5
        self.update(actionR,actionB,self.calcdist(self.R_Soldier[0],self.B_Soldier[0]))
        Rstate = 0
        Bstate = 0
        RB = 0
        BB = 0
        if len(self.R_Soldier) > 0:
            Rstate = self.R_Soldier[0].pos[0]*10+self.R_Soldier[0].pos[1]-10
            RB = self.R_Soldier[0].bullet_count
        if len(self.B_Soldier) > 0:
            Bstate = self.B_Soldier[0].pos[0]*10+self.B_Soldier[0].pos[1]-10
            BB = self.B_Soldier[0].bullet_count
        stateR = [Rstate,Bstate,RB,BB]
        stateB = [Bstate,Rstate,BB,RB]
        return stateR,stateB,self.R_r,self.B_r,self.done,[]


#test the env
if __name__ == '__main__':
    Env = env(10)
    for i_eposide in range(1):
        sR,sB = Env.reset()
        print('Ep: ', i_eposide)
        while Env.done == False:
            print(Env.state)
            dis = Env.calcdist(Env.R_Soldier[0],Env.B_Soldier[0])
            aR = Env.R_Soldier[0].act(sR,dis)
            aB = Env.B_Soldier[0].act(sB,dis)
            print(aR,aB)
            sR,sB = Env.stepshoot(aR,aB)











