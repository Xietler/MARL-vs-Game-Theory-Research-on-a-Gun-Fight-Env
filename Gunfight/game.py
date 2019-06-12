import sys,random,pygame,math,threading
from pygame.locals import *
import time
import Shoot
import env
import Soldier

block = 60
gamestate = 0
start = False
pygame.init()
screen = pygame.display.set_mode((800,600))
pygame.display.set_caption("Battle")
font = pygame.font.Font(None,30)
initimage = pygame.image.load("./image/background.jpg")
initimage = pygame.transform.smoothscale(initimage,(800,600))
grass = pygame.image.load("./image/grass.jpg")
grass = pygame.transform.smoothscale(grass,(600,600))
band = pygame.image.load("./image/band.jpg")
band = pygame.transform.smoothscale(band,(200,600))
redgun = pygame.image.load("./image/redgun.png")
redgun = pygame.transform.smoothscale(redgun,(60,60))
redgun = pygame.transform.flip(redgun,True,False)
bluegun = pygame.image.load("./image/bluegun.png")
bluegun = pygame.transform.smoothscale(bluegun,(60,60))
rednogun = pygame.image.load("./image/rednogun.png")
rednogun = pygame.transform.smoothscale(rednogun,(60,60))
rednogun = pygame.transform.flip(rednogun,True,False)
bluenogun = pygame.image.load("./image/bluenogun.png")
bluenogun = pygame.transform.smoothscale(bluenogun,(60,60))

result = 0


def init():
    global start
    global gamestate
    while True:
        screen.blit(initimage, (0, 0))
        for event in pygame.event.get():  # get mouse and key events
            if event.type == MOUSEBUTTONDOWN:
                start = True
        if start:
            gamestate = 1
            break
        pygame.display.update()

def game():
    global gamestate
    global result
    global action1
    Env = env.env(1,10)
    while gamestate == 1:
        while True:


            screen.blit(grass,(0,0))
            screen.blit(band,(600,0))
            #text = font.render("Red Win",10,(0,0,0))
            #screen.blit(text,(600,100))
            if Env.done == False:
                for i in range(Env.size):
                    for j in range(Env.size):
                        count = Env.state[i][j]
                        if( count % 10 == 1 ):
                            if Env.R_Soldier[0].bullet_count > 0:
                                screen.blit(redgun,(block*j,block*i))
                            else:
                                screen.blit(rednogun,(block*j,block*i))
                        if( count/10 == 1 ):
                            if Env.B_Soldier[0].bullet_count > 0:
                                screen.blit(bluegun,(block*j,block*i))
                            else:
                                screen.blit(bluenogun,(block*j,block*i))

            if (Env.gameover() == True):
                gamestate = 2
                if Env.win == 1:
                    result = 1
                elif Env.win == 0:
                    result = 0
                else:
                    result = -1
                break

            textRB = font.render("Red Bullet:" + str(Env.R_Soldier[0].bullet_count), 10, (255, 0, 0))
            screen.blit(textRB, (620, 100))
            textBB = font.render("Blue Bullet:" + str(Env.B_Soldier[0].bullet_count), 10, (0, 0, 255))
            screen.blit(textBB, (620, 150))
            distance = Env.calcdist(Env.R_Soldier[0], Env.B_Soldier[0])
            textdis = font.render("distance:" + str(distance), 10, (0, 0, 0))
            screen.blit(textdis, (620, 200))

            textRR = font.render("Red Acc:"+str(round(Shoot.f1(distance),3)),10,(255,0,0))
            screen.blit(textRR,(620,250))
            textBR = font.render("Blue Acc:"+str(round(Shoot.f2(distance),3)),10,(0,0,255))
            screen.blit(textBR,(620,300))
            if len(Env.R_Soldier) > 0 and len(Env.B_Soldier)>0:
                time.sleep(0.5)
                dis = Env.calcdist(Env.R_Soldier[0], Env.B_Soldier[0])
                action1 = Env.R_Soldier[0].act(dis,20,20)
                action2 = Env.B_Soldier[0].act(dis,21,21)
                Env.stepshoot(action1,action2)

            pygame.display.update()


def over():
    global result
    while True:
        if result == 1:
            text = font.render("Red Win",10,(255,0,0))
            screen.blit(text,(350,300))
        elif result == 0:
            text = font.render("Tie",10,(0,255,0))
            screen.blit(text,(350,300))
        else:
            text = font.render("Blue Win",10,(0,0,255))
            screen.blit(text,(350,300))

        pygame.display.update()

def loop():
    while(True):
        if gamestate == 0:
            init()
        elif gamestate == 1:
            game()
        else:
            over()


if __name__ == '__main__':
    loop()