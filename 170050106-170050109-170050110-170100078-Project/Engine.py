import utils
import pygame
from pygame.locals import *
import genetics
import variables as v
import sys
import time
import numpy as np
import nn
from layers import *

import copy

def ret_Loader(basex, upperPipes, lowerPipes, score):
    return {
        'y': 0,
        'groundCrash': True,
        'basex': basex,
        'upperPipes': upperPipes,
        'lowerPipes': lowerPipes,
        'score': score,
        'playerVelY': 0,
    }

def mainGame(movementInfo):
    score = playerIndex = loopIter = 0
    playerIndexGen = movementInfo['playerIndexGen']
    playersXList = []
    playersYList = []
    for idx in range(v.total_models):
        playerx, playery = int(v.SCREENWIDTH * 0.2), movementInfo['playery']
        playersXList.append(playerx)
        playersYList.append(playery)
    basex = movementInfo['basex']
    baseShift = v.IMAGES['base'].get_width() - v.IMAGES['background'].get_width()

    # get 2 new pipes to add to upperPipes lowerPipes list
    newPipe1 = utils.getRandomPipe()
    newPipe2 = utils.getRandomPipe()

    # list of upper pipes
    upperPipes = [
        {'x': v.SCREENWIDTH + 200, 'y': newPipe1[0]['y']},
        {'x': v.SCREENWIDTH + 200 + (v.SCREENWIDTH / 2), 'y': newPipe2[0]['y']},
    ]

    # list of lowerPipes
    lowerPipes = [
        {'x': v.SCREENWIDTH + 200, 'y': newPipe1[1]['y']},
        {'x': v.SCREENWIDTH + 200 + (v.SCREENWIDTH / 2), 'y': newPipe2[1]['y']},
    ]



    v.next_pipe_x = lowerPipes[0]['x']
    v.next_pipe_hole_y = (lowerPipes[0]['y'] + (upperPipes[0]['y'] + v.IMAGES['pipe'][0].get_height()))/2

    pipeVelX = -4

    # player velocity, max velocity, downward accleration, accleration on flap
    playersVelY    =  []   # player's velocity along Y, default same as playerFlapped
    playerMaxVelY =  10   # max vel along Y, max descend speed
    playerMinVelY =  -8   # min vel along Y, max ascend speed
    playersAccY    =  []   # players downward accleration
    playerFlapAcc =  -9   # players speed on flapping
    playersFlapped = [] # True when player flaps
    playersState = []

    for idx in range(v.total_models):
        playersVelY.append(-9)
        playersAccY.append(1)
        playersFlapped.append(False)
        playersState.append(True)

    alive_players = v.total_models


    while True:

        for event in pygame.event.get():
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                pygame.quit()
                sys.exit()

        for idxPlayer in range(v.total_models):
            if playersYList[idxPlayer] < 0 and playersState[idxPlayer] == True:
                alive_players -= 1
                playersState[idxPlayer] = False
        if alive_players == 0:
            return ret_Loader(basex, upperPipes, lowerPipes, score)


        v.next_pipe_x += pipeVelX
        for idxPlayer in range(v.total_models):
            if playersState[idxPlayer] == True:
                v.fitness[idxPlayer] += 1
                if genetics.predict_action(playersYList[idxPlayer], v.next_pipe_x, v.next_pipe_hole_y, idxPlayer) == 1:
                    if playersYList[idxPlayer] > -2 * v.IMAGES['player'][0].get_height():
                        playersVelY[idxPlayer] = playerFlapAcc
                        playersFlapped[idxPlayer] = True
                        #SOUNDS['wing'].play()
        



        # check for crash here, returns status list
        crashTest = utils.checkCrash({'x': playersXList, 'y': playersYList, 'index': playerIndex},
                               upperPipes, lowerPipes)

        for idx in range(v.total_models):
            if playersState[idx] == True and crashTest[idx] == True:
                alive_players -= 1
                # v.fitness[idx] -= ( - playersXList[idx])
                playersState[idx] = False
        if alive_players == 0:
            return ret_Loader(basex, upperPipes, lowerPipes, score)

        # check for score
        flag = 0
        for idx in range(v.total_models):
            if playersState[idx] == True:
                pipe_idx = 0
                playerMidPos = playersXList[idx]
                for pipe in upperPipes:
                    pipeMidPos = pipe['x'] + v.IMAGES['pipe'][0].get_width()
                    if pipeMidPos <= playerMidPos < pipeMidPos + 4:
                        v.next_pipe_x = lowerPipes[pipe_idx+1]['x']
                        v.next_pipe_hole_y = (lowerPipes[pipe_idx+1]['y'] + (upperPipes[pipe_idx+1]['y'] + v.IMAGES['pipe'][pipe_idx+1].get_height())) / 2
                        if(flag==0):
                            score += 1
                            flag = 1
                        v.fitness[idx] += 25
                        # SOUNDS['point'].play()
                    pipe_idx += 1
                    
        # playerIndex basex change
        if (loopIter + 1) % 3 == 0:
            playerIndex = next(playerIndexGen)
        loopIter = (loopIter + 1) % 30
        basex = -((-basex + 100) % baseShift)

        # player's movement
        for idx in range(v.total_models):
            if playersState[idx] == True:
                if playersVelY[idx] < playerMaxVelY and not playersFlapped[idx]:
                    playersVelY[idx] += playersAccY[idx]
                if playersFlapped[idx]:
                    playersFlapped[idx] = False
                playerHeight = v.IMAGES['player'][playerIndex].get_height()
                playersYList[idx] += min(playersVelY[idx], v.BASEY - playersYList[idx] - playerHeight)

        # move pipes to left
        for uPipe, lPipe in zip(upperPipes, lowerPipes):
            uPipe['x'] += pipeVelX
            lPipe['x'] += pipeVelX

        # add new pipe when first pipe is about to touch left of screen
        if 0 < upperPipes[0]['x'] < 5:
            newPipe = utils.getRandomPipe()
            upperPipes.append(newPipe[0])
            lowerPipes.append(newPipe[1])

        # remove first pipe if its out of the screen
        if upperPipes[0]['x'] < -v.IMAGES['pipe'][0].get_width():
            upperPipes.pop(0)
            lowerPipes.pop(0)

        # draw sprites
        v.SCREEN.blit(v.IMAGES['background'], (0,0))

        for uPipe, lPipe in zip(upperPipes, lowerPipes):
            v.SCREEN.blit(v.IMAGES['pipe'][0], (uPipe['x'], uPipe['y']))
            v.SCREEN.blit(v.IMAGES['pipe'][1], (lPipe['x'], lPipe['y']))

        v.SCREEN.blit(v.IMAGES['base'], (basex, v.BASEY))
        # print score so player overlaps the score
        utils.showScore(score)
        for idx in range(v.total_models):
            if playersState[idx] == True:
                v.SCREEN.blit(v.IMAGES['player'][playerIndex], (playersXList[idx], playersYList[idx]))

        pygame.display.update()
        v.FPSCLOCK.tick(v.FPS)

