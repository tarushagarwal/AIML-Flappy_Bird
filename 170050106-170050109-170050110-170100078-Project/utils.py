from itertools import cycle
import variables as v
import pygame
from pygame.locals import *
import random
import genetics
import numpy as np

import time
import nn
from layers import *

import copy
import pickle

def showWelcomeAnimation():
    return {
                'playery': int((v.SCREENHEIGHT - v.IMAGES['player'][0].get_height()) / 2),
                'basex': 0,
                'playerIndexGen': cycle([0, 1, 2, 1]),
            }


def getHitmask(image):
    """returns a hitmask using an image's alpha."""
    mask = []
    for x in range(image.get_width()):
        mask.append([])
        for y in range(image.get_height()):
            # print(image.get_at((x,y))[3])
            mask[x].append(bool(image.get_at((x,y))[3]))
    return mask


def getRandomPipe():
    """returns a randomly generated pipe"""
    # y of gap between upper and lower pipe
    gapY = random.randrange(0, int(v.BASEY * 0.6 - v.PIPEGAPSIZE))
    gapY += int(v.BASEY * 0.2)
    pipeHeight = v.IMAGES['pipe'][0].get_height()
    pipeX = v.SCREENWIDTH + 10

    return [
        {'x': pipeX, 'y': gapY - pipeHeight},  # upper pipe
        {'x': pipeX, 'y': gapY + v.PIPEGAPSIZE}, # lower pipe
    ]

def checkCrash(players, upperPipes, lowerPipes):
    """returns True if player collders with base or pipes."""
    statuses = []
    for idx in range(v.total_models):
        statuses.append(False)

    for idx in range(v.total_models):
        # statuses[idx] = False
        pi = players['index']
        players['w'] = v.IMAGES['player'][0].get_width()
        players['h'] = v.IMAGES['player'][0].get_height()
        # if player crashes into ground
        if players['y'][idx] + players['h'] >= v.BASEY - 1:
            statuses[idx] = True
        playerRect = pygame.Rect(players['x'][idx], players['y'][idx],
                      players['w'], players['h'])
        pipeW = v.IMAGES['pipe'][0].get_width()
        pipeH = v.IMAGES['pipe'][0].get_height()

        for uPipe, lPipe in zip(upperPipes, lowerPipes):
            # upper and lower pipe rects
            uPipeRect = pygame.Rect(uPipe['x'], uPipe['y'], pipeW, pipeH)
            lPipeRect = pygame.Rect(lPipe['x'], lPipe['y'], pipeW, pipeH)

            # player and upper/lower pipe hitmasks
            pHitMask = v.HITMASKS['player'][pi]
            uHitmask = v.HITMASKS['pipe'][0]
            lHitmask = v.HITMASKS['pipe'][1]

            # if bird collided with upipe or lpipe
            uCollide = pixelCollision(playerRect, uPipeRect, pHitMask, uHitmask)
            lCollide = pixelCollision(playerRect, lPipeRect, pHitMask, lHitmask)

            if uCollide or lCollide:
                statuses[idx] = True
    return statuses

def pixelCollision(rect1, rect2, hitmask1, hitmask2):
    """Checks if two objects collide and not just their rects"""
    rect = rect1.clip(rect2)

    if rect.width == 0 or rect.height == 0:
        return False

    x1, y1 = rect.x - rect1.x, rect.y - rect1.y
    x2, y2 = rect.x - rect2.x, rect.y - rect2.y

    for x in range(rect.width):
        for y in range(rect.height):
            if hitmask1[x1+x][y1+y] and hitmask2[x2+x][y2+y]:
                return True
    return False


def showScore(score):
    """displays score in center of screen"""
    scoreDigits = [int(x) for x in list(str(score))]
    totalWidth = 0 # total width of all numbers to be printed

    for digit in scoreDigits:
        totalWidth += v.IMAGES['numbers'][digit].get_width()

    Xoffset = (v.SCREENWIDTH - totalWidth) / 2

    for digit in scoreDigits:
        v.SCREEN.blit(v.IMAGES['numbers'][digit], (Xoffset, v.SCREENHEIGHT * 0.1))
        Xoffset += v.IMAGES['numbers'][digit].get_width()



def showGameOverScreen(crashInfo):
    """Perform genetic updates here"""
    new_weights = []
    
    ind = [x for _, x in sorted(zip(v.fitness, range(len(v.fitness))), reverse = True)]
    # print(ind, v.fitness)
    k = int(0.4*v.total_models)
    ind = ind[:k]
    for idx in range(k):
        new_weights.append(genetics.model_mutate(v.current_pool[ind[0]].getweights()))

    for _ in range(int(0.2*v.total_models)):
        idxA = random.randint(0,k-1)
        new_weights.append(genetics.model_mutate(v.current_pool[ind[idxA]].getweights()))

    for i in range(int(0.1*v.total_models)): 
        updated_weights1 = genetics.model_crossover(ind[i], ind[i+1])
        new_weights.append(genetics.model_mutate(updated_weights1[0]))
    for _ in range(int(0.3*v.total_models)):
        idxA = random.randint(0,k-1)
        idxB = random.randint(0,k-1)
        updated_weights1 = genetics.model_crossover(ind[idxA], ind[idxB])
        new_weights.append(genetics.model_mutate(updated_weights1[0]))

    for select in range(len(new_weights)):
        v.fitness[select] = -100
        v.current_pool[select].setweight(new_weights[select])
    if v.save_current_pool == 1:
        save_pool()
    v.generation = v.generation + 1
    return

def save_pool():
    for xi in range(v.total_models):
        to_save = v.current_pool[xi].getweights()
        file_name = "Trained_model/" + str(xi)+".pkl"
        with open(file_name, "wb") as f:
            pickle.dump(to_save, f)
    print("Saved current pool!")





