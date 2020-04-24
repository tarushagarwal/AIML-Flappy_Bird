import random
import sys
import utils 
import pygame
from pygame.locals import *
import variables as v
import Engine
import genetics
import time
import nn
from layers import *

import copy


def main():
    pygame.init()
    v.FPSCLOCK = pygame.time.Clock()
    v.SCREEN = pygame.display.set_mode((int(v.SCREENWIDTH), int(v.SCREENHEIGHT)))
    pygame.display.set_caption('Flappy Bird')

    # numbers sprites for score display
    v.IMAGES['numbers'] = (
        pygame.image.load('assets/sprites/0.png').convert_alpha(),
        pygame.image.load('assets/sprites/1.png').convert_alpha(),
        pygame.image.load('assets/sprites/2.png').convert_alpha(),
        pygame.image.load('assets/sprites/3.png').convert_alpha(),
        pygame.image.load('assets/sprites/4.png').convert_alpha(),
        pygame.image.load('assets/sprites/5.png').convert_alpha(),
        pygame.image.load('assets/sprites/6.png').convert_alpha(),
        pygame.image.load('assets/sprites/7.png').convert_alpha(),
        pygame.image.load('assets/sprites/8.png').convert_alpha(),
        pygame.image.load('assets/sprites/9.png').convert_alpha()
    )

    # game over sprite
    v.IMAGES['gameover'] = pygame.image.load('assets/sprites/gameover.png').convert_alpha()
    # message sprite for welcome screen
    v.IMAGES['message'] = pygame.image.load('assets/sprites/message.png').convert_alpha()
    # base (ground) sprite
    v.IMAGES['base'] = pygame.image.load('assets/sprites/base.png').convert_alpha()

    while True:
        # select random background sprites
        randBg = random.randint(0, len(v.BACKGROUNDS_LIST) - 1)
        v.IMAGES['background'] = pygame.image.load(v.BACKGROUNDS_LIST[randBg]).convert()

        # select random player sprites
        randPlayer = random.randint(0, len(v.PLAYERS_LIST) - 1)
        v.IMAGES['player'] = (
            pygame.image.load(v.PLAYERS_LIST[randPlayer][0]).convert_alpha(),
            pygame.image.load(v.PLAYERS_LIST[randPlayer][1]).convert_alpha(),
            pygame.image.load(v.PLAYERS_LIST[randPlayer][2]).convert_alpha(),
        )

        # select random pipe sprites
        pipeindex = random.randint(0, len(v.PIPES_LIST) - 1)
        v.IMAGES['pipe'] = (
            pygame.transform.rotate(
                pygame.image.load(v.PIPES_LIST[pipeindex]).convert_alpha(), 180),
            pygame.image.load(v.PIPES_LIST[pipeindex]).convert_alpha(),
        )

        # hismask for pipes
        v.HITMASKS['pipe'] = (
            utils.getHitmask(v.IMAGES['pipe'][0]),
            utils.getHitmask(v.IMAGES['pipe'][1]),
        )

        # hitmask for player
        v.HITMASKS['player'] = (
            utils.getHitmask(v.IMAGES['player'][0]),
            utils.getHitmask(v.IMAGES['player'][1]),
            utils.getHitmask(v.IMAGES['player'][2]),
        )

        movementInfo = utils.showWelcomeAnimation()
        # global fitness
        for idx in range(v.total_models):
            v.fitness[idx] = 0
        crashInfo = Engine.mainGame(movementInfo)
        utils.showGameOverScreen(crashInfo)


if __name__ == '__main__':
	# Initialize all models
	genetics.init()
	main()
