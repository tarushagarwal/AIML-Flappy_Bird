import numpy as np
import random
import time
import nn
from layers import *
import pickle
import copy
import variables as v

def init():
	for i in range(v.total_models):
		out_nodes=1
		alpha=0.01
		batchSize=1
		epochs=1
		nn1 = nn.NeuralNetwork(out_nodes, alpha, batchSize, epochs) 
		nn1.addLayer(FullyConnectedLayer(3,7,'sigmoid'))
		nn1.addLayer(FullyConnectedLayer(7,1,'sigmoid'))
		v.current_pool.append(nn1)
		v.fitness.append(-100)

	if v.load_saved_pool:
		for i in range(v.total_models):
			file_name = "Trained_model/" + str(i) + ".pkl"
			with open(file_name, "rb") as f:
				to_load = pickle.load(f)
				v.current_pool[i].setweight(to_load)
	if v.load_best:
		for i in range(v.total_models):
			file_name = "Trained_model_best/" + str(i) + ".pkl"
			with open(file_name, "rb") as f:
				to_load = pickle.load(f)
				v.current_pool[i].setweight(to_load)


def model_crossover(model_idx1, model_idx2):
	# global current_pool
	weights1 = v.current_pool[model_idx1].getweights()
	weights2 = v.current_pool[model_idx2].getweights()
	weightsnew1 = copy.deepcopy(weights1)
	weightsnew2 = copy.deepcopy(weights2)
	weightsnew1[0] = copy.deepcopy(weights2[0])
	weightsnew2[0] = copy.deepcopy(weights1[0])
	return np.asarray([weightsnew1, weightsnew2])

def model_mutate(weights):
	for xi in range(len(weights)):
		for yi in range(len(weights[xi])):
			if random.uniform(0, 1) > 0.85:
				change = random.uniform(-0.5,0.5)
				weights[xi][yi] += change
	return weights

def predict_action(height, dist, pipe_height, model_num):
	# global current_pool
	# The height, dist and pipe_height must be between 0 to 1 (Scaled by SCREENHEIGHT)
	height = min(v.SCREENHEIGHT, height) / v.SCREENHEIGHT - 0.5
	dist = dist / 450 - 0.5 # Max pipe distance from player will be 450
	pipe_height = min(v.SCREENHEIGHT, pipe_height) / v.SCREENHEIGHT - 0.5
	neural_input = np.asarray([height, dist, pipe_height])
	neural_input = np.atleast_2d(neural_input)
	output_prob = v.current_pool[model_num].feedforward(neural_input)[-1][0]
	if output_prob[0] <= 0.5:
		# Perform the jump action
		return 1
	return 2

