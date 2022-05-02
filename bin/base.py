# --------------------------------------------------------------------------- #
# Libraries we are going to use
# --------------------------------------------------------------------------- #

import numpy as np
import os
import sys
import math
import random
import time
import glob
import itertools as it
import multiprocessing as mp

from scipy.spatial import distance
from scipy import stats

import sklearn as sk
import sklearn.cluster as skclust
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler, normalize


# --------------------------------------------------------------------------- #
# Set up Paths
# --------------------------------------------------------------------------- #

P = f"{os.path.dirname(os.path.abspath(__file__))}/../"
T = f"{P}Temp/"
D = f"{P}data/"
C = f"{P}container/"



# --------------------------------------------------------------------------- #
# Compute distances
# --------------------------------------------------------------------------- #

def compute_distance_between_cells(a, dist):
	ncells = a.shape[0]
	d = np.zeros((ncells,ncells))
	
	for i in range(ncells):
		for j in range(ncells):
			
			if dist == "cosine":
				d[i][j] = distance.cosine(a[i],a[j])
			elif dist == "euclidean":
				d[i][j] = distance.euclidean(a[i],a[j])
			else:
				print(f"don't know yet distance {dist}")
				sys.exit()

	return d




# --------------------------------------------------------------------------- #
# Conversion of Distance to Similiarity and vice versa
# --------------------------------------------------------------------------- #

# currently defined as s = 1 - d_norm
def get_similiarity_from_distance(d):
	ncells = d.shape[0]
	dnorm = d/d.max()
	return np.subtract(np.ones((ncells, ncells)), dnorm)


def convert_diffusion_to_distance(l):
	ncells = l.shape[0]
	
	# now convert to distance on a per-row basis
	ld = np.zeros((ncells, ncells))
	for i in range(ncells):
		m = l[i,:].max()
		ld[i,:] = l[i,:]/m
	ld = np.subtract(np.ones((ncells, ncells)),ld)
	
	return ld
	

def zero_diagonal(a):
	for i in range(len(a)):
		a[i][i] = 0
	return a



# --------------------------------------------------------------------------- #
# reverse matches
# --------------------------------------------------------------------------- #

def parse_matching_base(fname):
	match = {}
	for line in open(fname):
		words = line.rstrip().split("\t")
		match[int(words[0])] = int(words[1])

	return match
	
def reverse_match_base(match):
	rev_match = {}
	for i in match:
		rev_match[match[i]] = i
	return rev_match


def write_match_base(match, fname):	
	fo = open(fname, "w")
	for i in range(len(match)):
		fo.write(f"{i}\t{match[i]}\n")
	fo.close()
	




