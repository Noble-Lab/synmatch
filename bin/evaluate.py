from base import *
from coopraiz import *


# --------------------------------------------------------------------------- #
# Neighborhood Overlap/FOSCTTM score
# --------------------------------------------------------------------------- #

def neighborhood_overlap_Coopraiz(match, match_reverse, d1, d2):
	h1, h2, answer = {}, {}, {}
	
	# 1. Do it first for each cell i in data1
	for i in cells:
		# sort the cells in data2 by how far they are from the cell in data2 that i has been matched
		h = [(j, d2[match[i]][j]) for j in cells]
		h.sort(key=lambda x: x[1])
		
		# how far do we need to go to find the correct match of cell i
		for d in cells:
			if h[d][0] == i:
				h1[i] = d
				
	# 2. Now do it first for each cell i in data2
	for i in cells:
		# sort the cells in data1 by how far they are from the cell in data1 that i has been matched
		h = [(j, d1[match_reverse[i]][j]) for j in cells]
		h.sort(key=lambda x: x[1])
		
		# how far do we need to go to find the correct match of cell i
		for d in cells:
			if h[d][0] == i:
				h2[i] = d
	
	
	# 3. Compute the percentage as you increase the distance d
	for d in cells:
		x1 = sum([h1[j] <= d for j in h1])
		x2 = sum([h2[j] <= d for j in h2])
		answer[d] = float(x1 + x2)/(len(h1) + len(h2))
	
	
	# 4. Also compute FOSCTTM- the fraction of cells that are closer than the true match
	foscttm = np.average([float(h1[i] + h2[i])/(2*(ncells-1)) for i in cells])
	

	return (answer, foscttm)




# --------------------------------------------------------------------------- #
# Label transfer accuracy
# --------------------------------------------------------------------------- #

# use side1 to predict labels on side2
def label_accuracy_oneside(domain1, domain2, type1, type2):
	knn = KNeighborsClassifier(n_neighbors=5)
	knn.fit(domain2, type2)
	type1_predict = knn.predict(domain1)
	count = 0.0
	for label1, label2 in zip(type1_predict, type1):
		if label1 == label2:
			count += 1
	la = count / len(type1)

	return la


def label_accuracy(emb1, emb2, type1, type2):
	l1 = label_accuracy_oneside(emb1, emb2, type1, type2)
	l2 = label_accuracy_oneside(emb2, emb1, type1, type2)
	
	return (l1 + l2)/2


def label_accuracy_Coopraiz(a1, a2, match, match_reverse, cell_labels):
	emb1into2 = a1[np.ix_(match,list(range(a1.shape[1])))]
	emb2into1 = a2[np.ix_(match_reverse,list(range(a2.shape[1])))]

	l1 = label_accuracy_oneside(a1, emb1into2, cell_labels, cell_labels)
	l2 = label_accuracy_oneside(emb2into1, a2, cell_labels, cell_labels)
	
	return (l1 + l2)/2


# --------------------------------------------------------------------------- #
# basic evaluation
# --------------------------------------------------------------------------- #

def read_basic_match_for_eval(bname):
	match = parse_matching(bname)
	rev_match = reverse_match(match)
	return [match[i] for i in range(len(match))], [rev_match[i] for i in range(len(rev_match))]
	

def eval_basic(a1_file, a2_file, match_file):
	a1 = np.load(a1_file)
	a2 = np.load(a2_file)

	d1 = compute_distance_between_cells(a1, "cosine")
	d2 = compute_distance_between_cells(a2, "cosine")

	match, rev_match = read_basic_match_for_eval(match_file)
	
	nv, f = neighborhood_overlap_Coopraiz(match, rev_match, d1, d2)
	
	print(f"foscttm score = {f}")



# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
	eval_basic(sys.argv[1], sys.argv[2], sys.argv[3])
	

