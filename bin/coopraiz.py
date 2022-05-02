from base import *
from wjson import *


CPATH = f"{P}container/"


# --------------------------------------------------------------------------- #
# Compute Complimentarity matrix 
# --------------------------------------------------------------------------- #

def get_compl_matrix_new(l1, l2, fname):
	ncells1, ncells2 = l1.shape[0], l2.shape[0]
	
	cmatrix = np.zeros((ncells1*ncells2, ncells1*ncells2))

	for i1 in range(ncells1):
		for i2 in range(ncells2):
			# this is the first edge i1---i2:
			e1 = edge_id_given_cells(i1, i2, ncells2)
			
			for j1 in range(i1 + 1, ncells1):
				if l1[i1,j1] == 0: continue
				
				for j2 in range(ncells2):
					if j2 == i2: continue
					if l2[i2,j2] == 0: continue
					
					# this is the second edge j1---j2:
					e2 = edge_id_given_cells(j1, j2, ncells2)
					
					cmatrix[e1,e2] = np.sqrt(l1[i1,j1]*l2[i2,j2])
					cmatrix[e2,e1] = cmatrix[e1,e2]


	# saving to disk
	np.save(fname, cmatrix)
	
	return cmatrix




# --------------------------------------------------------------------------- #
# Core functions
# --------------------------------------------------------------------------- #

# given cell ids i and j return the id of the corresponding edge
def edge_id_given_cells(i, j, ncells2):
	return (i*ncells2 + j)

# given edge id return the cells' ids
def cell_ids_given_edge(e, ncells2):
	i = int(e/ncells2)
	j = e%ncells2
	return (i,j)





# --------------------------------------------------------------------------- #
# Basic Diffusion
# --------------------------------------------------------------------------- #

def do_diffusion(a, gamma):
	ncells = a.shape[0]
	
	# no self edges
	for i in range(ncells): 
		a[i][i] = 0
	
	# define S
	s = np.zeros((ncells, ncells))
	for i in range(ncells): 
		s[i][i] = sum(a[i,:])
	
	# compute L
	l = np.subtract(a,s)
	l = np.subtract(l,gamma*np.identity(ncells))
	l = -1*l
	l = np.linalg.inv(l)
	l = np.transpose(l)

	
	# post processing
	for i in range(ncells): 
		l[i][i] = 0

	for i,j in it.combinations(list(range(ncells)), 2): 
		v = (l[i][j] + l[j][i])/2
		l[i][j] = v; l[j][i] = v
	
	l = np.divide(l,l.max())

	return l



# --------------------------------------------------------------------------- #


# remove the edges whose weight is in the bottom perc% to make the graph not full
def remove_lowest_edges(a, perc):
	# avoid removing potentially all edges if we have only a 2 or 3 of nodes
	if a.shape[0] < 4: return a
	
	# find the threshold value
	vals = a.reshape(-1,).tolist()
	vals.sort()
	val = vals[int(perc*len(vals))]

	b = np.zeros((a.shape[0], a.shape[1]))
	for (i,j) in it.product(list(range(a.shape[0])), list(range(a.shape[1]))):
		b[i][j] = (a[i][j] if a[i][j] > val else 0)
	return b


# makes all entries in matrix l that are in the bottom perc% zero
def reduce_L_values(l, perc):
	vals, ncells = [], l.shape[0]
	for i,j in it.combinations(list(range(ncells)), 2):
		vals.append(l[i][j])
	vals.sort()
	x = vals[int(perc*len(vals))]
	
	for i,j in it.product(list(range(ncells)), repeat=2):
		if l[i][j] < x:
			l[i][j] = 0
	

def handle_corner_case_two_cells(ncells1, ncells2):
	m = {}; m[0] = 0; 
	if ncells1 == 1 or ncells2 == 1: return m
	m[1] = 1
	return m





# --------------------------------------------------------------------------- #
# Parses coopraiz outputs to our format
# --------------------------------------------------------------------------- #


def parse_coopraiz_output(cfile, fname, ncells2):
	fo = open(fname, 'w')
	
	for line in open(cfile):
		edges = [cell_ids_given_edge(int(w), ncells2) for w in line.rstrip().split(",")]
		edges.sort(key=lambda x: x[0])
		[fo.write(str(e[0]) + "\t" + str(e[1]) + "\n") for e in edges]
		
		fo.close()
		return parse_matching(fname)
		
		
def parse_matching(fname):
	match = {}
	for line in open(fname):
		words = line.rstrip().split("\t")
		match[int(words[0])] = int(words[1])

	return match


def reverse_match(match):
	rev_match = {}
	for i in match:
		rev_match[match[i]] = i
	return rev_match


def write_match(match, fname):	
	fo = open(fname, "w")
	for i in range(len(match)):
		fo.write(f"{i}\t{match[i]}\n")
	fo.close()





# --------------------------------------------------------------------------- #
# Run Coopraiz using the Docker container
# --------------------------------------------------------------------------- #

def run_coopraiz_exe_container(cname, fname, jname):
	# load the docker container
	cmd_docker = f"docker load < {P}container/coopraiz-generic/coopraiz-generic-docker-image.tar.gz"
	os.system(cmd_docker)
	
	# now run coopraiz with the complimentarity matrix and json file we created 
	cmd_base = f"docker run --rm -t -a stdout -a stderr -v {P}container/:/container/ coopraiz-generic"

	cmd = f"{cmd_base} -imjson /container/jsons2/{jname} -ssdfilename /container/compl_matrices/{cname}.npy -outasciiresultsfilename /container/outputs/{fname}.txt  -partialenumeration 1 -ssdnonormalize T"
	
	os.system(cmd)




# --------------------------------------------------------------------------- #
# Entry function
# --------------------------------------------------------------------------- #

def entry_run_Coopraiz(a1, a2, bname):
	# 0. Preliminaries,  Form the output names
	frac, gamma = 0.5, 0.5
	ncells1, ncells2 = a1.shape[0], a2.shape[0]
	
	
	if ncells1 < 3 or ncells2 < 3:
		return handle_corner_case_two_cells(ncells1, ncells2)
	
	fname = f"{bname}_n{ncells1}_{ncells2}"
	cname = f"{fname}"
	jname = f"{ncells1}_{ncells2}.json"
	
	
	
	# 1. Compute distances and do Diffusion on each set
	d1 = compute_distance_between_cells(a1, "cosine")
	d2 = compute_distance_between_cells(a2, "cosine")

	l1 = do_diffusion(remove_lowest_edges(get_similiarity_from_distance(d1), frac), gamma)	
	l2 = do_diffusion(remove_lowest_edges(get_similiarity_from_distance(d2), frac), gamma)
	
	reduce_L_values(l1, 0.3)
	reduce_L_values(l2, 0.3)


	# 2. Compute the complementarity matrix and write json
	cmatrix = get_compl_matrix_new(l1, l2, f"{CPATH}compl_matrices/{cname}")
	writeJson(ncells1, ncells2, f"{CPATH}jsons2/{ncells1}_{ncells2}.json")
	
	
	# 3. Run Coopraiz
	run_coopraiz_exe_container(cname, fname, jname)

	
	# 5. Parse the matching
	match = parse_coopraiz_output(f"{CPATH}outputs/{fname}.txt", f"{CPATH}matches/{fname}.txt", ncells2)

	
	return match



def entry_coopraiz_cluster_matching(l1, l2, bname):
	# 0. Form the appropriate output names
	ncells1, ncells2 = l1.shape[0], l2.shape[0]
	
	fname = f"{bname}_n{ncells1}_{ncells2}"
	cname = f"{fname}"
	jname = f"{ncells1}_{ncells2}.json"
	
	
	# 1. Compute the complementarity matrix and write json
	cmatrix = get_compl_matrix_new(l1, l2, f"{CPATH}compl_matrices/{fname}")
	writeJson(ncells1, ncells2, f"{CPATH}jsons2/{ncells1}_{ncells2}.json")
	
	
	# 2. Run Coopraiz
	run_coopraiz_exe_container(cname, fname, jname)


	# 3. Parse the matching
	match = parse_coopraiz_output(f"{CPATH}outputs/{fname}.txt", f"{CPATH}matches/{fname}.txt", ncells2)

	
	return match




# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
	pass

	