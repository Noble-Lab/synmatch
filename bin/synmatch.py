from base import *
from coopraiz import *


# --------------------------------------------------------------------------- #
# Core Clustering Functions 
# --------------------------------------------------------------------------- #


# creates a hash of arrays with cells cl[cluster_number] = [cell4, cell7, cell23, ...]
def create_clusters_from_labels(labels):
	cl = {}
	for i in range(len(labels)):
		if not labels[i] in cl: cl[labels[i]] = []
		cl[labels[i]].append(i)
		
	return cl


# write down the clusters
def write_cluster_down(clusters, fname):
	fo = open(fname, "w")
	
	for cl in list(range(len(clusters))):
		for x in clusters[cl]:
			fo.write(f"{cl}\t{x}\n")

	
# read the clusters we had previously computed
def read_clusters(fname):
	clusters = {}
	for line in open(fname):
		words = line.rstrip().split("\t")
		cl = int(words[0])
		x  = int(words[1])
		if cl not in clusters:
			clusters[cl] = []
		clusters[cl].append(x)
		
	return clusters


# --------------------------------------------------------------------------- #
# Match the clusters via centroids
# --------------------------------------------------------------------------- #

# match the clusters using their centroids and with a simple call to Coopraiz 
def match_clusters_via_centroids(a1, a2, clusters1, clusters2, bname):
	centroids1 = compute_centroids(a1, clusters1)
	centroids2 = compute_centroids(a2, clusters2)

	match = entry_run_Coopraiz(centroids1, centroids2, f"{bname}_cluster_match_centroid")
	
	return match


# given cluster assignemnts of the cells compute the centroids of the meta-cells
def compute_centroids(a, clusters):
	centroids = np.zeros((len(clusters), a.shape[1]))
	
	for cl in list(range(len(clusters))):
		centroid_cl = np.mean(a[clusters[cl],:], axis=0)
		for i in list(range(a.shape[1])):
			centroids[cl][i] = centroid_cl[i]
			
	return centroids


# --------------------------------------------------------------------------- #
# Match the clusters via DUDV
# --------------------------------------------------------------------------- #


# match the clusters using Average Diffusion Similarty 
def match_clusters_via_DUDV(a1, a2, clusters1, clusters2, bname):
	l1 = do_diffusion_on_dataset(a1)
	l2 = do_diffusion_on_dataset(a2)
	
	du_clust1 = compute_DUDV_between_clusters(a1, l1, clusters1)
	du_clust2 = compute_DUDV_between_clusters(a2, l2, clusters2)
	
	match = entry_coopraiz_cluster_matching(du_clust1, du_clust2, f"{bname}_cluster_match_dudv")
	
	return match


# computes the average diffusion similarity between each pair of clusters in the given modality
def compute_DUDV_between_clusters(a, l, clusters):
	n_clust = len(clusters)
	du_clust = np.zeros((n_clust, n_clust))
	
	# 1. calculate diffusion similarity between each pair of clusters in the modality
	for cl1 in list(range(n_clust)):
		for cl2 in list(range(cl1, n_clust)):
			du_cells = []
			for cell1 in clusters[cl1]:
				for cell2 in clusters[cl2]:
					du_cells.append((l[cell1][cell2] + l[cell2][cell1])/2)
			du_clust[cl1][cl2] = np.mean(du_cells)
			#du_clust[cl1][cl2] = np.max(du_cells)
			
			du_clust[cl2][cl1] = du_clust[cl1][cl2]

	return du_clust


# Do basic diffusion on the given dataset. Move to hatch.py
def do_diffusion_on_dataset(a, frac = 0.5, gamma = 0.5):
	#  Compute distances and do Diffusion
	d = compute_distance_between_cells(a, "cosine")
	l = do_diffusion(remove_lowest_edges(get_similiarity_from_distance(d), frac), gamma)	
	reduce_L_values(l, 0.3)
	
	return l




# --------------------------------------------------------------------------- #
# Equal size KMenas
# --------------------------------------------------------------------------- #

def entry_EQKM(a, n_points_per_clust):
	
	random.seed(10)
	
	# Entry: how many clusters to use
	# --------------------------------------------------------------------------- #
	
	#global n_points, n_clust
	n_points = a.shape[0]
	n_clust = math.ceil(float(n_points)/n_points_per_clust)
	h_assign, h_clust = {}, [0]*n_clust


	#print(f"Will run EQKM with n_clust={n_clust}, n_points_per_clust={n_points_per_clust}\n")
	
	
	# Initialzie the centroids via KMenas++
	# --------------------------------------------------------------------------- #
	start_time = time.time()
	centroids = initialize_via_KMenas_plus(a, n_clust)

	
	# Initial assignment of points to clusters	
	# --------------------------------------------------------------------------- #
	start_time = time.time()
	avail_centroids = list(range(n_clust))
	avail_points = list(range(n_points))
	
	while len(avail_points) > 0:
		# 1. calculate the value (min_dist - max_dist) for every unassigned point
		distances = []
		for i in avail_points:
				dist = [distance.euclidean(a[i], centroids[c]) for c in avail_centroids]
				distances.append((i, (min(dist) - max(dist)), dist.index(min(dist))))
	
		# 2. sort the points by this value
		distances.sort(key=lambda x: float(x[1]), reverse=False)
	
		# 3. assign the points in the given order to their nearest cluster until a cluster gets full
		for d in distances:
			point_id = d[0]
			centroid_id = avail_centroids[d[2]]
		
			h_assign[point_id] = centroid_id
			avail_points.remove(point_id)
		
			h_clust[centroid_id] += 1
		
			if h_clust[centroid_id] == n_points_per_clust:
				avail_centroids.remove(centroid_id)
				break

	
	# Iteration: Main Loop
	# --------------------------------------------------------------------------- #
	
	for num_iter in range(200):
		
		start_time = time.time()
			
		# 1. compute current cluster means (the new centroids)
		centroids = []
		for i in range(n_clust):
			arr = [a[j] for j in range(n_points) if (h_assign[j] == i)]
		
			centroids.append(np.mean(arr, axis=0))
		

		# 2. compute distances from each point to the centroids (delta = current - best possible)
		distances = []
		for i in list(range(n_points)):
				dist = [distance.euclidean(a[i], centroids[c]) for c in list(range(n_clust))]
				distances.append((i, (dist[h_assign[i]] - min(dist)), dist, np.argsort(dist)))
	
		
		# 3. sort the points by value (delta = current cluster assignment - best possible)
		distances.sort(key=lambda x: float(x[1]), reverse=True)
	
	
	
		# keep track of which points want to change and which have moved
		want_to_change, already_moved = {}, []
		for i in range(n_clust):
			want_to_change[i] = []


		# 4. Try every point that may want to leave based on the value we sorted them by
		for d in distances:
			current_point = d[0]
			current_clust = h_assign[current_point]
			
			# if the point has moved, don't use it and skip it
			if current_point in already_moved: continue
				
			# the point wants to leave only if value > 0. DO WE WANT THIS???
			if d[1] <= 0: continue
			
			
			# try the other clusters based on how far they are
			for new_clust in d[3]:
				# if the point has moved, don't use it and skip it
				if current_point in already_moved: continue
		
				# if a given cluster has a point that wants to leave
				if len(want_to_change[new_clust]) > 0:
			
					# try every such point
					for other_point in want_to_change[new_clust]:
						
						# if the point has moved, don't use it and skip it
						if other_point in already_moved:
							continue
					
						# if the swap yields an improvement
						if (distances[other_point][2][current_clust] + distances[current_point][2][new_clust]) < (distances[other_point][2][new_clust] + distances[current_point][2][current_clust]):

							# do the swap
							h_assign[other_point] = current_clust
							h_assign[current_point] = new_clust
							already_moved.append(current_point)
							already_moved.append(other_point)

							break

				# if there is room in the other cluster then move it directly
				elif h_clust[new_clust] < n_points_per_clust:
					h_clust[current_clust] -= 1
					h_clust[new_clust] += 1
					h_assign[current_point] = new_clust
					already_moved.append(current_point)
					
		
			# if the point hasn't moved, add it to the lsit of willing to move
			if current_point not in already_moved:
				want_to_change[current_clust].append(current_point)
		
		
		# stop condition: no point has moved
		if len(already_moved) < 1:
			#print(f"no movement, so stop iterating\n")
			break

	
	return [h_assign[i] for i in range(n_points)]
	


# KMeans++ initialization
def initialize_via_KMenas_plus(a, n_clust):
	centroids = []

	# choose for the first centroid a data point at random
	centroids.append(a[random.randrange(0, a.shape[0])])
	
	# pick the remaining probabilistically based on their shortest distance to existing centroids
	for nt in range(n_clust - 1):
		min_dist = []
		for point in a:
			min_dist.append(min([distance.euclidean(point, c) for c in centroids]))
		
		x = pick_based_on_min_dist(min_dist)
		
		centroids.append(a[x])
		
	return centroids


# given an assignemnt of points to clusters, calculate the total distance to the centroids
def calculate_total_distance_to_centroids(a, h_assign, n_points, n_clust):
	centroids = [np.mean([a[j] for j in range(n_points) if (h_assign[j] == i)], axis=0) for i in range(n_clust)]
	
	dist = sum([sum([distance.euclidean(a[j], centroids[i]) for j in list(range(n_points)) if (h_assign[j] == i)]) for i in range(n_clust)])
	
	return dist


# given a vector of floats representing probabilities or equivalent pick an element based on them
def pick_based_on_min_dist(prob):
	x = random.uniform(0, sum(prob))
	for i in range(len(prob)):
		if x < prob[i]:
			return i
		else:
			x = x - prob[i]



# --------------------------------------------------------------------------- #
# Recursivly match cells within the clusters
# --------------------------------------------------------------------------- #


def match_RECURSIVELY(a1, a2, bname):
	ncells1, ncells2 = a1.shape[0], a2.shape[0]
	
	# make sure we fit into the memory of a generic laptop
	MAX_SIZE = 200
	
	# 0. If we have a small number of cells match them directly via Coopraiz:
	if ncells1*ncells2 < MAX_SIZE*MAX_SIZE: 
		match = entry_run_Coopraiz(a1, a2, bname)
		return prepare_return(match, list(range(ncells1)), list(range(ncells2)))


	# 1. If not cluster the cells into meta-cells
	
	# Do Equal-Size KMeans clustering
	nclust = 40
	h_assign1 = entry_EQKM(a1, int(ncells1/nclust))
	h_assign2 = entry_EQKM(a2, int(ncells2/nclust))
	
	clusters1 = create_clusters_from_labels(h_assign1)
	clusters2 = create_clusters_from_labels(h_assign2)
	

	# 2 Match the clusters
	cluster_match = match_clusters_via_DUDV(a1, a2, clusters1, clusters2, bname)
	

	# 3. For each pair of matched megacells, match the cells within it:
	assembled_match, assembled_unmatched_cells1, assembled_unmatched_cells2 = {}, [], []
	for cl_match in range(len(cluster_match)):
		cell_ids1 = clusters1[cl_match]
		cell_ids2 = clusters2[cluster_match[cl_match]]
		
		a1_sub = a1[cell_ids1,:]
		a2_sub = a2[cell_ids2,:]
		
		
		(match, unmatched_cells1, unmatched_cells2) = match_RECURSIVELY(a1_sub, a2_sub, f"{bname}_clmatch{cl_match}")
		
		for m in match:
			assembled_match[cell_ids1[m]] = cell_ids2[match[m]]
		
		for unmatched_cell1 in unmatched_cells1:
			assembled_unmatched_cells1.append(cell_ids1[unmatched_cell1])
		
		for unmatched_cell2 in unmatched_cells2:
			assembled_unmatched_cells2.append(cell_ids2[unmatched_cell2])
				

	return (assembled_match, assembled_unmatched_cells1, assembled_unmatched_cells2)



# given the match found and the cells ids prepare the proper return 
def prepare_return(match, cell_ids1, cell_ids2):
	assembled_match, unmatched_cells1, unmatched_cells2 = {}, [], []
	
	for i in range(len(cell_ids1)):
		if i in match: 
			assembled_match[cell_ids1[i]] = cell_ids2[match[i]]
		else:
			unmatched_cells1.append(cell_ids1[i])
	
	rev_match = reverse_match(match)
	for i in range(len(cell_ids2)):
		if i in rev_match:
			pass
		else:
			unmatched_cells2.append(cell_ids2[i])
	
	return (assembled_match, unmatched_cells1, unmatched_cells2)




# --------------------------------------------------------------------------- #
# Main Driving Function
# --------------------------------------------------------------------------- #

def match_exhaustively(a1, a2, bname):
	ncells1, ncells2 = a1.shape[0], a2.shape[0]
	
	# initially we have an empty match and all cells are unmatched
	assembled_match, assembled_unmatched_cells1, assembled_unmatched_cells2, count_t = {}, list(range(ncells1)), list(range(ncells2)), 0
	
	# while we have unmatched cells in both modalities
	while len(assembled_unmatched_cells1) > 0 and len(assembled_unmatched_cells2) > 0:
	
		# subset to those cells
		a1_sub = a1[assembled_unmatched_cells1,:]
		a2_sub = a2[assembled_unmatched_cells2,:]
		
		# match them the way we do
		(match, unmatched_cells1, unmatched_cells2) = match_RECURSIVELY(a1_sub, a2_sub, f"{bname}_t{count_t}")
	
		# add their match to the already found matches and keep track of those still remaining unmatched
		remain_unmatched_cells1 = []
		remain_unmatched_cells2 = []
		
		for m in match:
			assembled_match[assembled_unmatched_cells1[m]] = assembled_unmatched_cells2[match[m]]
			
		for unmatched_cell1 in unmatched_cells1:
			remain_unmatched_cells1.append(assembled_unmatched_cells1[unmatched_cell1])
		
		for unmatched_cell2 in unmatched_cells2:
			remain_unmatched_cells2.append(assembled_unmatched_cells2[unmatched_cell2])
		
		# repeat with those still left unmatched
		assembled_unmatched_cells1 = remain_unmatched_cells1
		assembled_unmatched_cells2 = remain_unmatched_cells2
		
		# write down the current assembled match
		if False:
			fo = open(f"{T}{bname}_assembled_match_{count_t}.txt", "w")
			for m in assembled_match:
				fo.write(f"{m}\t{assembled_match[m]}\n")
			fo.close()
		
		count_t += 1
	
	return assembled_match
	




# --------------------------------------------------------------------------- #
# Simple read user inputs function
# --------------------------------------------------------------------------- #

def clean_up():
	for used_dir in [f"{CPATH}compl_matrices/", f"{CPATH}jsons2/", f"{CPATH}matches/", f"{CPATH}outputs/" ]:
		files = glob.glob(f"{used_dir}*")
		for f in files:
		    os.remove(f)
	


def read_user_inputs():
	print(f"Starting Synmatch.\n")
	
	if len(sys.argv) < 4:
		print(f"insufficient arguments.\n")
	
	a1 = np.load(sys.argv[1])
	a2 = np.load(sys.argv[2])
	bname = os.path.basename(sys.argv[3])

	
	match = match_exhaustively(a1, a2, bname)
	
	write_match(match, sys.argv[3])
	
	clean_up()
	
	print(f"\nSynmatch done.\n")



# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
	read_user_inputs()
	

