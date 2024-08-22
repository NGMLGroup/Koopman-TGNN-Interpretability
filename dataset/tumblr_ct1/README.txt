README for dataset tumblr_ct1


=== Usage ===

This folder contains the following comma separated text files 
(replace DS by the name of the dataset):

n = total number of nodes
m = total number of edges
N = number of graphs

(1) 	DS_A.txt (m lines) 
	sparse (block diagonal) adjacency matrix for all graphs,
	each line corresponds to (row, col) resp. (node_id, node_id).

(2) 	DS_graph_indicator.txt (n lines)
	column vector of graph identifiers for all nodes of all graphs,
	the value in the i-th line is the graph_id of the node with node_id i.

(3) 	DS_graph_labels.txt (N lines) 
	class labels for all graphs in the dataset,
	the value in the i-th line is the class label of the graph with graph_id i.

(4) 	DS_node_labels.txt (n lines)
	each line contains a sequence of alternating time stamps and node labels, 
	where the i-th line corresponds to the node with node_id i. At line i, at 
	each uneven position (starting at 1) there is a time stamp t, followed by 
	the node label that node i is assigned at t. For example, if a node i has 
	label 0 at time 0 and the node’s label is then changed at t=4 for the 
	first time to the new label 1, the i-th line would start with “0, 0, 4, 1”.

(5) 	DS_edge_attributes.txt (m lines; same size as DS_A.txt)
	the availability times of the edges in DS_A.txt.

(6)	DS_info.txt
	Statistics for the dataset.


=== Description ===

Based on a graph containing quoting between Tumblr users, a subset of the Memetracker data set [2].

To obtain data sets for supervised graph classification, we generated induced subgraphs by starting a BFS run from each vertex.
We simulated a dissemination process on each of the subgraphs according to the usceptible-infected (SI) model.
A detailed description can be found in [1]. 

This dataset is for the classification task 1 described in [1].

Please cite [1] if you use the data set.


=== References ===

[1] Lutz Oettershagen, Nils Kriege, Christopher Morris, Petra Mutzel.
    Temporal Graph Kernels for Classifying Dissemination Processes.
    SIAM International Conference on Data Mining (SDM) 2020.

[2] J. Leskovec, L. Backstrom, J. Kleinberg. 
    Meme-tracking and the Dynamics of the News Cycle. 
    ACM SIGKDD Intl. Conf. on Knowledge Discovery and Data Mining, 2009.
