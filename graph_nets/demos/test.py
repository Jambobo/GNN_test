#-*-coding:utf-8-*-

import numpy as np
import graph_nets.demos.graph_utils as gu
import matplotlib.pyplot as plt
import networkx as nx
import collections
from scipy import spatial

DISTANCE_WEIGHT_NAME = "distance"

def draw(g):
    pos=nx.spring_layout(g)
    nx.draw(g,pos,arrows=True,with_labels=True,nodelist=g.nodes(),style='dashed',edge_color='b',width=2,\
          node_color='y',alpha=0.5)
    plt.show()
G = nx.DiGraph()

# G.add_edges_from([1,2,3,4,5,6])

G.add_edges_from([(1, 2), (3, 1), (4, 2), (5, 2), (2, 3), (1, 5)])

edges = (G.edges[(u, v)] for u, v in G.edges())
print('edges_list:', list(edges))

G.add_edge(1,2,features=True)

edges = (G.edges[(u, v)] for u, v in G.edges())
print('edges_list:', list(edges))

#
# draw(G)
#
# node_connected = nx.all_pairs_node_connectivity(G)
#
# path = nx.all_simple_paths(G, 1, 4)
#
# print("node_connected_list",list(node_connected))
# print(type(node_connected))
#
# for x,y in node_connected.items():
#     print("x, y: ", x,y)
#     # print(type(y))
#     for xx, yy in y.items():
#         print(x, xx, yy)
#
rand = np.random.RandomState(seed=1)
# min_length = 1
# # Map from node pairs to the length of their shortest path.
# pair_to_length_dict = {}
# lengths = list(nx.all_pairs_shortest_path_length(G))
#
# for x, yy in lengths:
#     print("x:yy",x, yy)
#     for y, l in yy.items():
#         if l >= min_length:
#             pair_to_length_dict[x, y] = l
# # The node pairs which exceed the minimum length.
# node_pairs = list(pair_to_length_dict)
# print(pair_to_length_dict)
# print(node_pairs)
#
# # Computes probabilities per pair, to enforce uniform sampling of each
# # shortest path lengths.
# # The counts of pairs per length.
# counts = collections.Counter(pair_to_length_dict.values())
# prob_per_length = 1.0 / len(counts)
# probabilities = [
#     prob_per_length / counts[pair_to_length_dict[x]] for x in node_pairs
# ]
#
# # Choose the start and end points.
# i = rand.choice(len(node_pairs))
# print(node_pairs[i])
# start, end = node_pairs[i]
# path = nx.shortest_path(
#     G, source=start, target=end)
#
# print(list(path))
#
# # Creates a directed graph, to store the directed path from start to end.
# digraph = G.to_directed()
#
# # Add the "start", "end", and "solution" attributes to the nodes and edges.
# nodes = (digraph.nodes[n] for n in G)
# print('nodes_list:', list(nodes))
#
# digraph.add_node(start, start=True)
# digraph.add_node(end, end=True)
#
# nodes = (digraph.nodes[n] for n in G)
# print('nodes_list:', list(nodes))
#
# # start之外所有的节点都附加start=False属性 end节点同理
# digraph.add_nodes_from(gu.set_diff(digraph.nodes(), [start]), start=False)
# digraph.add_nodes_from(gu.set_diff(digraph.nodes(), [end]), end=False)
# digraph.add_nodes_from(gu.set_diff(digraph.nodes(), path), solution=False)
# digraph.add_nodes_from(path, solution=True)
# path_edges = list(gu.pairwise(path))
# print(path_edges)
#
# nodes = (digraph.nodes[n] for n in G)
# print('nodes_list:', list(nodes))
#
# digraph.add_edges_from(gu.set_diff(digraph.edges(), path_edges), solution=False)
# digraph.add_edges_from(path_edges, solution=True)
#
# edges = (digraph.edges[(u,v)] for u,v in G.edges())
# print('edges_list:', list(edges))
#
# draw(digraph)


# theta = 20
# num_nodes_min_max = (8, 9)
# graph = gu.generate_graph_zero(rand, num_nodes_min_max, theta=theta)[0]
#
#
# dg = nx.generators.directed.gn_graph(8)
# # dg = nx.generators.directed.gnc_graph(10)
# # dg = nx.generators.directed.gnr_graph(10,p=0.8)
#
#
# pos_array = rand.uniform(size=(8, 2))
# pos = dict(enumerate(pos_array))
# weight = dict(enumerate(rand.exponential(1.0, size=8)))
# geo_graph = nx.geographical_threshold_graph(
#   8, 1000.0, pos=pos, weight=weight)
#
#
# combined_graph = nx.compose_all([dg.copy(), geo_graph.copy()])
#
# # dg.add_nodes_from(dg.nodes(), pos=[pos])
#
# distances = spatial.distance.squareform(spatial.distance.pdist(pos_array))
# i_, j_ = np.meshgrid(range(8), range(8), indexing="ij")
# weighted_edges = list(zip(i_.ravel(), j_.ravel(), distances.ravel()))
# mst_graph = nx.Graph()
# mst_graph.add_weighted_edges_from(weighted_edges, weight=DISTANCE_WEIGHT_NAME)
# mst_graph = nx.minimum_spanning_tree(mst_graph, weight=DISTANCE_WEIGHT_NAME)
#
#
# nodes = (combined_graph.nodes[n] for n in combined_graph)
# print('nodes_list:', list(nodes))
# edges = (combined_graph.edges[(u, v)] for u, v in combined_graph.edges())
# print('edges_list:', list(edges))
#
# nodes = (dg.nodes[n] for n in dg)
# print('nodes_list:', list(nodes))
# edges = (dg.edges[(u, v)] for u, v in dg.edges())
# print('edges_list:', list(edges))
#
# nodes = (graph.nodes[n] for n in graph)
# print('nodes_list:', list(nodes))
# edges = (graph.edges[(u, v)] for u, v in graph.edges())
# print('edges_list:', list(edges))
#
# combined_graph = nx.compose_all([combined_graph.copy(), mst_graph.copy()])
#
# nodes = (combined_graph.nodes[n] for n in combined_graph)
# print('nodes_list:', list(nodes))
# edges = (combined_graph.edges[(u, v)] for u, v in combined_graph.edges())
# print('edges_list:', list(edges))
#
# draw(combined_graph)


