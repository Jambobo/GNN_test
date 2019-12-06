#-*-coding:utf-8-*-

import numpy as np
import graph_nets.demos.graph_utils as gu
import matplotlib.pyplot as plt
import networkx as nx



# #@title Visualize example graphs  { form-width: "30%" }
# seed = 1  #@param{type: 'integer'}
# rand = np.random.RandomState(seed=seed)
#
# num_examples = 15  #@param{type: 'integer'}
# # Large values (1000+) make trees. Try 20-60 for good non-trees.
# theta = 20  #@param{type: 'integer'}
# num_nodes_min_max = (16, 17)
#
# input_graphs, target_graphs, graphs = gu.generate_networkx_graphs(
#     rand, num_examples, num_nodes_min_max, theta)

# num = min(num_examples, 16)
# w = 3
# h = int(np.ceil(num / w))
# fig = plt.figure(40, figsize=(w * 4, h * 4))
# fig.clf()
# for j, graph in enumerate(graphs):
#   ax = fig.add_subplot(h, w, j + 1)
#   pos = gu.get_node_dict(graph, "pos")
#   plotter = gu.GraphPlotter(ax, graph, pos)
#   plotter.draw_graph_with_solution()
#
# plt.show()


def draw(g):                   #显示图

    pos=nx.spring_layout(G)

    nx.draw(g,pos,arrows=True,with_labels=True,nodelist=G.nodes(),style='dashed',edge_color='b',width=2,\

          node_color='y',alpha=0.5)

    plt.show()


G = nx.DiGraph()

# G.add_path([1,2,3,4,5,6])

G.add_edges_from([(1, 2), (1, 5), (3, 1), (5, 2), (2, 3), (4, 2)])


draw(G)

node_connected = nx.all_pairs_node_connectivity(G)

path = nx.all_simple_paths(G, 1, 4)

print(node_connected)
print(len(list(path)))
print(list(path))

