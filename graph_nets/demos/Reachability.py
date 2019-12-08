import numpy as np
import graph_nets.demos.graph_utils as gu
import matplotlib.pyplot as plt
import networkx as nx
import collections
import random


def draw(g):
    pos=nx.spring_layout(g)
    nx.draw(g,pos,arrows=True,with_labels=True,nodelist=g.nodes(),style='dashed',edge_color='b',width=2,\
          node_color='y',alpha=0.5)
    plt.show()


def reachable(graph):
    # graph = nx.DiGraph()
    # graph.add_edges_from([(1, 2), (3, 1), (4, 2), (5, 2), (2, 3), (1, 5)])
    node_connected = nx.all_pairs_node_connectivity(graph)
    path = nx.all_simple_paths(graph, 1, 4)
    paths = []
    path_nodes = []

    # print
    print("node_connected_list", list(node_connected))
    # print(type(node_connected))

    i = random.choice(list(node_connected))
    source = i
    print(i)
    node_connected_pair = {}
    node_reachable = []
    for x, yy in node_connected.items():
        # print("x, y: ", x,yy)
        # print(type(y))
        for y, l in yy.items():
            if x == i and l > 0:
                node_connected_pair[x, y] = l
                path = nx.all_simple_paths(graph, x, y)
                node_reachable.append(y)
                # print(path)
                # if len(list(path)) > path_max_length:
                #     longest_path = path
                #     path_max_length = len(list(path))
                for p in list(path):
                    # print(p)
                    paths.extend(list(gu.pairwise(p)))
                    path_nodes.extend(p)
            # print(x, y, l)

    node_pairs = list(node_connected_pair)
    paths = set(paths)
    path_nodes = set(path_nodes)
    # print(path_nodes)
    # print(list(node_connected_pair))
    # print(node_reachable)
    # print("paths:", paths)
    # print(set(paths))

    digraph = graph.to_directed()

    digraph.add_node(source, source=True)
    digraph.add_nodes_from(gu.set_diff(digraph.nodes(), [source]), source=False)
    digraph.add_nodes_from(node_reachable, reachable=True)
    digraph.add_nodes_from(gu.set_diff(digraph.nodes(), node_reachable), reachable=False)
    digraph.add_nodes_from(gu.set_diff(digraph.nodes(), path_nodes), solution=False)
    digraph.add_nodes_from(path_nodes, solution=True)
    digraph.add_edges_from(gu.set_diff(digraph.edges(), paths), solution=False)
    digraph.add_edges_from(paths, solution=True)
    nodes = (digraph.nodes[n] for n in digraph)
    print('nodes_list:', list(nodes))
    edges = (digraph.edges[(u, v)] for u, v in digraph.edges())
    print('edges_list:', list(edges))
    # rand = np.random.RandomState(seed=1)
    # i = rand.choice(len(node_pairs))

    # print(node_pairs[i])
    # start, end = node_pairs[i]
    # path = nx.all_simple_paths(
    #     G, source=start, target=end)
    #
    # print(len(path))

    return digraph

if __name__ == '__main__':
    seed = 1
    rand = np.random.RandomState(seed=seed)
    theta = 20
    num_nodes_min_max = (16, 17)
    graph = gu.generate_graph(rand, num_nodes_min_max, theta=theta)[0]

    graph = reachable(graph)

    draw(graph)
    # graph = reachable(graph)

    # i_, j_ = np.meshgrid(range(16), range(16), indexing="ij")
    # print(list(zip(i_.ravel(), j_.ravel())))

    # draw(graph.to_directed())


