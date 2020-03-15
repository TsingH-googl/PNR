import networkx as nx
from split_train_test import *


def read_graph(weighted=0, input=None, directed=0):
	'''
	Reads the input network in networkx.
	'''
	if weighted:
		G = nx.read_edgelist(input, nodetype=int, data=(('weight',float),),
							 create_using=nx.DiGraph(), delimiter=' ')
	else:
		G = nx.read_edgelist(input, nodetype=int, create_using=nx.DiGraph(), delimiter=' ')
		for edge in G.edges():
			G[edge[0]][edge[1]]['weight'] = 1

	if not directed:
		G = G.to_undirected()

	return G


if __name__ == '__main__':
     filePath = 'input/PB_undirected_1.edgelist' # PB_directed_1.edgelist  blogcatalog_directed_1.edgelist
# 读入数据集
print("----Reading graph......")
G = read_graph(weighted=0, input=filePath, directed=0)
nx.write_edgelist(G, 'output/Graph.txt', data=False)
print(len(G))
print(len(G.edges()))

# 划分数据集
train_E, test_E = split_train_test(G, train_frac=0.9)
G.remove_edges_from(test_E)
print("G_giantCom :" + str(nx.is_connected(G)))
nx.write_edgelist(G, 'output/Graph_train.txt', data = False)
print(len(G))
print(len(G.edges()))

# 验证最大联通子图
G_simple = max(nx.connected_component_subgraphs(G), key=len)
nx.write_edgelist(G_simple, 'output/Graph_train_simple.txt', data=False)
print(len(G_simple))
print(len(G_simple.edges()))
pass