import networkx as nx
import os
from split_train_test import *


def read_graph(weighted=0, input=None, directed=0):
    '''
    Reads the input network in networkx.
    '''
    if weighted:
        G = nx.read_edgelist(input, nodetype=int, data=(('weight', float),),
                             create_using=nx.DiGraph(), delimiter=' ')
    else:
        G = nx.read_edgelist(input, nodetype=int, create_using=nx.DiGraph(), delimiter=' ')
        for edge in G.edges():
            G[edge[0]][edge[1]]['weight'] = 1

    if not directed:
        G = G.to_undirected()

    return G


if __name__ == '__main__':

    "node_id 从1开始；写入的train和test文件也是从1开始"

    dataset_dir = 'D:\hybridrec\dataset\preprocessing_code//' # 改这里
    split_dir   = 'D:\hybridrec\dataset\split_train_test\preprocessing_code//' # 改这里
    train_frac = 0.6

    dataset_files = os.listdir(dataset_dir)
    for fi in dataset_files:
        fi_d = os.path.join(dataset_dir, fi)
        if os.path.isdir(fi_d) & \
                ((fi == 'ca-hepph') or (fi == 'facebook_combined') or (fi == 'yeast')or (fi == 'email-eucore')
                 # or (fi == 'petster-friendships-hamster') or (fi == 'wiki')
                 # or (fi == 'usa-airports') or (fi == 'pubmed') or (fi == 'digg_reply')
                 # or (fi == 'ego-gplus') or (fi == 'enron') or (fi == 'epinions')
                ): # & (fi == 'router') 在test_ratio = 0.2不可以完全联通地划分
            print("dealing: "+str(fi))
            # 读
            fileName = str(fi) + '_undirected_1_giantCom.edgelist'
            filePath = os.path.join(fi_d, fileName)
            G = read_graph(weighted=0, input=filePath, directed=0)
            # print("G:" + str(len(G)))
            #             # print("G.edgesNum:" + str(len(G.edges())))
            train_E, test_E = split_train_test(G, train_frac=train_frac)
            G_train = G
            G_train.remove_edges_from(test_E)
            # print("G_train:" + str(len(G_train)))
            # print("G_train.edgesNum:" + str(len(G_train.edges())))

            ##############################  directed nodeid from 1 ###############################
            # 写
            write_dir_name = os.path.join(split_dir, fi)
            train_write_file_name = 'train_' + fi+'_directed_1_giantCom.edgelist'
            train_write_path = os.path.join(write_dir_name, train_write_file_name)
            test_write_file_name = 'test_' + fi + '_directed_1_giantCom.edgelist'
            test_write_path = os.path.join(write_dir_name, test_write_file_name)

            if not os.path.exists(write_dir_name):
                os.makedirs(write_dir_name)

            # nx.write_edgelist(G_train, train_write_path, data=False)
            with open(train_write_path, 'w+') as f:
                for edge in sorted(train_E):
                    f.write(str(edge[0])+' ' + str(edge[1]) + '\n')
                f.close()

            with open(test_write_path, 'w+') as f:
                for edge in sorted(test_E):
                    f.write(str(edge[0])+' ' + str(edge[1]) + '\n')
                f.close()

            ##############################  undirected nodeid from 1  ###############################
            # 写
            write_dir_name = os.path.join(split_dir, fi)
            train_write_file_name = 'train_' + fi+'_undirected_1_giantCom.edgelist'
            train_write_path = os.path.join(write_dir_name, train_write_file_name)
            test_write_file_name = 'test_' + fi + '_undirected_1_giantCom.edgelist'
            test_write_path = os.path.join(write_dir_name, test_write_file_name)

            if not os.path.exists(write_dir_name):
                os.makedirs(write_dir_name)

            # nx.write_edgelist(G_train, train_write_path, data=False)
            with open(train_write_path, 'w+') as f:
                for edge in sorted(train_E):
                    f.write(str(edge[0])+' ' + str(edge[1]) + '\n')
                    f.write(str(edge[1]) + ' ' + str(edge[0]) + '\n')
                f.close()

            with open(test_write_path, 'w+') as f:
                for edge in sorted(test_E):
                    f.write(str(edge[0])+' ' + str(edge[1]) + '\n')
                    f.write(str(edge[1]) + ' ' + str(edge[0]) + '\n')
                f.close()

            ##############################  directed nodeid from 0 ###############################
            # 写
            write_dir_name = os.path.join(split_dir, fi)
            train_write_file_name = 'train_' + fi+'_directed_0_giantCom.edgelist'
            train_write_path = os.path.join(write_dir_name, train_write_file_name)
            test_write_file_name = 'test_' + fi + '_directed_0_giantCom.edgelist'
            test_write_path = os.path.join(write_dir_name, test_write_file_name)

            if not os.path.exists(write_dir_name):
                os.makedirs(write_dir_name)

            # nx.write_edgelist(G_train, train_write_path, data=False)
            with open(train_write_path, 'w+') as f:
                for edge in sorted(train_E):
                    f.write(str(edge[0]-1)+' ' + str(edge[1]-1) + '\n')
                f.close()

            with open(test_write_path, 'w+') as f:
                for edge in sorted(test_E):
                    f.write(str(edge[0]-1)+' ' + str(edge[1]-1) + '\n')
                f.close()

            ##############################  undirected nodeid from 0  ###############################
            # 写
            write_dir_name = os.path.join(split_dir, fi)
            train_write_file_name = 'train_' + fi+'_undirected_0_giantCom.edgelist'
            train_write_path = os.path.join(write_dir_name, train_write_file_name)
            test_write_file_name = 'test_' + fi + '_undirected_0_giantCom.edgelist'
            test_write_path = os.path.join(write_dir_name, test_write_file_name)

            if not os.path.exists(write_dir_name):
                os.makedirs(write_dir_name)

            # nx.write_edgelist(G_train, train_write_path, data=False)
            with open(train_write_path, 'w+') as f:
                for edge in sorted(train_E):
                    f.write(str(edge[0]-1)+' ' + str(edge[1]-1) + '\n')
                    f.write(str(edge[1]-1) + ' ' + str(edge[0]-1) + '\n')
                f.close()

            with open(test_write_path, 'w+') as f:
                for edge in sorted(test_E):
                    f.write(str(edge[0]-1)+' ' + str(edge[1]-1) + '\n')
                    f.write(str(edge[1]-1) + ' ' + str(edge[0]-1) + '\n')
                f.close()

        pass
    pass




    # 下面丢弃，仅仅作为参考
    # # 读入数据集
    # print("----Reading graph......")
    # G = read_graph(weighted=0, input=filePath, directed=0)
    # nx.write_edgelist(G, 'output/Graph.txt', data=False)
    # print(len(G))
    # print(len(G.edges()))
    #
    # # 划分数据集
    # train_E, test_E = split_train_test(G, train_frac=0.9)
    # G.remove_edges_from(test_E)
    # print("G_giantCom :" + str(nx.is_connected(G)))
    # nx.write_edgelist(G, 'output/Graph_train.txt', data=False)
    # print(len(G))
    # print(len(G.edges()))
    #
    # # 验证最大联通子图
    # G_simple = max(nx.connected_component_subgraphs(G), key=len)
    # nx.write_edgelist(G_simple, 'output/Graph_train_simple.txt', data=False)
    # print(len(G_simple))
    # print(len(G_simple.edges()))
    pass
