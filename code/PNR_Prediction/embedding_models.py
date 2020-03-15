import numpy as np
import pandas as pd
import os
import sys

# 添加embedding baselines的目录
sys.path.append(r'D:\hybridrec\code\githubcode\deepwalk_master\deepwalk')
sys.path.append(r'D:\hybridrec\code\githubcode\node2vec-master(Stanford)\src')
sys.path.append(r'D:\hybridrec\code\githubcode\Splitter-master\src')
sys.path.append(r'D:\hybridrec\code\githubcode\ProNE-master')
sys.path.append(r'D:\hybridrec\code\githubcode\AttentionWalk-master\src')
sys.path.append(r'D:\hybridrec\code\githubcode\GraRep-master2\GraRep-master\src')
sys.path.append(r'D:\hybridrec\code\githubcode\SDNE_struc2vec_line\examples')
sys.path.append(r'D:\hybridrec\code\githubcode\DRNE-master\src')
sys.path.append(r'D:\hybridrec\code\githubcode\PRUNE-master\src')

from main_deepwalk import deepwalk_main
from main_node2vec import node2vec_main
from main_splitter import splitter_main
from main_prone import prone_main
from main_attentionwalk import attentionwalk_main
from main_grarep import grarep_main
from main_sdne import sdne_main
from main_struc2vec import struc2vec_main
from main_line import line_main
from main_drne import drne_main
from main_prune import prune_main
import configparser




def run_emb_method(input=None, output=None, emb_method_name=None):
    config_path = 'conf/' + emb_method_name + '.properties'
    config = configparser.ConfigParser()
    config.read(config_path)
    conf = dict(config.items("hyperparameters"))

    if emb_method_name=='deepwalk':
        deepwalk_main(input=input,
                      output=output,
                      representation_size=int(conf['embedding_size']),
                      number_walks = int(conf['number_walks']),
                      walk_length = int(conf['walk_length']),
                      window_size = int(conf['window_size']))
    elif emb_method_name=='node2vec':
        node2vec_main(input=input,
                      output=output,
                      dimensions=int(conf['embedding_size']),
                      walk_length=int(conf['walk_length']),
                      num_walks=int(conf['num_walks']),
                      window_size=int(conf['window_size']),
                      p_value=float(conf['p_value']),
                      q_value=float(conf['q_value']))
    elif emb_method_name=='splitter':
        splitter_main(input=input,
                      output=output,
                      number_of_walks=int(conf['number_of_walks']),
                      window_size=int(conf['window_size']),
                      negative_samples=int(conf['negative_samples']),
                      walk_length=int(conf['walk_length']),
                      learning_rate=float(conf['learning_rate']),
                      lambd=float(conf['lambd']),
                      dimensions=int(conf['embedding_size'])
                      )
    elif emb_method_name=='prone':
        prone_main(input=input,
                   output=output,
                   dimension=int(conf['embedding_size']),
                   step=int(conf['step']),
                   theta=float(conf['theta']),
                   mu=float(conf['mu'])
                   )
    elif emb_method_name=='attentionwalk':
        attentionwalk_main(input=input,
                           output=output,
                           dimensions=int(conf['embedding_size']),
                           epochs=int(conf['epochs']),
                           window_size=int(conf['window_size']),
                           num_of_walks=int(conf['num_of_walks']),
                           beta=float(conf['beta']),
                           gamma=float(conf['gamma']),
                           learning_rate=float(conf['learning_rate'])
                           )
    elif emb_method_name=='drne':
        drne_main(input=input,
                  output=output,
                  embedding_size=int(conf['embedding_size']),
                  epochs=int(conf['epochs']),
                  learning_rate=float(conf['learning_rate']),
                  lamb=float(conf['lamb']),
                  sampling_size=int(conf['sampling_size']))
    elif emb_method_name=='prune':
        prune_main(input=input,
                   output=output,
                   dimensions=int(conf['embedding_size']),
                   lamb=float(conf['lamb']),
                   learning_rate=float(conf['learning_rate']),
                   epoch=int(conf['epoch']),
                   batch_size=int(conf['batch_size'])
                   )
    elif emb_method_name=='grarep':
        grarep_main(input=input, output=output)
    elif emb_method_name=='sdne':
        sdne_main(input=input, output=output)
    elif emb_method_name=='struc2vec':
        struc2vec_main(input=input, output=output)
    elif emb_method_name=='line':
        line_main(input=input, output=output)
