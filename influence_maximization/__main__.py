from generalGreedy import *
import utils as ut
from IC import *
import matplotlib.pyplot as plt
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import numpy as np
import os
import networkx as nx


def dfs(v, mark, G, colors, num_labels):
    res = np.zeros(num_labels)
    res[int(colors[v])] += 1
    mark.update([v])
    for u in G[v]:
        if u not in mark:
            res += dfs(u, mark, G, colors, num_labels)
    return res


class fairInfMaximization():

    def __init__(self, num=-1, args=None):
        self.filename = args.graph_file
        self.weight = args.act_weight
        #if self.rice or self.rice_subset or self.sample_1000 or self.sample_4000_connected_subset or self.synthetic or self.synthetic_3g or self.synthetic1 or self.twitter or self.synthetic_extention or self.node2vec:
        self.G = ut.get_data(self.filename, self.weight)
        # elif self.twitter:
        #     self.G = ut.get_twitter_data(self.filename, self.weight)

    def test_greedy(self, filename, budget, G_greedy=None):
        generalGreedy_node_parallel(filename, self.G, budget=budget, gamma=None, G_greedy=G_greedy)

    def test_kmedoids(self, emb_filename, res_filename):

        print(res_filename)
        v, em = ut.load_embeddings(emb_filename, self.G.nodes())

        influenced, influenced_grouped = [], []
        seeds = []

        k=40
        print('--------', k)
        S = ut.get_kmedoids_centers(em, k, v)
        I, I_grouped = map_fair_IC((self.G, S))
        influenced.append(I)
        influenced_grouped.append(I_grouped)

        S_g = {c:[] for c in np.unique([self.G.nodes[v]['color'] for v in self.G.nodes])}
        for n in S:
            c = self.G.nodes[n]['color']
            S_g[c].append(n)

        seeds.append(S_g)

        ut.write_files(res_filename, influenced, influenced_grouped, seeds)

if __name__ == '__main__':

    parser = ArgumentParser("influence maximization",
                            formatter_class=ArgumentDefaultsHelpFormatter,
                            conflict_handler='resolve')

    parser.add_argument('--graph_file', type=str, help='file of graphs')
    parser.add_argument('--act_weight', type=float, help='activation weight for inf max algorithm')
    parser.add_argument('--input', type=str, help='directory of input embeddings')
    parser.add_argument('--output', type=str, help='output file')
    args = parser.parse_args()


    fair_inf = fairInfMaximization(args=args) 
    
    #create folder
    NEWPATH = '/'.join(args.output.split('/')[:-1])
    if not os.path.exists(NEWPATH):
        os.makedirs(NEWPATH)

    embfile = args.input
    resfile = args.output
    fair_inf.test_kmedoids(embfile, resfile)

    

