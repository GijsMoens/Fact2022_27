import networkx as nx
from node2vec import Node2Vec
import graph
import os
from argparse import ArgumentParser, FileType, ArgumentDefaultsHelpFormatter

parser = ArgumentParser("node2vec", formatter_class=ArgumentDefaultsHelpFormatter,
                        conflict_handler='resolve')

parser.add_argument("--input_file", type=str, help="graph file")
parser.add_argument("--sens_attr_file", type=str, help="graph attr file")
parser.add_argument("--method", type=str, help="weighing algorithm")
parser.add_argument("--weighted", type=str, help="method for edge weighing")
parser.add_argument("--nx_graph_file", type=str, help="path to networkx graph")
parser.add_argument("--output_file", type=str, help="file of output embeddings")
args = parser.parse_args()


input_file = args.input_file
attr_file = args.sens_attr_file
method= args.method
weight_method = args.weighted
nx_graph_file = args.nx_graph_file
output_file = args.output_file

#create folder
NEWPATH = '/'.join(args.output_file.split('/')[:-1])
if not os.path.exists(NEWPATH):
    os.makedirs(NEWPATH)

if not os.path.isfile(nx_graph_file):
    G = graph.load_edgelist(input_file, attr_file_name=attr_file)

    if method == "fairwalk" or method == "crosswalk":
        G = graph.set_weights(G, weight_method)
        nx_graph = graph.graph_to_networkx(G, nx_graph_file)


if method == "deepwalk":
    nx_graph = nx.read_edgelist(input_file)

elif method == "fairwalk" or method == "crosswalk":
    nx_graph = nx.read_edgelist(nx_graph_file)

node2vec = Node2Vec(nx_graph, dimensions=128, walk_length=40, num_walks=30, workers=30,p=0.5,q=0.5)
model = node2vec.fit(window=10) 
model.wv.save_word2vec_format(output_file)

