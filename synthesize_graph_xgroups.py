from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import os

if __name__ == '__main__':
    parser = ArgumentParser("Synthesize Graph",
                            formatter_class=ArgumentDefaultsHelpFormatter,
                            conflict_handler='resolve')
    parser.add_argument('--seed', type=int, help='seed for random processes')
    parser.add_argument('--nodes', type=int, help='Number nodes')
    parser.add_argument('--num_colors', type=int, help='Number of groups (colors) in the graph ')
    parser.add_argument('--Pcolors', type= str, help='Probability of being a color for each node as delimited string')
    parser.add_argument('--Phom_min', type=float, help='minimum probability of within gropup connections')
    parser.add_argument('--Phom_max', type=float, help='maximum probability of within group connections')
    parser.add_argument('--Phet_min', type=float, help='minimum probability of cross group connections')
    parser.add_argument('--Phet_max', type=float, help='maximum probability of cross group connections')
    parser.add_argument('--output_dir', type=str, help='directory of outputs')

    
    args = parser.parse_args()
    
    
    
    np.random.seed(args.seed)
    n = args.nodes
    
    Pcolors = np.array([float(item) for item in args.Pcolors.split(',')])
    if len(Pcolors) != args.num_colors:
        raise ValueError(f'number group probabilities ({len(Pcolors)}) is not equal to number of groups ({args.num_colors})')
    
    Pcolors = Pcolors / Pcolors.sum() #normalize group probabilities
    
    Phom = np.random.uniform(args.Phom_min, args.Phom_max, size= args.num_colors)
    Phet = np.triu(np.random.uniform(args.Phet_min, args.Phet_max, size=(args.num_colors, args.num_colors)))
    Ptot = Phet.copy()
    np.fill_diagonal(Ptot, Phom)

    #initialize number of nodes with each color
    n_colors = []
    for i in range(args.num_colors - 1):
        n_colors.append(int(n * Pcolors[i]))
    n_colors.append(n - sum(n_colors)) #compromise last probability to make sure number of nodes is n

   
    edges = []

    for idx_self, n_self in enumerate(n_colors):
        for idx_other, n_other in enumerate(n_colors[idx_self: ]):
            n_before_self = sum(n_colors[:idx_self])
            n_before_other = sum(n_colors[:idx_self + idx_other])
            
            if idx_self == idx_self + idx_other:
                for i in range(n_self):
                    for j in range(n_self):
                        if np.random.rand() < Ptot[idx_self, idx_self]:
                            edges.append((n_before_self + i, n_before_self + j))
                            edges.append((n_before_self + j, n_before_self + i))
            else:
                for i in range(n_self):
                    for j in range(n_other):
                        if np.random.rand() < Ptot[idx_self, idx_self + idx_other]:
                            
                            edges.append((n_before_self + i, n_before_other + j))
                            edges.append((n_before_other + j, n_before_self + i))


    #create folder to store graphs in
    NEWPATH = args.output_dir
    if not os.path.exists(NEWPATH):
        os.makedirs(NEWPATH)

    filename = NEWPATH + '/synth' + str(n) + \
               '_Phom' + str(args.Phom_min)+ '-' + str(args.Phom_max) + '_Phet' + str(args.Phet_min) + '-' + str(args.Phet_max)

    with open(filename + '.attr', 'w') as f:
        for idx, n_c in enumerate(n_colors):
            n_before = sum(n_colors[:idx])
            for i in range(n_c):
                f.write(str(n_before + i) + ' ' + str(idx) + '\n')

    with open(filename + '.links', 'w') as f:
        for e in edges:
            f.write(str(e[0]) + ' ' + str(e[1]) + '\n')

