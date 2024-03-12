from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.semi_supervised import LabelPropagation
from sklearn.metrics import pairwise_distances
import numpy as np
from argparse import ArgumentParser, FileType, ArgumentDefaultsHelpFormatter
import os
import warnings
warnings.filterwarnings('ignore')
label_type = 'college'

def read_embeddings(emb_file):
    emb = dict()
    with open(emb_file, 'r') as fin:
        for i_l, line in enumerate(fin):
            s = line.split()
            if i_l == 0:
                dim = int(s[1])
                continue
            emb[int(s[0])] = [float(x) for x in s[1:]]
    return emb, dim

def read_labels(label_file, emb):
    labels = dict()
    with open(label_file, 'r') as fin:
        for line in fin:
            s = line.split()
            id = int(s[0])
            if id in emb:
                if label_type == 'major':
                    labels[id] = int(s[3])
                elif label_type == 'college':
                    labels[id] = int(s[1])
                elif label_type == 'college_2':
                    tmp = int(s[1])
                    if tmp > 5:
                        labels[id] = 1
                    else:
                        labels[id] = 0
                else:
                    raise Exception('unknown label_type')
    return labels


def read_sensitive_attr(sens_attr_file, emb):
    sens_attr = dict()
    with open(sens_attr_file, 'r') as fin:
        for line in fin:
            s = line.split()
            id = int(s[0])
            if id in emb:
                sens_attr[id] = int(s[1])
    return sens_attr

if __name__ == '__main__':
    parser = ArgumentParser("node classification", formatter_class=ArgumentDefaultsHelpFormatter,
                        conflict_handler='resolve')

    
    parser.add_argument("--emb_file", type=str, help="file of the embeddings")
    parser.add_argument("--label_file", type=str, help="labels for classification")
    parser.add_argument("--sens_attr_file", type=str, help="sensitive attribute file")
    parser.add_argument("--method", type=str, help="embedding method")
    parser.add_argument("--out_file", type=str, help="file of output results")
    args = parser.parse_args()
    
    emb_file = args.emb_file
    label_file = args.label_file
    sens_attr_file = args.sens_attr_file
    method = args.method
    out_file = args.out_file
        
    res_avg=[]
    res_1_avg=[]
    res_0_avg=[]
    res_diff_avg=[]
    res_var_avg=[]

    
    res_total = []
    res_1_total = []
    res_0_total = []
    res_diff_total = []
    res_var_total = []
    
    for iter in range(200):
        run_i = 1 + np.mod(iter, 5)

        emb, dim = read_embeddings(emb_file)
        labels = read_labels(label_file, emb)
        sens_attr = read_sensitive_attr(sens_attr_file, emb)

        assert len(labels) == len(emb) == len(sens_attr)

        n = len(emb)

        X = np.zeros([n, dim])
        y = np.zeros([n])
        z = np.zeros([n])
        for i, id in enumerate(emb):
            X[i,:] = np.array(emb[id])
            y[i] = labels[id]
            z[i] = sens_attr[id]

        idx = np.arange(n)
        np.random.shuffle(idx)
        n_train = int(n // 2)

        X = X[idx,:]
        y = y[idx]
        z = z[idx]
        X_train = X
        X_test = X[n_train:]
        y_train = np.concatenate([y[:n_train], -1*np.ones([n-n_train])])
        y_test = y[n_train:]
        z_test = z[n_train:]
        g = np.mean(pairwise_distances(X))
        clf = LabelPropagation(gamma = g).fit(X_train, y_train)

        y_pred = clf.predict(X_test)

        res = 100 * np.sum(y_pred == y_test) / y_test.shape[0]

        idx_1 = (z_test == 1)
        res_1 = 100 * np.sum(y_pred[idx_1] == y_test[idx_1]) / np.sum(idx_1)

        idx_0 = (z_test == 0)
        res_0 = 100 * np.sum(y_pred[idx_0] == y_test[idx_0]) / np.sum(idx_0)


        res_diff = np.abs(res_1 - res_0)
        res_var = np.var([res_1, res_0])

        res_total.append(res)
        res_1_total.append(res_1)
        res_0_total.append(res_0)
        res_diff_total.append(res_diff)
        res_var_total.append(res_var)

    res_avg.append(np.mean(np.array(res_total), axis=0))
    res_1_avg.append(np.mean(np.array(res_1_total), axis=0))
    res_0_avg.append(np.mean(np.array(res_0_total), axis=0))
    res_diff_avg.append(np.mean(np.array(res_diff_total), axis=0))
    res_var_avg.append(np.mean(np.array(res_var_total), axis=0))
    
    #create folder
    NEWPATH = '/'.join(out_file.split('/')[:-1])
    if not os.path.exists(NEWPATH):
        os.makedirs(NEWPATH)

    res_avg_ = np.mean(res_avg)
    with open(out_file, 'w') as f:
        f.write(str(np.mean(res_avg))+ ', '+ str(np.mean(res_1_avg))+ ', '+ str(np.mean(res_0_avg))+ ', '+ str(np.mean(res_diff_avg))+ ', '+ str(np.mean(res_var_avg)))
