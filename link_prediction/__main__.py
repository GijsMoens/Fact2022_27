from sklearn.linear_model import LogisticRegression
import numpy as np
import pickle
import os.path
from argparse import ArgumentParser, FileType, ArgumentDefaultsHelpFormatter

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

def read_sensitive_attr(sens_attr_file, emb):
    sens_attr = dict()
    with open(sens_attr_file, 'r') as fin:
        for line in fin:
            s = line.split()
            id = int(s[0])
            if id in emb:
                sens_attr[id] = int(s[1])
    return sens_attr

def read_links(links_file, emb):
    with open(links_file, 'rb') as f:
        links = pickle.load(f)
    return [l for l in links if l[0] in emb.keys() and l[1] in emb.keys()]

def extract_features(u,v):
    return (u-v)**2
    # return np.array([np.sqrt(np.sum((u-v)**2))])

if __name__ == '__main__':
    parser = ArgumentParser("link prediction", formatter_class=ArgumentDefaultsHelpFormatter,
                          conflict_handler='resolve')
    parser.add_argument("--num_iters", type=int, help="number of iterations")
    parser.add_argument("--emb_file", type=str, help="directory of input")
    parser.add_argument("--sens_attr_file", type=str, help="file of the labels")
    parser.add_argument("--train_links_file", type=str, help="file of train links")
    parser.add_argument("--test_links_file", type=str, help="file of test links")
    parser.add_argument("--output_file", type=str, help="file where results are stored")
    args = parser.parse_args()

    num_iters = args.num_iters
    emb_file = args.emb_file
    sens_attr_file = args.sens_attr_file
    train_links_file = args.train_links_file
    test_links_file = args.test_links_file
    out_file = args.output_file

    #create folder
    NEWPATH = '/'.join(out_file.split('/')[:-1])
    if not os.path.exists(NEWPATH):
        os.makedirs(NEWPATH)
    
    emb, dim = read_embeddings(emb_file)
    sens_attr = read_sensitive_attr(sens_attr_file, emb)
    train_links = read_links(train_links_file, emb)
    test_links = read_links(test_links_file, emb)
    
    all_labels = list(set(sens_attr.values()))
    label_pairs = [str(all_labels[i]) + ',' + str(all_labels[j]) for i in range(len(all_labels)) for j in range(i, len(all_labels))]
    accuracy_keys = label_pairs + ['max_diff', 'var', 'total']
    accuracy = {k : [] for k in accuracy_keys}
    
    print(f'all attributes: {all_labels}')

    for key in label_pairs + ['total']:
        if key == 'total':
            valid_edge_pairs = [(all_labels[i],all_labels[j]) for i in range(len(all_labels)) for j in range(len(all_labels))]
        else:
            l1, l2 = [int(l) for l in key.split(',')]
            valid_edge_pairs = [(l1, l2)]
            if l1 != l2:
                valid_edge_pairs.append((l2,l1))

        filtered_train_links = [l for l in train_links if (l[2], l[3]) in valid_edge_pairs]
        filtered_test_links = [l for l in test_links if (l[2],l[3]) in valid_edge_pairs]

        filtered_train_links = train_links

        clf = LogisticRegression(solver='lbfgs')
        x_train = np.array([extract_features(np.array(emb[l[0]]), np.array(emb[l[1]])) for l in filtered_train_links])
        y_train = np.array([l[4] for l in filtered_train_links])
        x_test = np.array([extract_features(np.array(emb[l[0]]), np.array(emb[l[1]])) for l in filtered_test_links])
        y_test = np.array([l[4] for l in filtered_test_links])
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)
        accuracy[key].append(100 * np.sum(y_test == y_pred) / x_test.shape[0])

    last_accs = [accuracy[k][-1] for k in label_pairs]
    accuracy['max_diff'].append(np.max(last_accs) - np.min(last_accs))
    accuracy['var'].append(np.var(last_accs))


with open(out_file, 'w') as f:
    for k in accuracy_keys:
        f.write(str(k) + ' ' + str(np.mean(accuracy[k])) + ' ')
