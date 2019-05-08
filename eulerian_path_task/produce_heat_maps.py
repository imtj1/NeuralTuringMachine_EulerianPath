import pickle
import matplotlib.pyplot as plt
import numpy as np
import  networkx as nx

EXPERIMENT_NAME = 'nback'
MANN = 'NTM'
TASK = 'graph'

HEAD_LOG_FILE = 'head_logs/{0}.p'.format(EXPERIMENT_NAME)
GENERALIZATION_HEAD_LOG_FILE = 'head_logs/generalization_{0}.p'.format(EXPERIMENT_NAME)

outputs = pickle.load(open(HEAD_LOG_FILE, "rb"))
outputs.update(pickle.load(open(GENERALIZATION_HEAD_LOG_FILE, "rb")))


def one_hot_to_idx(vec):
    return np.where(np.asarray(vec) == 1)[0][0]


for seq_len, heat_maps_list in outputs.items():
    for step, heat_maps in enumerate(heat_maps_list[-2:] if len(heat_maps_list) >= 2 else heat_maps_list):
        inputs = heat_maps['inputs']
        labels = heat_maps['labels']
        outputs = heat_maps['outputs']

        if TASK == 'graph':
            input_adj = np.zeros([5, 5])

            for i in range(25):
                v1 = inputs[i][:5]
                v2 = inputs[i][5:10]
                vertex1 = one_hot_to_idx(v1)
                vertex2 = one_hot_to_idx(v2)
                edge = inputs[i][-1]
                input_adj[vertex1][vertex2] = edge

            input_adj = np.asarray(input_adj)
            graph = nx.from_numpy_matrix(input_adj)
            nx.draw(graph, with_labels=True)
            plt.show()
            if np.all(labels[0] == 0):
                print("Label has no Eulerian")

            for j in range(20):
                if np.all(labels[j] == 0):
                    break
                node1 = np.where(np.asarray(labels[j]) == 1)[0][0]
                print(node1)

        print("output")
        if np.all(outputs[0] == 0):
            print("no Eulerian for ", 0)
            print("********")
            continue
        for k in range(20):
            if np.all(outputs[k] == 0):
                break
            node1 = np.where(np.asarray(outputs[j]) == 1)[0]
            print(node1, )
        print("********")
        pass
