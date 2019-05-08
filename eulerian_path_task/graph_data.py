import networkx as nx
import numpy as np

import graph_utils


class GraphData:

    def generate_batches(self, num_batches, batch_size, bits_per_vector=0, curriculum_point=6, max_seq_len=6,
                         curriculum='uniform', pad_to_max_seq_len=False, num_vertices=5, ablation=True):
        batches = []
        for i in range(num_batches):
            graphs_generated = []

            def generate_sequence(k):
                graph = None
                if (np.random.random_sample() > 0.8):
                    isNotGen = True
                    while (isNotGen):
                        graph = nx.generators.erdos_renyi_graph(num_vertices, 0.5)
                        isNotGen = not graph_utils.is_semieulerian(graph)
                else:
                    graph = nx.generators.erdos_renyi_graph(num_vertices, 0.5)

                adj_matrix = nx.to_numpy_matrix(graph)
                adj_matrix = adj_matrix.tolist()
                graphs_generated.append(adj_matrix)

                x = 0
                final_input = []
                for row in adj_matrix:
                    z = 0
                    for elem in row:
                        x_onehot = np.zeros(max_seq_len)
                        x_onehot[x] = 1
                        z_onehot = np.zeros(max_seq_len)
                        z_onehot[z] = 1
                        edge_onehot = [0]
                        if elem == 1:
                            edge_onehot = [1]
                        edge_onehot = np.asarray(edge_onehot)
                        onevec = np.concatenate((x_onehot, z_onehot))
                        final_row = np.concatenate((onevec, edge_onehot))
                        final_input.append(final_row)
                        z += 1
                    x += 1
                input_padding = np.zeros([max_seq_len * max_seq_len - len(final_input), (2 * max_seq_len) + 1])
                eos = np.ones([2, (2 * max_seq_len) + 1])
                padding = np.zeros([max_seq_len * (max_seq_len - 1), (2 * max_seq_len) + 1])
                final_input = np.asarray(final_input)
                if ablation:
                    final_input = final_input + np.random.random_sample(final_input.shape)
                final_input = np.concatenate((final_input, input_padding, eos, padding))
                return final_input

            def generate_output(m):
                inputs = graphs_generated[m]
                path = graph_utils.findpath(inputs)
                output_vec = []
                if len(path) > 0:
                    for elem in path:
                        elem_onehot = np.zeros(max_seq_len)
                        elem_onehot[elem - 1] = 1
                        output_vec.append(elem_onehot)
                else:
                    output_vec = np.zeros([max_seq_len * (max_seq_len - 1), max_seq_len])
                output_vec = np.asarray(output_vec)
                end_padding_len = max_seq_len * (max_seq_len - 1) - np.shape(output_vec)[0]
                end_padding = np.zeros([end_padding_len, max_seq_len])
                output = np.concatenate((output_vec, end_padding))
                return output

            batch_input = np.asarray([generate_sequence(k) for k in range(batch_size)])
            batch_output = np.asarray([generate_output(m) for m in range(batch_size)])
            batches.append((max_seq_len * max_seq_len + 2, batch_input, batch_output))
        return batches

    @staticmethod
    def error_per_seq(labels, outputs, num_seq):
        outputs[outputs >= 0.5] = 1.0
        outputs[outputs < 0.5] = 0.0
        bit_errors = np.sum(np.abs(labels - outputs))
        return bit_errors / num_seq


pass
