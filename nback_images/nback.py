from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
import numpy as np


class NBackImageData:

    def generate_batches(self, num_batches, batch_size, bits_per_vector=0, curriculum_point=6, max_seq_len=5,
                         curriculum='uniform', pad_to_max_seq_len=True, hasAblation=False):
        batches = []
        for i in range(num_batches):
            def generate_sequence(k):
                from_set = np.random.randint(0, mnist.train.images.shape[0] - max_seq_len)
                to_set = from_set + max_seq_len
                x_train = mnist.train.images[from_set:to_set, :]
                bits_in_vec = np.size(x_train[0])
                inp = np.insert(x_train, bits_in_vec, 0, axis=1)
                eos = np.ones(bits_in_vec + 1)
                input_with_eos = np.append(inp, [eos], axis=0)
                empty_arr = np.zeros_like(inp)
                full_input = np.concatenate((input_with_eos, empty_arr), axis=0)
                return full_input

            def generate_output(batch_input, m):
                batch_input_data = batch_input[m]
                output_first = batch_input_data[max_seq_len-3][:784]
                bits_in_vec = np.size(output_first)
                empty_arr = np.zeros([max_seq_len - 1, bits_in_vec])
                full_output = np.concatenate(([output_first], empty_arr), axis=0)
                return full_output
            batch_input = np.asarray([generate_sequence(k) for k in range(batch_size)])

            if hasAblation:
                random = np.random.random_sample((batch_size, 2 * max_seq_len + 1, 785))
                choice = random > 0.98
                selected_random = np.select(choice, random)
                batch_input = batch_input + selected_random
                batch_input = np.clip(batch_input, 0, 1)
            batch_output = np.asarray([generate_output(batch_input, m) for m in range(batch_size)])
            batches.append((max_seq_len, batch_input, batch_output))
        return batches

    def error_per_seq(self, labels, outputs, num_seq):
        bit_errors = np.sum(np.abs(labels - outputs))
        return bit_errors / num_seq


# nb = NBackImageData()
# a = nb.generate_batches(num_batches=10, batch_size=4,  max_seq_len=4, hasAblation=True)
# x_train = mnist.train.images[:10,:]
# y_train = mnist.train.labels[:10,:]
# label = y_train[0].argmax(axis=0)
# image = x_train[0].reshape([28,28])
# plt.title('Example: %d  Label: %d' % (0, label))
# plt.imshow(image, cmap=plt.get_cmap('gray_r'))
# # plt.show()
