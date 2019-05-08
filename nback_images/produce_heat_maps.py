import pickle
import matplotlib.pyplot as plt
import numpy as np
# matplotlib.use('macosx')
EXPERIMENT_NAME = 'nback_ntm_gpu'
MANN = 'NTM'
TASK = 'Nback'

HEAD_LOG_FILE = 'head_logs/{0}.p'.format(EXPERIMENT_NAME)
GENERALIZATION_HEAD_LOG_FILE = 'head_logs/generalization_{0}.p'.format(EXPERIMENT_NAME)

outputs = pickle.load(open(HEAD_LOG_FILE, "rb"))
outputs.update(pickle.load(open(GENERALIZATION_HEAD_LOG_FILE, "rb")))


for seq_len, heat_maps_list in outputs.items():
    for step, heat_maps in enumerate(heat_maps_list[-2:] if len(heat_maps_list) >= 2 else heat_maps_list):
        inputs = heat_maps['inputs'].T
        labels = heat_maps['labels'].T
        outputs = heat_maps['outputs'].T

        if TASK == 'Nback':
            inputs = inputs.T
            labels = labels.T
            outputs = outputs.T
            input_image = []
            output = outputs[0]
            output_np = np.asarray(output)
            output_image = output_np.reshape([28, 28])
            for input in inputs:
                arr = np.asarray(input[:784])
                image = arr.reshape([28, 28])
                input_image.append(image)
            input_image_np = np.array(input_image)
            for i in range(seq_len):
                plt.title('Example: %d  Label: %d' % (0, i))
                plt.imshow(input_image_np[i], cmap=plt.get_cmap('gray_r'))
                plt.show()
            plt.title('Output image ')
            plt.imshow(input_image_np[i], cmap=plt.get_cmap('gray_r'))
            plt.show()
            a =1
