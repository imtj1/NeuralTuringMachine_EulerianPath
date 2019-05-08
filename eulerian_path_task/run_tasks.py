import logging
import pickle

import matplotlib.pyplot as plt
import tensorflow as tf
import argparse

from graph_data import GraphData

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


parser.add_argument('--mann', type=str, default='none', help='none | ntm')
parser.add_argument('--num_layers', type=int, default=1)
parser.add_argument('--num_units', type=int, default=100)
parser.add_argument('--num_memory_locations', type=int, default=128)
parser.add_argument('--memory_size', type=int, default=50)
parser.add_argument('--num_read_heads', type=int, default=1)
parser.add_argument('--num_write_heads', type=int, default=1)
parser.add_argument('--conv_shift_range', type=int, default=1, help='only necessary for ntm')
parser.add_argument('--clip_value', type=int, default=20, help='Maximum absolute value of controller and outputs.')
parser.add_argument('--init_mode', type=str, default='constant', help='learned | constant | random')

parser.add_argument('--optimizer', type=str, default='Adam', help='RMSProp | Adam')
parser.add_argument('--learning_rate', type=float, default=0.01)
parser.add_argument('--max_grad_norm', type=float, default=50)
parser.add_argument('--num_train_steps', type=int, default=2100)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--eval_batch_size', type=int, default=640)

parser.add_argument('--curriculum', type=str, default='none',
                    help='none | uniform | naive | look_back | look_back_and_forward | prediction_gain')
parser.add_argument('--pad_to_max_seq_len', type=str2bool, default=False)

parser.add_argument('--task', type=str, default='graph', help='copy | associative_recall | nback | graph')
parser.add_argument('--num_bits_per_vector', type=int, default=3)
parser.add_argument('--max_seq_len', type=int, default=5)

parser.add_argument('--store_result', type=str2bool, default=True, help='stores result')
parser.add_argument('--verbose', type=str2bool, default=False, help='if true prints lots of feedback')
parser.add_argument('--experiment_name', type=str, required=True)
parser.add_argument('--job-dir', type=str, required=False)
parser.add_argument('--steps_per_eval', type=int, default=100)
parser.add_argument('--use_local_impl', type=str2bool, default=True,
                    help='whether to use the repos local NTM implementation or the TF contrib version')
parser.add_argument('--ablation', type=str2bool, required=False)

args = parser.parse_args()

if args.mann == 'ntm':
    if args.use_local_impl:
        from ntm import NTMCell
    else:
        from tensorflow.contrib.rnn.python.ops.rnn_cell import NTMCell

HEAD_LOG_FILE = 'head_logs/{0}.p'.format(args.experiment_name)
GENERALIZATION_HEAD_LOG_FILE = 'head_logs/generalization_{0}.p'.format(args.experiment_name)


class BuildModel(object):
    def __init__(self, max_seq_len, inputs):
        self.max_seq_len = max_seq_len
        self.inputs = inputs
        self._build_model

    @property
    def _build_model(self):
        if args.use_local_impl:
            cell = NTMCell(args.num_layers, args.num_units, args.num_memory_locations, args.memory_size,
                           args.num_read_heads, args.num_write_heads, addressing_mode='content_and_location',
                           shift_range=args.conv_shift_range, reuse=False, output_dim=5,
                           clip_value=args.clip_value, init_mode=args.init_mode)
        else:
            def single_cell(num_units):
                return tf.contrib.rnn.BasicLSTMCell(num_units, forget_bias=1.0)

            controller = tf.contrib.rnn.MultiRNNCell(
                [single_cell(args.num_units) for _ in range(args.num_layers)])

            cell = NTMCell(controller, args.num_memory_locations, args.memory_size,
                           args.num_read_heads, args.num_write_heads, shift_range=args.conv_shift_range,
                           output_dim=5,
                           clip_value=args.clip_value)

        output_sequence, _ = tf.nn.dynamic_rnn(
            cell=cell,
            inputs=self.inputs,
            time_major=False,
            dtype=tf.float32,
            initial_state=None)

        self.output_logits = output_sequence[:, self.max_seq_len:, :]
        self.outputs = tf.sigmoid(self.output_logits)


class BuildTModel(BuildModel):
    def __init__(self, max_seq_len, inputs, outputs):
        super(BuildTModel, self).__init__(max_seq_len, inputs)

        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=outputs, logits=self.output_logits)
        self.loss = tf.reduce_sum(cross_entropy) / args.batch_size

        optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate)

        trainable_variables = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, trainable_variables), args.max_grad_norm)
        self.train_op = optimizer.apply_gradients(zip(grads, trainable_variables))


with tf.variable_scope('root'):
    max_seq_len_placeholder = tf.placeholder(tf.int32)
    inputs_placeholder = tf.placeholder(tf.float32, shape=(args.batch_size, None, 2 * args.max_seq_len + 1))
    outputs_placeholder = tf.placeholder(tf.float32, shape=(args.batch_size, None, args.max_seq_len))
    model = BuildTModel(max_seq_len_placeholder, inputs_placeholder, outputs_placeholder)
    initializer = tf.global_variables_initializer()

# training

convergence_on_target_task = None
convergence_on_multi_task = None
performance_on_target_task = None
performance_on_multi_task = None
generalization_from_target_task = None
generalization_from_multi_task = None

data_generator = GraphData()
target_point = args.max_seq_len
curriculum_point = 1 if args.curriculum not in ('prediction_gain', 'none') else target_point
progress_error = 1.0
convergence_error = 0.1

sess = tf.Session()
sess.run(initializer)

pickle.dump({target_point: []}, open(HEAD_LOG_FILE, "wb"))
pickle.dump({}, open(GENERALIZATION_HEAD_LOG_FILE, "wb"))


def run_eval(batches, store_heat_maps=False, generalization_num=None):
    task_loss = 0
    task_error = 0
    num_batches = len(batches)
    for seq_len, inputs, labels in batches:
        task_loss_, outputs = sess.run([model.loss, model.outputs],
                                       feed_dict={
                                           inputs_placeholder: inputs,
                                           outputs_placeholder: labels,
                                           max_seq_len_placeholder: seq_len
                                       })

        task_loss += task_loss_
        task_error += data_generator.error_per_seq(labels, outputs, args.batch_size)

    if store_heat_maps:
        if generalization_num is None:
            tmp = pickle.load(open(HEAD_LOG_FILE, "rb"))
            tmp[target_point].append({
                'labels': labels[0],
                'outputs': outputs[0],
                'inputs': inputs[0]
            })
            pickle.dump(tmp, open(HEAD_LOG_FILE, "wb"))
        else:
            tmp = pickle.load(open(GENERALIZATION_HEAD_LOG_FILE, "rb"))
            if tmp.get(generalization_num) is None:
                tmp[generalization_num] = []
            tmp[generalization_num].append({
                'labels': labels[0],
                'outputs': outputs[0],
                'inputs': inputs[0]
            })
            pickle.dump(tmp, open(GENERALIZATION_HEAD_LOG_FILE, "wb"))

    task_loss /= float(num_batches)
    task_error /= float(num_batches)
    return task_loss, task_error


def eval_performance(curriculum_point, store_heat_maps=False):
    # target task
    batches = data_generator.generate_batches(
        int(int(args.eval_batch_size / 2) / args.batch_size),
        args.batch_size,
        bits_per_vector=args.num_bits_per_vector,
        max_seq_len=args.max_seq_len,
        num_vertices=5,
        pad_to_max_seq_len=args.pad_to_max_seq_len

    )

    target_task_loss, target_task_error = run_eval(batches, store_heat_maps=store_heat_maps)

    # multi-task

    batches = data_generator.generate_batches(
        int(args.eval_batch_size / args.batch_size),
        args.batch_size,
        bits_per_vector=args.num_bits_per_vector,
        max_seq_len=args.max_seq_len,
        num_vertices=5,
        pad_to_max_seq_len=args.pad_to_max_seq_len
    )

    multi_task_loss, multi_task_error = run_eval(batches)

    return target_task_error, target_task_loss, multi_task_error, multi_task_loss


def eval_generalization(seq_lens, isMulti=False):
    res = []

    loss_y = []
    error_y = []
    for i in seq_lens:
        batches = data_generator.generate_batches(
            6,
            args.batch_size,
            bits_per_vector=args.num_bits_per_vector,
            num_vertices=i,
            max_seq_len=args.max_seq_len,
            pad_to_max_seq_len=False,
            ablation=args.ablation
        )
        loss, error = run_eval(batches, store_heat_maps=args.store_result, generalization_num=i)
        loss_y.append(loss)
        error_y.append(error)
        res.append(error)
    plt.plot(seq_lens, error_y, '-b', label='error')
    plt.plot(seq_lens, loss_y, '-r', label='loss')
    plt.plot([], [], ' ', label="Learning Rate = " + str(args.learning_rate))
    plt.plot([], [], ' ', label="Number of Layers = " + str(args.num_layers))
    plt.plot([], [], ' ', label="Number of Units = " + str(args.num_units))
    plt.plot([], [], ' ', label="Number of Training steps = " + str(args.num_train_steps))
    plt.plot([], [], ' ', label="Memory Size = " + str(args.memory_size))
    plt.legend(loc='upper right')
    plt.title("Generalization- Target Task")
    if isMulti:
        plt.title("Generalization - Multi Task")
    plt.xlabel("Seq length")
    plt.ylabel("Error/Loss")
    if isMulti:
        plt.savefig("MultiTask.png")
    else:
        plt.savefig("targetTask.png")

    plt.show()

    return res


training_error_x_axis = []
training_error_y_axis = []
training_loss_y_axis = []
for i in range(args.num_train_steps):
    seq_len, inputs, labels = data_generator.generate_batches(
        1,
        args.batch_size,
        bits_per_vector=args.num_bits_per_vector,
        max_seq_len=args.max_seq_len,
        num_vertices=5,
        pad_to_max_seq_len=args.pad_to_max_seq_len
    )[0]

    train_loss, _, outputs = sess.run([model.loss, model.train_op, model.outputs],
                                      feed_dict={
                                          inputs_placeholder: inputs,
                                          outputs_placeholder: labels,
                                          max_seq_len_placeholder: seq_len
                                      })

    avg_errors_per_seq = data_generator.error_per_seq(labels, outputs, args.batch_size)

    if args.verbose:
        logger.info('Train loss ({0}): {1}'.format(i, train_loss))
        logger.info('curriculum_point: {0}'.format(curriculum_point))
        logger.info('Average errors/sequence: {0}'.format(avg_errors_per_seq))
        logger.info('TRAIN_PARSABLE: {0},{1},{2},{3}'.format(i, curriculum_point, train_loss, avg_errors_per_seq))

    if i % args.steps_per_eval == 0:
        target_task_error, target_task_loss, multi_task_error, multi_task_loss \
            = eval_performance(curriculum_point if args.curriculum != 'prediction_gain' else None,
                               store_heat_maps=args.store_result)

        training_error_x_axis.append(i)
        training_error_y_axis.append(target_task_error)
        training_loss_y_axis.append(target_task_loss)

        if convergence_on_multi_task is None and multi_task_error < convergence_error:
            convergence_on_multi_task = i

        if convergence_on_target_task is None and target_task_error < convergence_error:
            convergence_on_target_task = i

        gen_evaled = False
        if convergence_on_multi_task is not None and (
                performance_on_multi_task is None or multi_task_error < performance_on_multi_task):
            performance_on_multi_task = multi_task_error
            generalization_from_multi_task = eval_generalization()
            gen_evaled = True

        if convergence_on_target_task is not None and (
                performance_on_target_task is None or target_task_error < performance_on_target_task):
            performance_on_target_task = target_task_error
            if gen_evaled:
                generalization_from_target_task = generalization_from_multi_task
            else:
                generalization_from_target_task = eval_generalization()

        logger.info('----EVAL----')
        logger.info('target task error/loss: {0},{1}'.format(target_task_error, target_task_loss))
        logger.info('multi task error/loss: {0},{1}'.format(multi_task_error, multi_task_loss))
        logger.info('EVAL_PARSABLE: {0},{1},{2},{3},{4}'.format(i, target_task_error, target_task_loss,
                                                                multi_task_error, multi_task_loss))

plt.plot(training_error_x_axis, training_error_y_axis, '-b', label='error')
plt.plot(training_error_x_axis, training_loss_y_axis, '-r', label='loss')
plt.plot([], [], ' ', label="Learning Rate = " + str(args.learning_rate))
plt.plot([], [], ' ', label="Number of Layers = " + str(args.num_layers))
plt.plot([], [], ' ', label="Number of Units = " + str(args.num_units))
plt.plot([], [], ' ', label="Number of Training steps = " + str(args.num_train_steps))
plt.plot([], [], ' ', label="Memory Size = " + str(args.memory_size))
plt.plot([], [], ' ', label="Seq Length = " + str(5))
plt.legend(loc='upper right')
plt.title("Training Error and loss")
plt.xlabel("Time")
# plt.yticks(np.arange(0, 101, 2.0))
plt.ylabel("Error/Loss")
plt.savefig("training.png")
plt.show()

performance_on_multi_task = multi_task_error
generalization_from_multi_task = eval_generalization([3, 4, 5], True)

performance_on_target_task = target_task_error
generalization_from_target_task = eval_generalization([3, 4, 5])

logger.info('----SUMMARY----')
logger.info('convergence_on_target_task: {0}'.format(convergence_on_target_task))
logger.info('performance_on_target_task: {0}'.format(performance_on_target_task))
logger.info('convergence_on_multi_task: {0}'.format(convergence_on_multi_task))
logger.info('performance_on_multi_task: {0}'.format(performance_on_multi_task))

logger.info('SUMMARY_PARSABLE: {0},{1},{2},{3}'.format(convergence_on_target_task, performance_on_target_task,
                                                       convergence_on_multi_task, performance_on_multi_task))

logger.info('generalization_from_target_task: {0}'.format(
    ','.join(map(str, generalization_from_target_task)) if generalization_from_target_task is not None else None))
logger.info('generalization_from_multi_task: {0}'.format(
    ','.join(map(str, generalization_from_multi_task)) if generalization_from_multi_task is not None else None))
