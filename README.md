# NeuralTuringMachine_EulerianPath
A neural net project for Eulerian path in Graphs


Run the Eulerian task from the euleraian_path_task folder
command to run 
`python run_tasks.py --experiment_name="Eulerian_path" --mann=ntm --task="graph"`


Run the Nback Image task from the nback_images folder
command to run 
`python run_tasks.py --experiment_name="N-BackImages" --mann=ntm --task="nback"`

Arguments that can be passed to change the hyper parameters are-

--mann', default='ntm', help='none | ntm'<br />
--num_layers', default=1<br />
--num_units', default=100<br />
--num_memory_locations', default=128<br />
--memory_size', default=20<br />
--num_read_heads', default=1<br />
--num_write_heads', default=1<br />
--conv_shift_range', default=1, help='only necessary for ntm'<br />
--clip_value', default=20, help='Maximum absolute value of controller and outputs.'<br />
--init_mode', default='constant', help='learned | constant | random'<br />

--optimizer', default='Adam', help='RMSProp | Adam'<br />
--learning_rate', default=0.01<br />
--max_grad_norm', default=50<br />
--num_train_steps', default=402<br />
--batch_size', default=32<br />
--eval_batch_size', default=640<br />

--curriculum', default='none',
                    help='none | uniform | naive | look_back | look_back_and_forward | prediction_gain'<br />
--pad_to_max_seq_len', default=False<br />

--task', default='nback', help='graph' | nback'<br />
--num_bits_per_vector', default=784<br />
--max_seq_len', default=5<br />

--store_result', default=True, help='stores result'<br />
--verbose', default=False, help='if true prints lots of feedback'<br />
--experiment_name', required=True<br />
--job-dir', required=False<br />
--steps_per_eval', default=200<br />
--use_local_impl', default=True,
                    help='whether to use the repos local NTM implementation or the TF contrib version'<br />
