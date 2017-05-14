import random
import tensorflow as tf

from src.agent import Agent
from src.environment import GymEnvironment, SimpleGymEnvironment
from config import get_config

flags = tf.app.flags

# Model
flags.DEFINE_string('model', 'm1', 'Type of model')
flags.DEFINE_boolean('dueling', False, 'Whether to use dueling deep q-network')
flags.DEFINE_boolean('double_q', False, 'Whether to use double q-learning')

# Environment
flags.DEFINE_string('env_name', 'Pong-v0', 'The name of gym environment to use')
flags.DEFINE_integer('action_repeat', 1, 'The number of action to be repeated')

# Optimizer
flags.DEFINE_float('decay', 0.99, 'Decay of RMSProp optimizer')
flags.DEFINE_float('epsilon', 0.1, 'Epsilon of RMSProp optimizer')
flags.DEFINE_float('momentum', 0.0, 'Momentum of RMSProp optimizer')
flags.DEFINE_float('beta', 0.01, 'Beta of RMSProp optimizer')

# Distributed
flags.DEFINE_string("ps_hosts", "0.0.0.0:2222", "Comma-separated list of hostname:port pairs")
flags.DEFINE_string("worker_hosts", "0.0.0.0:2223,0.0.0.0:2224", "Comma-separated list of hostname:port pairs")
flags.DEFINE_string("job_name", "", "One of 'ps', 'worker'")
flags.DEFINE_integer("task_index", 0, "Index of task within the job")

# Misc
flags.DEFINE_string('gpu_fraction', '1/1', 'idx / # of gpu fraction e.g. 1/3, 2/3, 3/3')
flags.DEFINE_boolean('display', False, 'Whether to do display the game screen or not')
flags.DEFINE_boolean('is_train', True, 'Whether to do training or testing')
flags.DEFINE_integer('random_seed', 123, 'Value of random seed')

FLAGS = flags.FLAGS

# Set random seed
tf.set_random_seed(FLAGS.random_seed)
random.seed(FLAGS.random_seed)

def main(_):
  config = get_config(FLAGS) or FLAGS
  config.cnn_format = 'NHWC'

  ps_hosts = FLAGS.ps_hosts.split(",")
  worker_hosts = FLAGS.worker_hosts.split(",")

  cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})
  server = tf.train.Server(cluster,
                           job_name=FLAGS.job_name,
                           task_index=FLAGS.task_index)

  if FLAGS.job_name == "ps":
    server.join()
  elif FLAGS.job_name == "worker":
    env = GymEnvironment(config)

    with tf.device(tf.train.replica_device_setter(
        worker_device="/job:worker/task:%d" % FLAGS.task_index,
        cluster=cluster)):
      lr_op = tf.placeholder('float', None, name='learning_rate')
      optimizer = tf.train.RMSPropOptimizer(
          lr_op, decay=0.99, momentum=0, epsilon=0.1)
      agent = Agent(config, env, optimizer, lr_op)

      agent.ep_end = random.sample([0.1, 0.01, 0.5], 1)[0]

    print(agent.model_dir)
    logdir = "./logs/" + agent.model_dir

    is_chief = (FLAGS.task_index == 0)
    if is_chief:
        agent.summary_writer = tf.summary.FileWriter(logdir)

    if FLAGS.is_train:
      if is_chief:
        train_or_play = agent.train_with_summary
      else:
        train_or_play = agent.train
    else:
      train_or_play = agent.play

    hooks = [
        tf.train.StopAtStepHook(last_step=config.max_step),
    ]
    scaffold = tf.train.Scaffold(saver=agent.saver)

    with tf.train.MonitoredTrainingSession(
            master=server.target,
            is_chief=is_chief,
            scaffold=scaffold,
            checkpoint_dir=logdir,
            save_summaries_steps=None,  # Disable default SummarySaverHook.
            hooks=hooks) as sess:
      agent.sess = sess
      agent.update_target_q_network()

      train_or_play(is_chief)

if __name__ == '__main__':
  tf.app.run()
