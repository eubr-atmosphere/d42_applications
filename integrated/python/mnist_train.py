import argparse
import sys
import tensorflow as tf


def train(_):
    import os
    os.environ['GRPC_POLL_STRATEGY'] = "poll"

    import tensorflow as tf

    ps_hosts = FLAGS.ps_hosts.split(",")
    worker_hosts = FLAGS.worker_hosts.split(",")
    cluster = tf.train.ClusterSpec({"ps":ps_hosts, "worker":worker_hosts})
    server_target='grpc://localhost:8000'
    job_name=FLAGS.job_name
    task_index=FLAGS.task_index
    
    # Defining the neural network architecture
    batch_size = 100
    learning_rate = 0.001
    training_epochs = 20
    logs_path = "/tmp/logs"

    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets("/mnt/share/vol001/MNIST_data/", one_hot=True)

    if job_name == "worker":
        with tf.device(tf.train.replica_device_setter(worker_device="/job:worker/task:%d" % task_index, 
                                                      cluster=cluster)):
            global_step = tf.get_variable('global_step', [], initializer = tf.constant_initializer(0), 
                                          trainable = False)
            x = tf.placeholder(tf.float32, shape=[None, 784], name="x-input")
            y_ = tf.placeholder(tf.float32, shape=[None, 10], name="y-input")

            W1 = tf.Variable(tf.random_normal([784, 100]))
            W2 = tf.Variable(tf.random_normal([100, 10]))
            b1 = tf.Variable(tf.zeros([100]))
            b2 = tf.Variable(tf.zeros([10]))

            z2 = tf.add(tf.matmul(x,W1),b1)
            a2 = tf.nn.sigmoid(z2)
            z3 = tf.add(tf.matmul(a2,W2),b2)
            y  = tf.nn.softmax(z3)

            cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
            grad_op = tf.train.GradientDescentOptimizer(learning_rate)
            train_op = grad_op.minimize(cross_entropy, global_step=global_step)
            correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

            tf.summary.scalar("cost", cross_entropy)
            tf.summary.scalar("accuracy", accuracy)
            summary_op = tf.summary.merge_all()
            init_op = tf.global_variables_initializer()

            sv = tf.train.Supervisor(is_chief=(task_index == 0), logdir=logs_path, 
                                     global_step = global_step, init_op=init_op)

            with sv.prepare_or_wait_for_session(server_target) as sess:
                writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
                for epoch in range (training_epochs):
                    batch_count = int(mnist.train.num_examples/batch_size)
                    for i in range(batch_count):
                        batch_x, batch_y = mnist.train.next_batch(batch_size)
                        _, cost, summary, step = sess.run([train_op, cross_entropy, summary_op,  global_step], 
                                                          feed_dict={x: batch_x, y_: batch_y})
                        writer.add_summary(summary, step)

                        if i%100 == 0:
                            print( " Epoch: %2d," % (epoch+1),
                                   " Batch:  %3d of %3d,"  % (i+1, batch_count),
                                   " Cost: %.4f,"  % cost)
            print ("Task %s %d ended" % (job_name, task_index))


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.register("type", "bool", lambda v: v.lower() == "true")
  # Flags for defining the tf.train.ClusterSpec
  parser.add_argument(
      "--ps_hosts",
      type=str,
      default="",
      help="Comma-separated list of hostname:port pairs"
  )
  parser.add_argument(
      "--worker_hosts",
      type=str,
      default="",
      help="Comma-separated list of hostname:port pairs"
  )
  parser.add_argument(
      "--job_name",
      type=str,
      default="",
      help="One of 'ps', 'worker'"
  )
  # Flags for defining the tf.train.Server
  parser.add_argument(
      "--task_index",
      type=int,
      default=0,
      help="Index of task within the job"
  )
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=train, argv=[sys.argv[0]] + unparsed)
