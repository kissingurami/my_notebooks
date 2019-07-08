import tensorflow as tf


# Create computation graph
hello = tf.constant('Hello, TensorFlow!')


# Run computation
sess = tf.Session()
print(sess.run(hello))

