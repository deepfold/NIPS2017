# Copyright (c) 2017 Jes Frellsen and Wouter Boomsma. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

import os

import numpy as np
import tensorflow as tf

from Deepfold.batch_factory import get_batch


class BaseModel:
    """Base model with training and testing procedures"""

    def __init__(self,
                 reg_fact,
                 learning_rate,
                 model_checkpoint_path,
                 max_to_keep, *args, **kwargs):

        self.model_checkpoint_path = model_checkpoint_path

        # Define the model though subclass
        self._init_model(*args, **kwargs)

        # Set loss function
        self.entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.layers[-1]['dense'], labels=self.y)
        self.regularization = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if not v.name.startswith("b")]) * reg_fact

        self.loss = tf.reduce_mean(self.entropy) + self.regularization

        print "Number of parameters: ", sum(reduce(lambda x, y: x*y, v.get_shape().as_list()) for v in tf.trainable_variables())

        # Set the optimizer #
        self.train_step = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-08).minimize(self.loss)

        # Session and saver
        self.saver = tf.train.Saver(max_to_keep=max_to_keep)
        self.session = tf.Session()

        # Initialize variables
        tf.global_variables_initializer().run(session=self.session)
        print "Variables initialized"

    def _init_model(self, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def print_layer(layers, idx, name):
        """"Method for print architecture during model construction"""

        if layers[-1][name].get_shape()[0].value is None:
            size = int(np.prod(layers[-1][name].get_shape()[1:]))
        else:
            size = int(np.prod(layers[-1][name].get_shape()))

        print "layer %2d (high res) - %10s: %s [size %s]" % (len(layers), name, layers[idx][name].get_shape(), "{:,}".format(size))

    @staticmethod
    def create_dense_layer(index,
                           input,
                           output_size):
        """Method for creating a dense layer"""

        reshaped_input = tf.reshape(input, [-1, np.prod(input.get_shape().as_list()[1:])])

        if output_size == -1:
            output_size = reshaped_input.get_shape().as_list()[1]

        W = tf.Variable(tf.truncated_normal([reshaped_input.get_shape().as_list()[1], output_size], stddev=0.1), name="W%d" % index)
        b = tf.Variable(tf.truncated_normal([output_size], stddev=0.1), name="b%d" % index)
        dense = tf.nn.bias_add(tf.matmul(reshaped_input, W), b)

        return {'W': W, 'b': b, 'dense': dense}

    def train(self,
              train_batch_factory,
              num_passes=100,
              max_batch_size=1000,
              subbatch_max_size=25,
              validation_batch_factory=None,
              output_interval=10,
              dropout_keep_prob=0.5):

        print "dropout keep probability: ", dropout_keep_prob

        # Training loop
        iteration = 0

        with self.session.as_default():

            for i in range(num_passes):
                more_data = True

                while more_data:

                    batch, gradient_batch_sizes = train_batch_factory.next(max_batch_size,
                                                                           subbatch_max_size=subbatch_max_size,
                                                                           enforce_protein_boundaries=False)
                    more_data = (train_batch_factory.feature_index != 0)

                    grid_matrix = batch["high_res"]

                    labels = batch["model_output"]

                    for sub_iteration, (index, length) in enumerate(zip(np.cumsum(gradient_batch_sizes)-gradient_batch_sizes, gradient_batch_sizes)):

                        grid_matrix_batch, labels_batch = get_batch(index, index+length, grid_matrix, labels)

                        feed_dict = dict({self.x_high_res: grid_matrix_batch,
                                          self.y: labels_batch,
                                          self.dropout_keep_prob: dropout_keep_prob})

                        _, loss_value = self.session.run([self.train_step, self.loss], feed_dict=feed_dict)

                        print "[%d, %d, %02d] loss = %f" % (i, iteration, sub_iteration,  loss_value)

                    if (iteration+1) % output_interval == 0:
                        Q_training_batch, loss_training_batch = self.Q_accuracy_and_loss(batch, gradient_batch_sizes)
                        print "[%d, %d] Q%s score (training batch) = %f" % (i, iteration,  self.output_size, Q_training_batch)
                        print "[%d, %d] loss (training batch) = %f" % (i, iteration, loss_training_batch)

                        validation_batch, validation_gradient_batch_sizes = validation_batch_factory.next(validation_batch_factory.data_size(),
                                                                                                          subbatch_max_size=subbatch_max_size,
                                                                                                          enforce_protein_boundaries=False)
                        Q_validation, loss_validation = self.Q_accuracy_and_loss(validation_batch, validation_gradient_batch_sizes)
                        print "[%d, %d] Q%s score (validation set) = %f" % (i, iteration,  self.output_size, Q_validation)
                        print "[%d, %d] loss (validation set) = %f" % (i, iteration, loss_validation)

                        self.save(self.model_checkpoint_path, iteration)

                    iteration += 1

    def save(self, checkpoint_path, step):

        if not os.path.exists(checkpoint_path):
            os.mkdir(checkpoint_path)

        self.saver.save(self.session, os.path.join(checkpoint_path, 'model.ckpt'), global_step=step)
        print("Model saved")

    def restore(self, checkpoint_path, step=None):
        ckpt = tf.train.get_checkpoint_state(checkpoint_path)

        if ckpt and ckpt.model_checkpoint_path:
            if step is None or step == -1:
                print "Restoring from: last checkpoint"
                self.saver.restore(self.session, tf.train.latest_checkpoint(checkpoint_path))
            else:
                checkpoint_file = checkpoint_path+("/model.ckpt-%d" % step)
                print "Restoring from:", checkpoint_file
                self.saver.restore(self.session, checkpoint_file)
        else:
            print "Could not load file"

    def _infer(self, batch, gradient_batch_sizes, var, include_output=False):
        grid_matrix = batch["high_res"]

        if include_output:
            labels = batch["model_output"]

        results = []

        for index, length in zip(np.cumsum(gradient_batch_sizes)-gradient_batch_sizes, gradient_batch_sizes):
            grid_matrix_batch, = get_batch(index, index+length, grid_matrix)

            feed_dict = {self.x_high_res: grid_matrix_batch, self.dropout_keep_prob: 1.0}

            if include_output:
                labels_batch, = get_batch(index, index+length, labels)
                feed_dict[self.y] = labels_batch

            results.append(self.session.run(var, feed_dict=feed_dict))

        return results

    def infer(self, batch, gradient_batch_sizes):
        results = self._infer(batch, gradient_batch_sizes, var=self.layers[-1]['activation'], include_output=False)
        return np.concatenate(results)

    def Q_accuracy_and_loss(self, batch, gradient_batch_sizes, return_raw=False):
        y = batch["model_output"]
        y_argmax = np.argmax(y, 1)
        results = self._infer(batch, gradient_batch_sizes, var=[self.layers[-1]['dense'], self.entropy], include_output=True)

        y_, entropies = map(np.concatenate, zip(*results))

        predictions = np.argmax(y_, 1)
        identical = (predictions == y_argmax)

        Q_accuracy = np.mean(identical)

        regularization = self.session.run(self.regularization, feed_dict={})
        loss = np.mean(entropies) + regularization

        if return_raw:
            return loss, identical, entropies, regularization
        else:
            return Q_accuracy, loss