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

import tensorflow as tf

from BaseModel import BaseModel
import Deepfold.Ops as Ops


class SphericalBaseModel(BaseModel):
    @staticmethod
    def create_spherical_avgpool_layer(index,
                                       input,
                                       ksize,
                                       strides):

        pool = Ops.avg_pool_spherical(input, ksize, strides, padding='VALID', name="spherical_avg_pool%d" % (index))

        return {'pool': pool}

    @staticmethod
    def create_spherical_conv_layer(index,
                                    input,
                                    window_size_r,
                                    window_size_theta,
                                    window_size_phi,
                                    channels_out,
                                    stride_r=1,
                                    stride_theta=1,
                                    stride_phi=1,
                                    padding='VALID'):

        filter_shape = [window_size_r, window_size_theta, window_size_phi, input.get_shape().as_list()[-1],
                        channels_out]
        W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W_%d" % index)
        b = tf.Variable(tf.truncated_normal(filter_shape[-1:], stddev=0.1), name="bias_%d" % index)

        conv = Ops.conv_spherical(input=input,
                                  filter=W,
                                  strides=[1, stride_r, stride_theta, stride_phi, 1],
                                  padding=padding,
                                  name="conv_%d" % (index))

        output = tf.nn.bias_add(conv, b)

        return {'W': W, 'b': b, 'conv': output}
