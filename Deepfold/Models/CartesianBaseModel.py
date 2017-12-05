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


class CartesianBaseModel(BaseModel):

    @staticmethod
    def create_conv3d_layer(index,
                            input,
                            ksize_x,
                            ksize_y,
                            ksize_z,
                            channels_out,
                            stride_x=1,
                            stride_y=1,
                            stride_z=1,
                            use_padding=True):
        if use_padding:
            input = tf.pad(input, [(0, 0), (ksize_x/2, ksize_x/2), (ksize_y/2, ksize_y/2), (ksize_z/2, ksize_z/2), (0, 0)], "CONSTANT")

        filter_shape = [ksize_x, ksize_y, ksize_z, input.get_shape().as_list()[-1], channels_out]
        W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W%d" % index)
        b = tf.Variable(tf.truncated_normal(filter_shape[-1:], stddev=0.1), name="b%d" % index)

        conv = tf.nn.bias_add(
            tf.nn.conv3d(input,
                         W,
                         strides=[1, stride_x, stride_y, stride_z, 1],
                         padding="VALID",
                         name="cubed_sphere_conv%d" % index),
            b)

        return {'W': W, 'b': b, 'conv': conv}

    @staticmethod
    def create_avgpool_layer(index,
                             input,
                             ksize_x,
                             ksize_y,
                             ksize_z,
                             stride_x,
                             stride_y,
                             stride_z,
                             use_padding=True):

        if use_padding:
            input = tf.pad(input, [(0, 0), (ksize_x/2, ksize_x/2), (ksize_y/2, ksize_y/2), (ksize_z/2, ksize_z/2), (0, 0)], "CONSTANT")

        pool = tf.nn.avg_pool3d(input,
                                ksize=[1, ksize_x, ksize_y, ksize_z, 1],
                                strides=[1, stride_x, stride_y, stride_z, 1],
                                padding='VALID')

        return {'pool': pool}
