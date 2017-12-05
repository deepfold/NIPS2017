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
import Deepfold.Ops as Ops

from BaseModel import BaseModel


class CubedSphereBaseModel(BaseModel):
    @staticmethod
    def create_cubed_sphere_conv_layer(index,
                                       input,
                                       ksize_r,
                                       ksize_xi,
                                       ksize_eta,
                                       channels_out,
                                       stride_r=1,
                                       stride_xi=1,
                                       stride_eta=1,
                                       use_r_padding=True):

        if use_r_padding:
            padding = "SAME"
        else:
            padding = "VALID"

        filter_shape = [ksize_r, ksize_xi, ksize_eta, input.get_shape().as_list()[-1], channels_out]
        W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W%d" % index)
        b = tf.Variable(tf.truncated_normal(filter_shape[-1:], stddev=0.1), name="b%d" % index)

        conv = Ops.conv_spherical_cubed_sphere(input=input,
                                               filter=W,
                                               strides=[1, stride_r, stride_xi, stride_eta, 1],
                                               padding=padding,
                                               name="cubed_sphere_conv%d" % (index))

        output = tf.nn.bias_add(conv, b)

        return {'W': W, 'b': b, 'conv': output}

    @staticmethod
    def create_cubed_sphere_conv_banded_disjoint_layer(index,
                                                       input,
                                                       ksize_r,
                                                       ksize_xi,
                                                       ksize_eta,
                                                       channels_out,
                                                       stride_r=1,
                                                       stride_xi=1,
                                                       stride_eta=1,
                                                       use_r_padding=True):
        return CubedSphereBaseModel.create_cubed_sphere_conv_banded_layer(index,
                                                                          input,
                                                                          ksize_r=ksize_r,
                                                                          ksize_xi=ksize_xi,
                                                                          ksize_eta=ksize_eta,
                                                                          channels_out=channels_out,
                                                                          kstride_r=1,
                                                                          kstride_xi=stride_xi,
                                                                          kstride_eta=stride_eta,
                                                                          window_size_r=ksize_r,
                                                                          window_stride_r=stride_r,
                                                                          use_r_padding=use_r_padding)

    @staticmethod
    def create_cubed_sphere_conv_banded_layer(index,
                                              input,
                                              ksize_r,
                                              ksize_xi,
                                              ksize_eta,
                                              channels_out,
                                              kstride_r,
                                              kstride_xi,
                                              kstride_eta,
                                              window_size_r,
                                              window_stride_r,
                                              use_r_padding):

        # Pad input with periodic image
        if use_r_padding:
            r_padding = (ksize_r / 2, ksize_r / 2)
        else:
            r_padding = (0, 0)

        padded_input = Ops.pad_cubed_sphere_grid(input,
                                                 r_padding=r_padding,
                                                 xi_padding=(ksize_xi / 2, ksize_xi / 2),
                                                 eta_padding=(ksize_eta / 2, ksize_eta / 2))

        # Create convolutions for each r value
        convs_r = []

        for i in range(0, padded_input.shape[2] - ksize_r + 1, window_stride_r):

            padded_input_r_band = padded_input[:, :, i:i+window_size_r, :, :, :]

            filter_shape = [ksize_r, ksize_xi, ksize_eta, padded_input_r_band.get_shape().as_list()[-1], channels_out]
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W_%d_r%d" % (index, i))
            b = tf.Variable(tf.truncated_normal(filter_shape[-1:], stddev=0.1), name="b_%d_r%d" % (index, i))

            convs_patches = []

            for patch in range(padded_input_r_band.get_shape().as_list()[1]):
                convs_patches.append(tf.nn.bias_add(
                    tf.nn.conv3d(padded_input_r_band[:, patch, :, :, :, :],
                                 W,
                                 strides=[1, kstride_r, kstride_xi, kstride_eta, 1],
                                 padding="VALID",
                                 name="cubed_sphere_conv_%d_r%d_p%d" % (index, i, patch)),
                    b))

            conv_r = tf.stack(convs_patches, axis=1, name="cubed_sphere_conv_%d_r%d" % (index, i))
            convs_r.append(conv_r)

        conv = tf.concat(convs_r, axis=2, name="cubed_sphere_conv%d" % (index))

        return {'conv': conv}

    @staticmethod
    def create_cubed_sphere_avgpool_layer(index,
                                          input,
                                          ksize_r,
                                          ksize_xi,
                                          ksize_eta,
                                          stride_r,
                                          stride_xi,
                                          stride_eta,
                                          use_r_padding=True):

        if use_r_padding:
            padding = "SAME"
        else:
            padding = "VALID"

        pool = Ops.avg_pool_spherical_cubed_sphere(input,
                                                   ksize=[1, ksize_r, ksize_xi, ksize_eta, 1],
                                                   strides=[1, stride_r, stride_xi, stride_eta, 1],
                                                   padding=padding,
                                                   name="cubed_sphere_pool%d" % (index))

        return {'pool': pool}
