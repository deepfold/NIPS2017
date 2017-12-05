# Copyright 2017 Jes Frellsen and Wouter Boomsma. All Rights Reserved.
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


def pad_cubed_sphere_grid(tensor, r_padding=(0, 0), xi_padding=(0, 0), eta_padding=(0, 0), name=None):
    r"""Adds padding to the six faces of a tensor (dimension 1) to emulate the effect of a cubed sphere

    Args:
        tensor: A `Tensor`. Must be one of the following types: `float32`, `float64`.
            Shape `[batch, in_patch, in_r, in_theta, in_phi, in_channels]`, where patch
            denotes the 6 faces of the cube.
        r_padding: A `tuple` specifying padding sizes for the r dimension.
        xi_padding: A `tuple` specifying padding sizes for the xi dimension.
        eta_padding: A `tuple` specifying padding sizes for the eta dimension.
        name: A name for the operation (optional).
    """
    assert (xi_padding[0] > 0)
    assert (xi_padding[1] > 0)
    assert (eta_padding[0] > 0)
    assert (eta_padding[1] > 0)

    # Zero pad the tensor in the r dimension
    tensor = tf.pad(tensor, [(0, 0), (0, 0), r_padding, (0, 0), (0, 0), (0, 0)], "CONSTANT")

    # Transpose xi and eta axis
    tensorT = tf.transpose(tensor, [0, 1, 2, 4, 3, 5])

    # Pad xi left (0) and right (1)
    wrap_chunk0 = tf.stack([tensor[:, 3, :, -xi_padding[0]:, :, :],  # Patch 0
                            tensor[:, 0, :, -xi_padding[0]:, :, :],  # Patch 1
                            tensor[:, 1, :, -xi_padding[0]:, :, :],  # Patch 2
                            tensor[:, 2, :, -xi_padding[0]:, :, :],  # Patch 3
                            tf.reverse(tensorT[:, 3, :, -xi_padding[0]:, :, :], axis=[3]),  # Patch 4
                            tf.reverse(tensorT[:, 3, :, :xi_padding[0], :, :], axis=[2])],  # Patch 5
                           axis=1)

    wrap_chunk1 = tf.stack([tensor[:, 1, :, :xi_padding[1], :, :],  # Patch 0
                            tensor[:, 2, :, :xi_padding[1], :, :],  # Patch 1
                            tensor[:, 3, :, :xi_padding[1], :, :],  # Patch 2
                            tensor[:, 0, :, :xi_padding[1], :, :],  # Patch 3
                            tf.reverse(tensorT[:, 1, :, -xi_padding[1]:, :, :], axis=[2]),  # Patch 4
                            tf.reverse(tensorT[:, 1, :, :xi_padding[1], :, :], axis=[3])],  # Patch 5
                           axis=1)

    padded_tensor = tf.concat([wrap_chunk0, tensor, wrap_chunk1], axis=3)

    # Pad eta bottom (0) and top (1)
    wrap_chunk0 = tf.stack([tensor[:, 5, :, :, -eta_padding[0]:, :],    # Patch 0
                            tf.reverse(tensorT[:, 5, :, :, -eta_padding[0]:, :], axis=[2]),  # Patch 1
                            tf.reverse(tensor[:, 5, :, :, :eta_padding[0], :], axis=[2, 3]),  # Patch 2
                            tf.reverse(tensorT[:, 5, :, :, :eta_padding[0], :], axis=[3]),  # Patch 3
                            tensor[:, 0, :, :, -eta_padding[0]:, :],  # Patch 4
                            tf.reverse(tensor[:, 2, :, :, :eta_padding[0], :], axis=[2, 3])],  # Patch 5
                           axis=1)

    wrap_chunk1 = tf.stack([tensor[:, 4, :, :, :eta_padding[1], :],  # Patch 0
                            tf.reverse(tensorT[:, 4, :, :, -eta_padding[1]:, :], axis=[3]),  # Patch 1
                            tf.reverse(tensor[:, 4, :, :, -eta_padding[1]:, :], axis=[2, 3]),  # Patch 2
                            tf.reverse(tensorT[:, 4, :, :, :eta_padding[1], :], axis=[2]),  # Patch 3
                            tf.reverse(tensor[:, 2, :, :, -eta_padding[1]:, :], axis=[2, 3]),  # Patch 4
                            tensor[:, 0, :, :, :eta_padding[1], :]],  # Patch 5
                           axis=1)

    wrap_chunk0_padded = tf.pad(wrap_chunk0, [(0, 0), (0, 0), (0, 0), xi_padding, (0, 0), (0, 0)], "CONSTANT")
    wrap_chunk1_padded = tf.pad(wrap_chunk1, [(0, 0), (0, 0), (0, 0), xi_padding, (0, 0), (0, 0)], "CONSTANT")

    padded_tensor = tf.concat([wrap_chunk0_padded, padded_tensor, wrap_chunk1_padded], axis=4, name=name)

    return padded_tensor
