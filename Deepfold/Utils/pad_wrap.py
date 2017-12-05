# Copyright 2017 Jes Frellsen and Wouter Boomsma. All Rights Reserved.
# Copyright (c) 2005, NumPy Developers
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

import collections
import tensorflow as tf
import numpy as np
from numpy.lib.arraypad import _validate_lengths

__all__ = ['pad_wrap']


_DummyArray = collections.namedtuple("_DummyArray", ["ndim"])


def _pad_wrap(arr, pad_amt, axis=-1):
    """
    Modified from numpy.lib.arraypad._pad_wrap
    """

    # Implicit booleanness to test for zero (or None) in any scalar type
    if pad_amt[0] == 0 and pad_amt[1] == 0:
        return arr

    ##########################################################################
    # Prepended region

    # Slice off a reverse indexed chunk from near edge to pad `arr` before
    start = arr.shape[axis] - pad_amt[0]
    end = arr.shape[axis]
    wrap_slice = tuple(slice(None) if i != axis else slice(start, end)
                       for (i, x) in enumerate(arr.shape))
    wrap_chunk1 = arr[wrap_slice]

    ##########################################################################
    # Appended region

    # Slice off a reverse indexed chunk from far edge to pad `arr` after
    wrap_slice = tuple(slice(None) if i != axis else slice(0, pad_amt[1])
                       for (i, x) in enumerate(arr.shape))
    wrap_chunk2 = arr[wrap_slice]

    # Concatenate `arr` with both chunks, extending along `axis`
    return tf.concat((wrap_chunk1, arr, wrap_chunk2), axis=axis)


def pad_wrap(array, pad_width):
    """
    Modified from numpy.lib.arraypad.wrap
    """

    if not np.asarray(pad_width).dtype.kind == 'i':
        raise TypeError('`pad_width` must be of integral type.')

    pad_width = _validate_lengths(_DummyArray(array.get_shape().ndims), pad_width)

    for axis, (pad_before, pad_after) in enumerate(pad_width):
        if array.get_shape().as_list()[axis] is None and (pad_before > 0 or pad_after > 0):
            raise TypeError('`pad_width` must be zero for dimensions that are None.')

    # If we get here, use new padding method
    newmat = tf.identity(array)

    for axis, (pad_before, pad_after) in enumerate(pad_width):
        # Recursive padding along any axis where `pad_amt` is too large
        # for indexing tricks. We can only safely pad the original axis
        # length, to keep the period of the reflections consistent.
        safe_pad = newmat.get_shape().as_list()[axis]

        if safe_pad is None:
            continue

        while ((pad_before > safe_pad) or
               (pad_after > safe_pad)):
            pad_iter_b = min(safe_pad,
                             safe_pad * (pad_before // safe_pad))
            pad_iter_a = min(safe_pad, safe_pad * (pad_after // safe_pad))
            newmat = _pad_wrap(newmat, (pad_iter_b, pad_iter_a), axis)

            pad_before -= pad_iter_b
            pad_after -= pad_iter_a
            safe_pad += pad_iter_b + pad_iter_a
        newmat = _pad_wrap(newmat, (pad_before, pad_after), axis)

    return newmat
