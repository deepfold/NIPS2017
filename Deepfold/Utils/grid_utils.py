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

'''Utility code for constructing spherical and cubed sphere grids'''

import numpy as np
import enum


def get_spherical_conv_grid_shape(max_radius, n_channels, bins_per_radial_unit):
    r"""Defines shape of spherical grid matrix

    Args:
        max_radius: A `float`. Specifies the extent of the radial dimension.
        n_channels: An `int`. Specifies the number of channels.
        bins_per_radial_unit: An `int`. Specifies the bin density in the
            radial dimension.

    Returns:
        A `tuple`. 
    """
    return (int(np.ceil(bins_per_radial_unit*max_radius)),
            int(np.ceil(bins_per_radial_unit*max_radius*np.pi)),
            int(np.ceil(bins_per_radial_unit*max_radius*2*np.pi)),
            n_channels)


def create_spherical_conv_grid(max_radius, n_features, bins_per_radial_unit):
    r"""Creates a spherical convolution grid.

    Args:
        max_radius: A `float`. Specifies the extent of the radial dimension.
        n_channels: An `int`. Specifies the number of channels.
        bins_per_radial_unit: An `int`. Specifies the bin density in the
            radial dimension.

    Returns:
        A `numpy.array` of zeros. 
    """

    grid_matrix = np.zeros(shape=get_spherical_grid_shape(max_radius, n_features, bins_per_radial_unit))

    return grid_matrix


def discretize_into_spherical_grid_bins(r, theta, phi, max_r,
                                        nbins_r, nbins_theta, nbins_phi):
    r"""Map spherical coordinates to corresponding bins

    Args:
        r: `float`. Radial value.
        theta: `float`. Polar angle.
        phi: `float`. Azimuthal angle.
        max_r: `float`. The maximum value in the radial dimension.
        nbins_r: `int`. Number of bins in the radial dimension.
        nbins_theta: `int`. Number of bins in the polar angle dimension.
        nbins_phi: `int`. Number of bins in the azimuthal angledimension.

    Returns:
        A `tuple` of r,theta,phi values. 

    """

    # Bin each dimension independently
    r_boundaries = np.linspace(0, max_r, r_shape, endpoint=False)
    r_boundaries += (r_boundaries[1]-r_boundaries[0])
    theta_boundaries = np.linspace(0, np.pi, theta_shape, endpoint=False)
    theta_boundaries += (theta_boundaries[1]-theta_boundaries[0])
    phi_boundaries = np.linspace(-np.pi, np.pi, phi_shape, endpoint=False)
    phi_boundaries += (phi_boundaries[1]-phi_boundaries[0])
    r_bin = np.digitize(r, r_boundaries)
    theta_bin = np.digitize(theta, theta_boundaries)
    phi_bin = np.digitize(phi, phi_boundaries)

    # For phi angle, check for periodicity issues
    # When phi=pi, it will be mapped to the wrong bin
    phi_bin[phi_bin == phi_shape] = 0

    # Disallow any larger phi angles 
    assert(not np.any(phi_bin > phi_shape))
    assert(not np.any(theta_bin > theta_shape))
    
    return r_bin, theta_bin, phi_bin


def cartesian_to_spherical_coordinates(xyz):
    """Convert Cartesian to spherical coordinates

    Args:
        xyz: a (?,3) `numpy array` of Cartesian values

    Returns:
        The radial, polar and azimuthal components, each in a `numpy.array`.
    """

    xy = xyz[:,0]**2 + xyz[:,1]**2
    r = np.sqrt(xy + xyz[:,2]**2)
    theta = np.arctan2(np.sqrt(xy), xyz[:,2])  # polar angle - inclination from z-axis
    phi =  np.arctan2(xyz[:,1], xyz[:,0])

    return r, theta, phi


def get_cubed_sphere_conv_grid_shape(max_radius, n_channels, bins_per_radial_unit):
    r"""Defines shape of cubed-sphere grid matrix

    Args:
        max_radius: A `float`. Specifies the extent of the radial dimension.
        n_channels: An `int`. Specifies the number of channels.
        bins_per_radial_unit: An `int`. Specifies the bin density in the
            radial dimension.

    Returns:
        A `tuple`. 
    """
    return (6,
            int(np.ceil(bins_per_radial_unit*max_radius)),
            int(np.ceil(bins_per_radial_unit*max_radius*np.pi/2)),
            int(np.ceil(bins_per_radial_unit*max_radius*np.pi/2)),
            n_channels)


def create_cubed_sphere_conv_grid(max_radius, n_features, bins_per_radial_unit):
    r"""Creates a cubed-sphere convolution grid.

    Args:
        max_radius: A `float`. Specifies the extent of the radial dimension.
        n_channels: An `int`. Specifies the number of channels.
        bins_per_radial_unit: An `int`. Specifies the bin density in the
            radial dimension.

    Returns:
        A `numpy.array` of zeros. 
    """

    grid_matrix = np.zeros(shape=get_cubed_sphere_conv_grid_shape(max_radius, n_features, bins_per_radial_unit))

    return grid_matrix


def discretize_into_cubed_sphere_grid_bins(patch, r, xi, eta,  max_r,
                                           nbins_r, nbins_xi, nbins_eta):
    r"""Map spherical coordinates to corresponding bins

    Args:
        patch: `int`. Index of cube-face.
        r: `float`. Radial value.
        xi: `float`. Value in Xi dimension.
        phi: `float`. Value in eta dimension.
        max_r: `float`. The maximum value in the radial dimension.
        nbins_r: `int`. Number of bins in the radial dimension.
        nbins_xi: `int`. Number of bins in the polar angle dimension.
        nbins_eta: `int`. Number of bins in the azimuthal angledimension.

    Returns:
        A `tuple` of patch,r,xi,theta values. 

    """
    # Bin each dimension independently
    r_boundaries = np.linspace(0, max_r, r_shape, endpoint=False)
    r_boundaries += (r_boundaries[1]-r_boundaries[0])
    xi_boundaries = np.linspace(-np.pi/4, np.pi/4, xi_shape, endpoint=False)
    xi_boundaries += (xi_boundaries[1]-xi_boundaries[0])
    eta_boundaries = np.linspace(-np.pi/4, np.pi/4, eta_shape, endpoint=False)
    eta_boundaries += (eta_boundaries[1]-eta_boundaries[0])
    r_bin = np.digitize(r, r_boundaries)
    xi_bin = np.digitize(xi, xi_boundaries)
    eta_bin = np.digitize(eta, eta_boundaries)

    # Disallow any larger xi, eta angles
    assert(not np.any(r_bin < 0))
    assert(not np.any(xi_bin < 0))
    assert(not np.any(eta_bin < 0))
    assert(not np.any(xi_bin > xi_shape))
    assert(not np.any(eta_bin > eta_shape))

    return patch, r_bin, xi_bin, eta_bin


def cartesian_to_cubed_sphere(x, y, z, rtol=1e-05):
    r"""Convert Cartesian to spherical coordinates

    Args:
        x: `float`. X-value.
        y: `float`. Y-value.
        z: `float`. Z-value.
        rtol: `float`. Value beneath which r values are considered zero.

    Returns:
        A `tuple` of patch, r, xi, eta values. 

    """

    r = np.sqrt(x**2 + y**2 + z**2)

    if r < rtol:
        patch = 0
        xi = 0.
        eta = 0.

    elif x >= np.abs(y) and x >= np.abs(z):
        # Front patch (I)
        patch = 0
        xi = np.arctan(y/x)
        eta = np.arctan(z/x)

    elif y >= np.abs(x) and y >= np.abs(z):
        # East patch (II)
        patch = 1
        xi = np.arctan(-x/y)
        eta = np.arctan(z/y)

    elif -x >= np.abs(y) and -x >= np.abs(z):
        # Back patch (III)
        patch = 2
        xi = np.arctan(y/x)
        eta = np.arctan(-z/x)

    elif -y >= np.abs(x) and -y >= np.abs(z):
        # West  patch (IV)
        patch = 3
        xi = np.arctan(-x/y)
        eta = np.arctan(-z/y)

    elif z >= np.abs(x) and z >= np.abs(y):
        # North patch (V)
        patch = 4
        xi = np.arctan(y/z)
        eta = np.arctan(-x/z)

    elif -z >= np.abs(x) and -z >= np.abs(y):
        # South pathc (VI)
        patch = 5
        xi = np.arctan(-y/z)
        eta = np.arctan(-x/z)

    else:
        raise ArithmeticError("Should never happen")

    return patch, r, xi, eta

# Vectorized version of cartesian_to_cubed_sphere
cartesian_to_cubed_sphere_vectorized = np.vectorize(cartesian_to_cubed_sphere, otypes=[np.int, np.float, np.float, np.float])


def cubed_sphere_to_unfolded_plane(patch, xi, eta, offsets=np.array([[1, 1], [2, 1], [3, 1], [0, 1], [1, 2], [1, 0]])):
    r"""Unfold points on the cubed sphere into a plane.

    Args:
        patch: `int`. Cube face.
        xi: `int`. Xi value.
        eta: `int`. Eta value.
        offsets: `numpy.array` of X,Y offsets


    Returns:
        A `numpy.array` of points in the XY plane.

    """
    return offsets[patch] + (np.array([xi, eta]) + np.pi/4) / (np.pi/2)



if __name__ == "__main__":

    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Create and plot function on the sphere
    c = lambda t, a: np.array([np.cos(t), np.sin(t), -a*t]) / np.sqrt(1 + a**2 * t**2)
    p = np.transpose(np.array([c(t, .1) for t in np.linspace(-40, 40, 1000)]))
    ax.plot(p[0, :], p[1, :], p[2, :])
    fig.savefig("plot_3d.png")
    plt.close()


    # Plot same function in unfolded representation
    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal')

    p_cubed = [cartesian_to_cubed_sphere(*v) for v in np.transpose(p)]
    p_unfolded_plane = np.array([cubed_sphere_to_unfolded_plane(v[0], v[2], v[3]) for v in p_cubed])

    offsets=np.array([[1, 1], [2, 1], [3, 1], [0, 1], [1, 2], [1, 0]])
    for v in offsets:
        ax.add_patch(matplotlib.patches.Rectangle(v, 1., 1., fill=False))

    colors = plt.get_cmap("hsv")(np.linspace(0, 1, p_unfolded_plane.shape[0]-1))
        
    for i in xrange(p_unfolded_plane.shape[0]-1):
        if p_cubed[i][0] == p_cubed[i+1][0]:
            linestyle = '-'
        else:
            linestyle = ':'

        ax.plot(p_unfolded_plane[i:i+2, 0], p_unfolded_plane[i:i+2, 1], color=colors[i], linestyle=linestyle)
    
        
    plt.axis('off')
    e = 0.1
    ax.set_xlim(0-e, 4+e)
    ax.set_ylim(0-e, 3+e)
    
    fig.savefig("plot_2d.png")
    plt.close()
