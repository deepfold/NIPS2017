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

"""Grid functionality stored in seperate module so it can be used
both by feature extraction and training code"""

import numpy as np
import Bio
import enum

# Enum for different typs of coordinate system
# DO NOT changes the numbers, as the number value is stored as a feature in files
CoordinateSystem = enum.Enum("CoordinateSystem", {"spherical":1, "cubed_sphere":2, "cartesian":3})
ZDirection = enum.Enum("ZDirection", {"sidechain":1, "backbone":2, "outward":3})


def get_spherical_grid_shape(max_radius, n_features, bins_per_angstrom):
    """Defines shape of grid matrix"""
    return (int(np.ceil(bins_per_angstrom*max_radius)),
            int(np.ceil(bins_per_angstrom*max_radius*np.pi)),
            int(np.ceil(bins_per_angstrom*max_radius*2*np.pi)),
            n_features)


def create_spherical_grid(max_radius, n_features, bins_per_angstrom):
    """Creates spherical grid"""

    grid_matrix = np.zeros(shape=get_spherical_grid_shape(max_radius, n_features, bins_per_angstrom))

    return grid_matrix


def get_cubed_sphere_grid_shape(max_radius, n_features, bins_per_angstrom):
    """Defines shape of grid matrix"""
    return (6,
            int(np.ceil(bins_per_angstrom*max_radius)),
            int(np.ceil(bins_per_angstrom*max_radius*np.pi/2)),
            int(np.ceil(bins_per_angstrom*max_radius*np.pi/2)),
            n_features)


def create_cubed_sphere_grid(max_radius, n_features, bins_per_angstrom):
    """Creates cubed sphere grid"""

    grid_matrix = np.zeros(shape=get_cubed_sphere_grid_shape(max_radius, n_features, bins_per_angstrom))

    return grid_matrix


def get_cartesian_grid_shape(max_radius, n_features, bins_per_angstrom):
    """Defines shape of grid matrix"""
    return 3*(int(np.ceil(2*bins_per_angstrom*max_radius)),) + (n_features,)


def create_cartesian_grid(max_radius, n_features, bins_per_angstrom):
    """Creates cubed sphere grid"""

    grid_matrix = np.zeros(shape=get_cartesian_grid_shape(max_radius, n_features, bins_per_angstrom))

    return grid_matrix


get_grid_shape_map = {CoordinateSystem.spherical: get_spherical_grid_shape,
                      CoordinateSystem.cubed_sphere: get_cubed_sphere_grid_shape,
                      CoordinateSystem.cartesian: get_cartesian_grid_shape}

create_grid_map = {CoordinateSystem.spherical: create_spherical_grid,
                   CoordinateSystem.cubed_sphere: create_cubed_sphere_grid,
                   CoordinateSystem.cartesian: create_cartesian_grid}


def define_coordinate_system(pos_N, pos_CA, pos_C, z_direction):
    """Defines a local reference system based on N, CA, and C atom positions"""
    
    # Define local coordinate system
    e1 = (pos_C-pos_N)
    e1 /= np.linalg.norm(e1)

    # Define CB positions by rotating N atoms around CA-C axis 120 degr
    pos_N_res = pos_N - pos_CA
    axis = pos_CA - pos_C
    pos_CB = np.dot(Bio.PDB.rotaxis((120./180.)*np.pi, Bio.PDB.Vector(axis)), pos_N_res)
    e2 = pos_CB
    e2 /= np.linalg.norm(e2)
    e3 = np.cross(e1, e2)

    # N-C and e2 are not perfectly perpendical to one another. We adjust e2.
    e2 = np.cross(e1, -e3)

    if z_direction == ZDirection.outward:
        # Use e3 as z-direction
         rot_matrix = np.array([e1,e2,e3])
    elif z_direction == ZDirection.backbone:
        # Use backbone direction as z-direction
        rot_matrix = np.array([e2,e3,e1])
    elif z_direction == ZDirection.sidechain:
        # Use sidechain direction as z-direction
        rot_matrix = np.array([e3,e1,e2])
    else:
        raise "Unknown z-direction "
    
    return rot_matrix


def cartesian_to_spherical_coordinates(xyz):
    """Convert set of Cartesian coordinates to spherical-polar coordinates"""

    # Convert to spherical coordinates
    xy = xyz[:,0]**2 + xyz[:,1]**2
    r = np.sqrt(xy + xyz[:,2]**2)
    theta = np.arctan2(np.sqrt(xy), xyz[:,2])  # polar angle - inclination from z-axis
    phi = np.arctan2(xyz[:,1], xyz[:,0])

    return r, theta, phi


def discretize_into_spherical_bins(r, theta, phi, max_r,
                                   r_shape, theta_shape, phi_shape):
    """Map r, theta, phi values to discrete grid bin"""

    # Bin each dimension independently
    r_boundaries = np.linspace(0, max_r, r_shape, endpoint=False)
    r_boundaries += (max_r - r_boundaries[-1])
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


def discretize_into_cubed_sphere_bins(patch, r, xi, eta,  max_r,
                                      r_shape, xi_shape, eta_shape):
    """Map r, theta, phi values to discrete grid bin"""
    
    # Bin each dimension independently
    r_boundaries = np.linspace(0, max_r, r_shape, endpoint=False)
    r_boundaries += (max_r - r_boundaries[-1])
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
    """Convert set of Cartesian coordinates to cubed-sphere coordinates"""

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


def discretize_into_cartesian_bins(xyz, max_radius, shape):
    """Map x,y,z values to discrete grid bin"""

    assert(len(shape) == 4)
    assert(shape[0] == shape[1] == shape[2])

    n_bins = shape[0]
    boundaries = np.linspace(-max_radius, max_radius, n_bins, endpoint=False)
    boundaries += (boundaries[1]-boundaries[0])

    indices = np.digitize(xyz, boundaries)

    return indices



if __name__ == "__main__":

    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    c = lambda t, a: np.array([np.cos(t), np.sin(t), -a*t]) / np.sqrt(1 + a**2 * t**2)
    p = np.transpose(np.array([c(t, .1) for t in np.linspace(-40, 40, 1000)]))

    ax.plot(p[0, :], p[1, :], p[2, :])

    fig.savefig("plot_3d.png")
    plt.close()

    p_cubed = [cartesian_to_cubed_sphere(*v) for v in np.transpose(p)]
    p_unfolded_plane = np.array([cubed_sphere_to_unfolded_plane(v[0], v[2], v[3]) for v in p_cubed])

    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal')

    for v in _offsets:
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
