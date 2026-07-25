"""Regression tests for three-dimensional receiver interpolation."""

from types import SimpleNamespace

import modepy
import numpy
import torch

from edg_acoustics.acoustics_simulation import AcousticsSimulation


def test_receiver_interpolation_uses_reference_coordinates_on_small_tetrahedron():
    """Physical metre coordinates must not be used as simplex basis coordinates."""
    vertices = numpy.array(
        [
            [0.20, -0.30, 0.70],
            [0.23, -0.29, 0.71],
            [0.21, -0.25, 0.72],
            [0.22, -0.28, 0.75],
        ],
        dtype=numpy.float64,
    )
    barycentric = numpy.array([0.1, 0.2, 0.3, 0.4], dtype=numpy.float64)
    receiver = (barycentric @ vertices).reshape(3, 1)

    simulation = object.__new__(AcousticsSimulation)
    simulation.mesh = SimpleNamespace(
        vertices=vertices.T,
        EToV=torch.tensor([[0], [1], [2], [3]], dtype=torch.long),
    )
    simulation.rec = receiver
    simulation.Nx = 4
    simulation.dim = 3
    simulation.device = torch.device("cpu")
    simulation.rst = modepy.warp_and_blend_nodes(3, simulation.Nx)
    basis = modepy.simplex_onb(simulation.dim, simulation.Nx)
    simulation.V = torch.as_tensor(
        modepy.vandermonde(basis, simulation.rst), dtype=torch.float64
    )

    weights, nodeindex = simulation.sample3D("brute_force")
    weights = weights.numpy()[0]
    rst = simulation.rst
    nodal_barycentric = numpy.vstack(
        (
            -(1.0 + rst.sum(axis=0)) / 2.0,
            (1.0 + rst[0]) / 2.0,
            (1.0 + rst[1]) / 2.0,
            (1.0 + rst[2]) / 2.0,
        )
    ).T
    nodal_coordinates = nodal_barycentric @ vertices
    nodal_values = (
        2.0
        + 3.0 * nodal_coordinates[:, 0]
        - 4.0 * nodal_coordinates[:, 1]
        + 5.0 * nodal_coordinates[:, 2]
    )
    expected = 2.0 + 3.0 * receiver[0, 0] - 4.0 * receiver[1, 0] + 5.0 * receiver[2, 0]

    assert nodeindex.tolist() == [0]
    assert numpy.isclose(weights.sum(), 1.0, rtol=0.0, atol=1.0e-12)
    assert numpy.isclose(weights @ nodal_values, expected, rtol=0.0, atol=1.0e-12)
