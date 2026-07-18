"""Tests for reusable acoustic PML helpers."""

from __future__ import annotations

import pytest
import torch

import edg_acoustics


def test_pml_region_and_quadratic_damping_masks_complete_elements():
    xyz = torch.tensor(
        [
            [[-0.5, 1.2, 0.0], [-0.4, 1.3, 0.0]],
            [[0.0, 0.0, 1.4], [0.1, 0.2, 1.5]],
        ],
        dtype=torch.float64,
    )
    region = edg_acoustics.PMLRegion((1.0, 1.0), mode="centered")
    damping = edg_acoustics.PMLDamping(amp_sigma=2.0, profile="quadratic")

    mask = region.element_mask(xyz)
    sigma = damping.compute(xyz, region)

    torch.testing.assert_close(mask, torch.tensor([False, True, True]))
    assert sigma.shape == xyz.shape
    torch.testing.assert_close(sigma[:, :, 0], torch.zeros_like(sigma[:, :, 0]))
    assert torch.count_nonzero(sigma[:, :, 1:]) > 0


def test_pml_damping_uses_mesh_region_when_provided():
    xyz = torch.zeros((2, 2, 3), dtype=torch.float64)
    xyz[0, :, 1] = 1.2
    xyz[1, :, 2] = 1.4
    region = edg_acoustics.PMLRegion(
        (1.0, 1.0),
        mode="ground",
        region_name="PML",
    )
    element_regions = {"PML": torch.tensor([2])}

    sigma = edg_acoustics.PMLDamping(amp_sigma=1.0).compute(
        xyz,
        region,
        element_regions,
    )

    torch.testing.assert_close(sigma[:, :, :2], torch.zeros_like(sigma[:, :, :2]))
    assert torch.count_nonzero(sigma[:, :, 2]) > 0


def test_pml_augmentation_matches_split_field_equations():
    sigma = torch.tensor(
        [
            [[0.0], [2.0]],
            [[3.0], [1.0]],
        ],
        dtype=torch.float64,
    )
    q = torch.tensor(
        [
            [[4.0], [5.0], [6.0]],
            [[7.0], [8.0], [9.0]],
        ],
        dtype=torch.float64,
    )
    memory = torch.tensor(
        [
            [[0.5], [0.25]],
            [[1.0], [0.75]],
        ],
        dtype=torch.float64,
    )
    acoustic_rhs = torch.tensor(
        [
            [[10.0], [11.0], [12.0]],
            [[13.0], [14.0], [15.0]],
        ],
        dtype=torch.float64,
    )
    rho_c2 = 2.0 * 3.0 * 3.0

    expected_rhs = acoustic_rhs.clone()
    expected_rhs[:, 0, :] -= (sigma[0] + sigma[1]) * q[:, 0, :]
    expected_rhs[:, 1, :] -= memory[:, 0, :] / rho_c2
    expected_rhs[:, 2, :] -= memory[:, 1, :] / rho_c2
    expected_memory_rhs = torch.empty_like(memory)
    expected_memory_rhs[:, 0, :] = (
        rho_c2 * (sigma[0] - sigma[1]) * expected_rhs[:, 1, :]
        - sigma[1] * memory[:, 0, :]
    )
    expected_memory_rhs[:, 1, :] = (
        rho_c2 * (sigma[1] - sigma[0]) * expected_rhs[:, 2, :]
        - sigma[0] * memory[:, 1, :]
    )

    actual_memory_rhs = torch.empty_like(memory)
    pml = edg_acoustics.PMLAugmentation(sigma, rho0=2.0, c0=3.0)
    returned_rhs, returned_memory_rhs = pml.apply_in_place(
        q,
        memory,
        acoustic_rhs,
        actual_memory_rhs,
    )

    torch.testing.assert_close(returned_rhs, expected_rhs)
    torch.testing.assert_close(returned_memory_rhs, expected_memory_rhs)
    torch.testing.assert_close(acoustic_rhs, expected_rhs)
    torch.testing.assert_close(actual_memory_rhs, expected_memory_rhs)


def test_pml_region_rejects_missing_region_name():
    xyz = torch.zeros((2, 1, 1), dtype=torch.float64)
    region = edg_acoustics.PMLRegion((1.0, 1.0), region_name="PML")

    with pytest.raises(ValueError, match="PML"):
        region.element_mask(xyz, {"Air": torch.tensor([0])})
