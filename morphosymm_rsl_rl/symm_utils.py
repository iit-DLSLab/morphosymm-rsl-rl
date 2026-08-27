# Created by Daniel Ordoñez (daniels.ordonez@gmail.com) at 14/03/25
from __future__ import annotations

import re
from typing import Sequence

import numpy as np
from escnn.group import Group, Representation, directsum


_HEIGHTMAP_NAME_RE = re.compile(r"^heightmap:(?P<rows>\d+)x(?P<cols>\d+)$")


def _parse_heightmap_shape(obs_name: str) -> tuple[int, int]:
    match = _HEIGHTMAP_NAME_RE.match(obs_name)
    if match is None:
        raise ValueError(
            "Heightmap observations must be named as 'heightmap:<rows>x<cols>', "
            f"for example 'heightmap:64x32'. Got {obs_name!r}."
        )

    rows = int(match.group("rows"))
    cols = int(match.group("cols"))
    if rows <= 0 or cols <= 0:
        raise ValueError(f"Heightmap dimensions must be positive. Got {obs_name!r}.")

    return rows, cols


def _heightmap_flat_index(row: int, col: int, rows: int) -> int:
    return col * rows + row


def _heightmap_signed_permutation_action(rep_Rd: Representation, group_element) -> np.ndarray:
    action = rep_Rd(group_element)[:2, :2]
    rounded_action = np.rint(action).astype(int)

    is_integral = np.allclose(action, rounded_action)
    is_signed_permutation = (
        np.all(np.isin(rounded_action, [-1, 0, 1]))
        and np.all(np.sum(np.abs(rounded_action), axis=0) == 1)
        and np.all(np.sum(np.abs(rounded_action), axis=1) == 1)
    )
    if not (is_integral and is_signed_permutation):
        raise NotImplementedError(
            "Heightmap observations only support signed-permutation horizontal actions. "
            f"Got action:\n{action}"
        )

    return rounded_action


def _heightmap_permutation(
    rows: int,
    cols: int,
    rep_Rd: Representation,
    group_element,
) -> np.ndarray:
    """Return dst indices for flattened src indices.

    Heightmaps are flattened in column-major order from the bottom-right corner:
    index 0 is row 0, col 0; index 1 is one row up in the same column.
    """
    action = _heightmap_signed_permutation_action(rep_Rd, group_element)

    row_coords = {2 * row - (rows - 1): row for row in range(rows)}
    col_coords = {2 * col - (cols - 1): col for col in range(cols)}
    permutation = np.empty(rows * cols, dtype=int)

    for row in range(rows):
        for col in range(cols):
            src_idx = _heightmap_flat_index(row, col, rows)
            src_coords = np.array([2 * row - (rows - 1), 2 * col - (cols - 1)], dtype=int)
            dst_row_coord, dst_col_coord = action @ src_coords

            if dst_row_coord not in row_coords or dst_col_coord not in col_coords:
                raise NotImplementedError(
                    "Heightmap action maps samples outside the declared grid. "
                    "Axis swaps require matching row/column coordinate sets."
                )

            dst_row = row_coords[dst_row_coord]
            dst_col = col_coords[dst_col_coord]
            permutation[src_idx] = _heightmap_flat_index(dst_row, dst_col, rows)

    return permutation


def _permutation_matrix(permutation: np.ndarray) -> np.ndarray:
    matrix = np.zeros((permutation.size, permutation.size))
    matrix[permutation, np.arange(permutation.size)] = 1.0
    return matrix


def _heightmap_representation(
    G: Group,
    rep_Rd: Representation,
    rows: int,
    cols: int,
    escnn_representation_form_mapping,
) -> Representation:
    permutations = {g: _heightmap_permutation(rows, cols, rep_Rd, g) for g in G.elements}
    full_representation = {g: _permutation_matrix(permutation) for g, permutation in permutations.items()}

    visited = np.zeros(rows * cols, dtype=bool)
    orbit_reps = []
    orbit_to_flat = np.zeros((rows * cols, rows * cols))
    orbit_basis_idx = 0
    orbit_rep_cache = {}

    for first_idx in range(rows * cols):
        if visited[first_idx]:
            continue

        orbit = sorted({permutation[first_idx] for permutation in permutations.values()})
        for idx in orbit:
            visited[idx] = True

        orbit_pos = {idx: pos for pos, idx in enumerate(orbit)}
        orbit_signature = tuple(
            tuple(orbit_pos[permutations[g][idx]] for idx in orbit)
            for g in G.elements
        )
        orbit_rep = orbit_rep_cache.get(orbit_signature)
        if orbit_rep is None:
            orbit_matrices = {
                g: _permutation_matrix(np.array([orbit_pos[permutations[g][idx]] for idx in orbit], dtype=int))
                for g in G.elements
            }
            orbit_rep = escnn_representation_form_mapping(
                G,
                orbit_matrices,
                name=f"heightmap_orbit_{len(orbit_rep_cache)}",
            )
            orbit_rep_cache[orbit_signature] = orbit_rep

        orbit_reps.append(orbit_rep)
        for idx in orbit:
            orbit_to_flat[idx, orbit_basis_idx + orbit_pos[idx]] = 1.0
        orbit_basis_idx += len(orbit)

    heightmap_orbit_rep = directsum(orbit_reps, name=f"heightmap:{rows}x{cols}:orbits")
    change_of_basis = orbit_to_flat @ heightmap_orbit_rep.change_of_basis
    change_of_basis_inv = heightmap_orbit_rep.change_of_basis_inv @ orbit_to_flat.T

    return Representation(
        G,
        name=f"heightmap:{rows}x{cols}",
        irreps=heightmap_orbit_rep.irreps,
        change_of_basis=change_of_basis,
        change_of_basis_inv=change_of_basis_inv,
        representation=full_representation,
    )


def configure_observation_space_representations(
    robot_name: str,
    obs_names: Sequence[str],
    joints_order: list,
) -> list[Group, dict[str, Representation]]:
    """Utility function to configure the morphological symmetries of a robot's observation space.

    Args:
        robot_name (str): Name of the robot listed in the morpho_symm package.
        obs_names (Sequence[str]): List of observation names to configure representations for.
            Each observation name will be matched with its corresponding (left) group representation.
        joints_order (list): Custom joint-space order if different from the one used by the package
            `morpho_symm`/`robot_descriptions.py`

    Returns:
        tuple: A tuple containing:
            - Group: The symmetry group (`escnn.group.Group`) of the robotic system.
            - dict: A dictionary mapping observation names to their corresponding representations.
    """
    try:
        import morpho_symm  # noqa: I001
        from morpho_symm.utils.rep_theory_utils import escnn_representation_form_mapping
        from morpho_symm.utils.robot_utils import load_symmetric_system
    except ImportError as e:
        raise ImportError("morpho_symm package is required to configure observation group representations") from e

    G = load_symmetric_system(robot_name=robot_name, return_robot=False, joint_space_order=joints_order)
    rep_Q_js = G.representations["Q_js"]  # Representation on joint space position coordinates
    rep_TqQ_js = G.representations["TqQ_js"]  # Representation on joint space velocity coordinates
    rep_Rd = G.representations["R3"]  # Representation on vectors in R^d
    rep_Rd_pseudo = G.representations["R3_pseudo"]  # Representation on pseudo vectors in R^d
    rep_euler_xyz = G.representations["euler_xyz"]  # Representation on Euler angles
    rep_kin_three = G.representations["kin_chain"]  # Permutation of legs
    rep_Rd_on_limbs = rep_kin_three.tensor(rep_Rd)  # Representation on signals R^d on the limbs
    rep_Rd_on_limbs.name = "Rd_on_limbs"
    rep_Rd_pseudo_on_limbs = rep_kin_three.tensor(rep_Rd_pseudo)  # Representation on pseudo vect R^d on the limbs
    rep_Rd_pseudo_on_limbs.name = "Rd_pseudo_on_limbs"
    rep_SO3_flat = {}
    for h in G.elements:
        rep_SO3_flat[h] = np.kron(rep_Rd(h), rep_Rd(~h).T)
    rep_SO3_flat = escnn_representation_form_mapping(G, rep_SO3_flat)
    rep_SO3_flat.name = "SO3_flat"

    rep_reflection_sign_flipping_scalar = escnn_representation_form_mapping(
        G,
        {g: np.array([[round(np.linalg.det(rep_Rd(g)))]], dtype=float) for g in G.elements},
        name="reflection_sign_flipping_scalar",
    )
    rep_invariant_scalar = escnn_representation_form_mapping(
        G,
        {g: np.ones((1, 1), dtype=float) for g in G.elements},
        name="invariant_scalar",
    )
    # rep_roll = escnn_representation_form_mapping(
    #     G, {g: rep_Rd_pseudo(g)[0:1, 0:1] for g in G.elements}, name="base_roll"
    # )
    # rep_pitch = escnn_representation_form_mapping(
    #     G, {g: rep_Rd_pseudo(g)[1:2, 1:2] for g in G.elements}, name="base_pitch"
    # )
    # rep_yaw = escnn_representation_form_mapping(
    #     G, {g: rep_Rd_pseudo(g)[2:3, 2:3] for g in G.elements}, name="base_yaw"
    # )

    rep_ctrl_commands_lin = escnn_representation_form_mapping(
        G, 
        {g: rep_Rd(g)[0:2, 0:2] for g in G.elements}, 
        name="ctrl_commands_lin_xy_dot"
    )
    rep_ctrl_commands_ang = escnn_representation_form_mapping(
        G, 
        {g: rep_Rd_pseudo(g)[2:3, 2:3] for g in G.elements}, 
        name="ctrl_commands_yaw_rate",
    )

    obs_reps = {}
    for obs_name in obs_names:
        # Generalized position, velocity, and force (torque) spaces
        if obs_name == "qpos":
            continue  # Quaternion does not have a left-group action definition.
        elif obs_name == "qvel":
            obs_reps[obs_name] = rep_Rd + rep_Rd_pseudo + rep_TqQ_js  # lin_vel , ang_vel, joint_vel
        elif obs_name == "joints_pos":  # Joint space position configuration
            obs_reps[obs_name] = rep_Q_js
        elif obs_name == "joints_vel":  # Joint space velocity configuration
            obs_reps[obs_name] = rep_TqQ_js
        elif obs_name == "base_pos":
            obs_reps[obs_name] = rep_Rd
        elif obs_name == "reflection_sign_flipping_scalar":
            obs_reps[obs_name] = rep_reflection_sign_flipping_scalar
        elif obs_name == "invariant_scalar":
            obs_reps[obs_name] = rep_invariant_scalar
        elif obs_name == "base_lin_vel" or obs_name == "base_lin_acc":  
            obs_reps[obs_name] = rep_Rd
        elif obs_name == "base_ang_vel":
            obs_reps[obs_name] = rep_Rd_pseudo
        elif obs_name == "base_ori_euler_xyz":
            obs_reps[obs_name] = rep_euler_xyz
        elif obs_name == "base_ori_quat_wxyz":
            continue  # Quaternion does not have a left-group action definition.
        elif obs_name == "base_ori_SO3":
            obs_reps[obs_name] = rep_SO3_flat
        elif obs_name == "feet_pos" or obs_name == "feet_vel":  # feet_pos:frame := feet_pos:world or feet_pos:base
            obs_reps[obs_name] = rep_Rd_on_limbs
        elif obs_name == "contact_state" or obs_name == "clock_data":
            obs_reps[obs_name] = rep_kin_three
        elif obs_name == "contact_forces":
            obs_reps[obs_name] = rep_Rd_on_limbs
        elif obs_name == "gravity" or obs_name == "imu_acc":
            obs_reps[obs_name] = rep_Rd
        elif obs_name == "imu_gyro":  # Same as angular velocity
            obs_reps[obs_name] = rep_Rd_pseudo
        elif obs_name == "des_base_lin_vel_xy":
            obs_reps[obs_name] = rep_ctrl_commands_lin
        elif obs_name == "des_base_ang_vel_yaw":
            obs_reps[obs_name] = rep_ctrl_commands_ang
        elif obs_name == "ctrl_commands":
            obs_reps[obs_name] = rep_ctrl_commands_lin + rep_ctrl_commands_ang
        elif "heightmap" in obs_name:
            heightmap_rows, heightmap_cols = _parse_heightmap_shape(obs_name)
            obs_reps[obs_name] = _heightmap_representation(
                G,
                rep_Rd,
                heightmap_rows,
                heightmap_cols,
                escnn_representation_form_mapping,
            )
        else:
            raise ValueError(f"Invalid observation name: {obs_name}")

    return G, obs_reps
