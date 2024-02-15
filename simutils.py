import numpy as np
import pandas as pd
from skimage.segmentation import find_boundaries
from tqdm.auto import tqdm


def move_without_collision(
    position, displacement, boundaries, boundary_mask, touching_distance
):
    """recursive function to move a particle

    NOTE: position should be an array with shape (1,2)
    the displacement should just be a vector with shape (2,)
    """
    dists = (
        np.linalg.norm(boundaries - (position + displacement), axis=1)
        - touching_distance
    )

    clash = np.any(dists < 0)

    proposed_position = position + displacement

    # check if next position is within 'image'
    Ny, Nx = boundary_mask.shape
    if (1 < proposed_position[0, 0] < (Nx - 1)) and (
        1 < proposed_position[0, 1] < (Ny - 1)
    ):
        # round to nearest neighbor
        rx, ry = (np.round(proposed_position)[0]).astype(int)
        out_of_bounds = boundary_mask[ry, rx]
    else:
        out_of_bounds = True

    # collision detected
    if clash or out_of_bounds:
        # backtrack to find collision point
        alpha_low = 0.0
        alpha_high = 1.0
        prev_alpha = 1.0
        rel_alpha = 1.0

        while abs(rel_alpha) > 1e-3:
            alpha_mid = (alpha_high + alpha_low) / 2.0
            rel_alpha = prev_alpha - alpha_mid
            trial_step = position + alpha_mid * displacement

            dists = (
                np.linalg.norm(boundaries - trial_step, axis=1)
                - touching_distance
            )

            clash = np.any(dists < 0)

            if clash:
                alpha_high = alpha_mid
            else:
                alpha_low = alpha_mid

            prev_alpha = alpha_mid

        # compute 'point-of-contact' step length
        alpha = (alpha_high + alpha_low) / 2.0
        stopping_position = position + alpha * displacement

        dists = (
            np.linalg.norm(boundaries - stopping_position, axis=1)
            - touching_distance
        )

        closest_boundary_id = np.argmin(dists)
        boundary_point = boundaries[closest_boundary_id]

        # now compute normal vector
        collision_normal = boundary_point - stopping_position[0]
        normalized_normal = collision_normal / np.linalg.norm(collision_normal)

        # remainder vector
        remainder = displacement - (alpha * displacement)
        # reflect the remainder vectro
        reflected = (
            remainder
            - 2 * np.dot(remainder, normalized_normal) * normalized_normal
        )

        # compute recursion on new position
        return move_without_collision(
            stopping_position,
            reflected,
            boundaries,
            boundary_mask,
            touching_distance,
        )

    else:
        # no collision, just return the passed parameters
        return (
            position,
            displacement,
            boundaries,
            boundary_mask,
            touching_distance,
        )


def run_simulation(
    init_pos,
    boundary_mask,
    n_steps=99,
    D=1.0,
    dt=0.02,
    boundary_radius=0.5,
    particle_radius=0.8,
):
    """run constrained simulation within arbitrary mask

    init_pos(np.array): initial particle positions (npts, 2)
    boundary_mask(np.array): binary array of mask. 1 is a 'wall' and 0 is
    'empty'
    n_steps(int): number of steps to simulate
    D(float): diffusivity in pixel^2/seconds
    dt(float): time increment
    boundary_radius(float): radius of boundary, 0.5 by default.
    particle_radius(float): radius of diffusing particle, 0.8 by default.
    """
    touching_distance = boundary_radius + particle_radius
    npts = init_pos.shape[0]

    # compute boundary coordinates
    boundary_pixels = find_boundaries(boundary_mask, mode="outer")
    by, bx = np.where(boundary_pixels)
    boundary_xy = np.vstack([bx, by]).T

    tracks = []
    for p in tqdm(range(npts)):
        coordinates = np.zeros((n_steps + 1, 2))
        coordinates[0] = init_pos[p]

        for n in range(n_steps):
            step = np.random.randn(2) * np.sqrt(2 * D * dt)
            pos = np.array([coordinates[n, :]])
            _pos, _step, *_ = move_without_collision(
                pos, step, boundary_xy, boundary_mask, touching_distance
            )
            coordinates[n + 1, :] = _pos + _step

        _df = pd.DataFrame(
            {
                "particle": p + 1,
                "frames": np.arange(n_steps + 1),
                "y": coordinates[:, 1],
                "x": coordinates[:, 0],
            }
        )

        tracks.append(_df)

    return pd.concat(tracks, axis=0, ignore_index=True)
