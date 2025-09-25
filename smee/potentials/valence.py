"""Valence potential energy functions."""

import torch

import smee.geometry
import smee.potentials
import smee.utils


@smee.potentials.potential_energy_fn(
    smee.PotentialType.BONDS, smee.EnergyFn.BOND_HARMONIC
)
def compute_harmonic_bond_energy(
    system: smee.TensorSystem,
    potential: smee.TensorPotential,
    conformer: torch.Tensor,
) -> torch.Tensor:
    """Compute the potential energy [kcal / mol] of a set of bonds for a given
    conformer using a harmonic potential of the form ``1/2 * k * (r - length) ** 2``

    Args:
        system: The system to compute the energy for.
        potential: The potential energy function to evaluate.
        conformer: The conformer [Å] to evaluate the potential at with
            ``shape=(n_confs, n_particles, 3)`` or ``shape=(n_particles, 3)``.

    Returns:
        The computed potential energy [kcal / mol].
    """

    parameters = smee.potentials.broadcast_parameters(system, potential)
    particle_idxs = smee.potentials.broadcast_idxs(system, potential)

    _, distances = smee.geometry.compute_bond_vectors(conformer, particle_idxs)

    k = parameters[:, potential.parameter_cols.index("k")]
    length = parameters[:, potential.parameter_cols.index("length")]

    return (0.5 * k * (distances - length) ** 2).sum(-1)


@smee.potentials.potential_energy_fn(
    smee.PotentialType.ANGLES, smee.EnergyFn.ANGLE_HARMONIC
)
def compute_harmonic_angle_energy(
    system: smee.TensorSystem,
    potential: smee.TensorPotential,
    conformer: torch.Tensor,
) -> torch.Tensor:
    """Compute the potential energy [kcal / mol] of a set of valence angles for a given
    conformer using a harmonic potential of the form ``1/2 * k * (theta - angle) ** 2``

    Args:
        system: The system to compute the energy for.
        potential: The potential energy function to evaluate.
        conformer: The conformer [Å] to evaluate the potential at with
            ``shape=(n_confs, n_particles, 3)`` or ``shape=(n_particles, 3)``.

    Returns:
        The computed potential energy [kcal / mol].
    """

    parameters = smee.potentials.broadcast_parameters(system, potential)
    particle_idxs = smee.potentials.broadcast_idxs(system, potential)

    theta = smee.geometry.compute_angles(conformer, particle_idxs)

    k = parameters[:, potential.parameter_cols.index("k")]
    angle = parameters[:, potential.parameter_cols.index("angle")]

    return (0.5 * k * (theta - angle) ** 2).sum(-1)


def _compute_cosine_torsion_energy(
    system: smee.TensorSystem,
    potential: smee.TensorPotential,
    conformer: torch.Tensor,
) -> torch.Tensor:
    """Compute the potential energy [kcal / mol] of a set of torsions for a given
    conformer using a cosine potential of the form
    ``k/idivf*(1+cos(periodicity*phi-phase))``

    Args:
        system: The system to compute the energy for.
        potential: The potential energy function to evaluate.
        conformer: The conformer [Å] to evaluate the potential at with
            ``shape=(n_confs, n_particles, 3)`` or ``shape=(n_particles, 3)``.

    Returns:
        The computed potential energy [kcal / mol].
    """

    parameters = smee.potentials.broadcast_parameters(system, potential)
    particle_idxs = smee.potentials.broadcast_idxs(system, potential)

    phi = smee.geometry.compute_dihedrals(conformer, particle_idxs)

    k = parameters[:, potential.parameter_cols.index("k")]
    periodicity = parameters[:, potential.parameter_cols.index("periodicity")]
    phase = parameters[:, potential.parameter_cols.index("phase")]
    idivf = parameters[:, potential.parameter_cols.index("idivf")]

    return ((k / idivf) * (1.0 + torch.cos(periodicity * phi - phase))).sum(-1)


@smee.potentials.potential_energy_fn(
    smee.PotentialType.PROPER_TORSIONS, smee.EnergyFn.TORSION_COSINE
)
def compute_cosine_proper_torsion_energy(
    system: smee.TensorSystem,
    potential: smee.TensorPotential,
    conformer: torch.Tensor,
) -> torch.Tensor:
    """Compute the potential energy [kcal / mol] of a set of proper torsions
    for a given conformer using a cosine potential of the form:

    `k*(1+cos(periodicity*theta-phase))`

    Args:
        system: The system to compute the energy for.
        potential: The potential energy function to evaluate.
        conformer: The conformer [Å] to evaluate the potential at with
            ``shape=(n_confs, n_particles, 3)`` or ``shape=(n_particles, 3)``.

    Returns:
        The computed potential energy [kcal / mol].
    """
    return _compute_cosine_torsion_energy(system, potential, conformer)


@smee.potentials.potential_energy_fn(
    smee.PotentialType.IMPROPER_TORSIONS, smee.EnergyFn.TORSION_COSINE
)
def compute_cosine_improper_torsion_energy(
    system: smee.TensorSystem,
    potential: smee.TensorPotential,
    conformer: torch.Tensor,
) -> torch.Tensor:
    """Compute the potential energy [kcal / mol] of a set of improper torsions
    for a given conformer using a cosine potential of the form:

    `k*(1+cos(periodicity*theta-phase))`

    Args:
        system: The system to compute the energy for.
        potential: The potential energy function to evaluate.
        conformer: The conformer [Å] to evaluate the potential at with
            ``shape=(n_confs, n_particles, 3)`` or ``shape=(n_particles, 3)``.

    Returns:
        The computed potential energy [kcal / mol].
    """
    return _compute_cosine_torsion_energy(system, potential, conformer)


@smee.potentials.potential_energy_fn(
    "HarmonicHeight",
    "HarmonicHeight"
)
def compute_harmonic_height_improper_energy(
    system: smee.TensorSystem,
    potential: smee.TensorPotential,
    conformer: torch.Tensor,
) -> torch.Tensor:
    """Compute the potential energy [kcal / mol] of a set of impropers for a given
    conformer using a harmonic potential of the form
    ``1/2 * k * (h - h0) ** 2``.

    Args:
        system: The system to compute the energy for.
        potential: The potential energy function to evaluate.
        conformer: The conformer [Å] to evaluate the potential at with
            ``shape=(n_confs, n_particles, 3)`` or ``shape=(n_particles, 3)``.

    Returns:
        The computed potential energy [kcal / mol].
    """

    parameters = smee.potentials.broadcast_parameters(system, potential)
    particle_idxs = smee.potentials.broadcast_idxs(system, potential)

    a, b, c, d = (
        conformer[:, particle_idxs[:, 0], :],
        conformer[:, particle_idxs[:, 1], :],
        conformer[:, particle_idxs[:, 2], :],
        conformer[:, particle_idxs[:, 3], :],
    )

    v1 = b - c
    v2 = d - c

    n = torch.cross(v1, v2, dim=-1)

    ab = a - b

    h = torch.sum(n * ab, dim=-1) / torch.linalg.norm(n, dim=-1)

    k = parameters[:, potential.parameter_cols.index("k")]
    h0 = parameters[:, potential.parameter_cols.index("h0")]

    return (0.5 * k * (h - h0) ** 2).sum(-1)


@smee.potentials.potential_energy_fn(
    "LeeKrimm",
    "LeeKrimm"
)
def compute_lee_krimm_improper_energy(
    system: smee.TensorSystem,
    potential: smee.TensorPotential,
    conformer: torch.Tensor,
) -> torch.Tensor:
    """Compute the potential energy [kcal / mol] of a set of impropers for a given
    conformer using Lee Krimm functional form.

    Args:
        system: The system to compute the energy for.
        potential: The potential energy function to evaluate.
        conformer: The conformer [Å] to evaluate the potential at with
            ``shape=(n_confs, n_particles, 3)`` or ``shape=(n_particles, 3)``.

    Returns:
        The computed potential energy [kcal / mol].
    """

    parameters = smee.potentials.broadcast_parameters(system, potential)
    particle_idxs = smee.potentials.broadcast_idxs(system, potential)

    a, b, c, d = (
        conformer[:, particle_idxs[:, 0], :],
        conformer[:, particle_idxs[:, 1], :],
        conformer[:, particle_idxs[:, 2], :],
        conformer[:, particle_idxs[:, 3], :],
    )

    V2 = parameters[:, potential.parameter_cols.index("V2")]
    V4 = parameters[:, potential.parameter_cols.index("V4")]
    t  = parameters[:, potential.parameter_cols.index("t")]
    s  = parameters[:, potential.parameter_cols.index("s")]

    v1 = b - c
    v2 = d - c
    n = torch.cross(v1, v2, dim=-1)
    ab = a - b

    h = torch.sum(n * ab, dim=-1) / torch.linalg.norm(n, dim=-1)

    h_abs = torch.abs(h)
    denom = 1.0 - torch.pow(h_abs, s)
    safe_denom = torch.where(
        denom == 0,
        torch.tensor(1e-8, device=denom.device, dtype=denom.dtype),
        denom
    )

    frac = torch.pow(h_abs, t) / safe_denom
    energy = (V2 * frac**2) + (V4 * frac**4)

    return energy.sum(dim=-1)


@smee.potentials.potential_energy_fn(
    "HarmonicAngle", "HarmonicAngle"
)
def compute_harmonic_plane_angle_energy(
    system: smee.TensorSystem,
    potential: smee.TensorPotential,
    conformer: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the harmonic potential energy for the angle between two planes
    defined by four atoms (i, j, k, l).

    The potential has the form:
        1/2 * k * (theta - theta0)^2

    where theta is the dihedral angle between planes (i,j,k) and (j,k,l).

    Args:
        system: The system to compute the energy for.
        potential: The potential energy function to evaluate.
        conformer: The conformer [Å] with shape
            (n_confs, n_particles, 3) or (n_particles, 3).

    Returns:
        The computed potential energy [kcal / mol].
    """

    if conformer.ndim == 2:
        conformer = conformer.unsqueeze(0)

    parameters = smee.potentials.broadcast_parameters(system, potential)
    particle_idxs = smee.potentials.broadcast_idxs(system, potential)

    theta = smee.geometry.compute_dihedrals(conformer, particle_idxs)

    theta = torch.abs(theta)

    k = parameters[:, potential.parameter_cols.index("k")]
    theta0 = parameters[:, potential.parameter_cols.index("theta0")]

    return (0.5 * k * (theta - theta0) ** 2).sum(-1)
