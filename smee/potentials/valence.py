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
<<<<<<< HEAD
    smee.PotentialType.UREY_BRADLEY, smee.EnergyFn.BOND_HARMONIC
)
def compute_urey_bradley_energy(
=======
    smee.PotentialType.LINEAR_BONDS, smee.EnergyFn.BOND_LINEAR
)
def compute_linear_bond_energy(
>>>>>>> 0d0b42d2096b27ea518bc24c18979b40de5ce93e
    system: smee.TensorSystem,
    potential: smee.TensorPotential,
    conformer: torch.Tensor,
) -> torch.Tensor:
<<<<<<< HEAD
    """Compute the potential energy [kcal / mol] of Urey-Bradley 1-3 interactions
    for a given conformer using a harmonic potential of the form
    ``1/2 * k * (r - length) ** 2``.

    This term acts on 1-3 atom pairs extracted from angle terms.
=======
    """Compute the potential energy [kcal / mol] of a set of bonds for a given
    conformer using a linearized harmonic potential of the form
    ``1/2 * (k1+k2) * (r - (k1 * b1 + k2 * b2) / k) ** 2``
>>>>>>> 0d0b42d2096b27ea518bc24c18979b40de5ce93e

    Args:
        system: The system to compute the energy for.
        potential: The potential energy function to evaluate.
        conformer: The conformer [Å] to evaluate the potential at with
            ``shape=(n_confs, n_particles, 3)`` or ``shape=(n_particles, 3)``.

    Returns:
        The computed potential energy [kcal / mol].
    """
<<<<<<< HEAD

=======
>>>>>>> 0d0b42d2096b27ea518bc24c18979b40de5ce93e
    parameters = smee.potentials.broadcast_parameters(system, potential)
    particle_idxs = smee.potentials.broadcast_idxs(system, potential)

    _, distances = smee.geometry.compute_bond_vectors(conformer, particle_idxs)

<<<<<<< HEAD
    k = parameters[:, potential.parameter_cols.index("k")]
    length = parameters[:, potential.parameter_cols.index("length")]

    return (0.5 * k * (distances - length) ** 2).sum(-1)


@smee.potentials.potential_energy_fn(
    smee.PotentialType.HARMONIC_HEIGHT, smee.EnergyFn.HARMONIC_HEIGHT
)
def compute_harmonic_height_energy(
=======
    k1 = parameters[:, potential.parameter_cols.index("k1")]
    k2 = parameters[:, potential.parameter_cols.index("k2")]
    b1 = parameters[:, potential.parameter_cols.index("b1")]
    b2 = parameters[:, potential.parameter_cols.index("b2")]
    k0 = k1 + k2
    b0 = (k1 * b1 + k2 * b2) / k0
    return (0.5 * k0 * (distances - b0) ** 2).sum(-1)


@smee.potentials.potential_energy_fn(
    smee.PotentialType.LINEAR_ANGLES, smee.EnergyFn.ANGLE_LINEAR
)
def compute_linear_angle_energy(
>>>>>>> 0d0b42d2096b27ea518bc24c18979b40de5ce93e
    system: smee.TensorSystem,
    potential: smee.TensorPotential,
    conformer: torch.Tensor,
) -> torch.Tensor:
<<<<<<< HEAD
    """Compute the potential energy [kcal / mol] of harmonic pyramid-height improper
    interactions for a given conformer.

    The energy is ``1/2 * k * (h - h0) ** 2`` where *h* is the signed height of
    the central atom (p1) above the plane formed by its three neighbours
    (p2, p3, p4).
=======
    """Compute the potential energy [kcal / mol] of a set of valence angles for a given
        conformer using a linearized harmonic potential of the form
    ``1/2 * (k1+k2) * (r - (k1 * angle1 + k2 * angle2) / k) ** 2``
>>>>>>> 0d0b42d2096b27ea518bc24c18979b40de5ce93e

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
<<<<<<< HEAD

    h = smee.geometry.compute_pyramid_heights(conformer, particle_idxs)

    k = parameters[:, potential.parameter_cols.index("k")]
    h0 = parameters[:, potential.parameter_cols.index("h0")]

    return (0.5 * k * (h - h0) ** 2).sum(-1)


@smee.potentials.potential_energy_fn(
    smee.PotentialType.LEE_KRIMM, smee.EnergyFn.LEE_KRIMM
)
def compute_lee_krimm_energy(
    system: smee.TensorSystem,
    potential: smee.TensorPotential,
    conformer: torch.Tensor,
) -> torch.Tensor:
    """Compute the potential energy [kcal / mol] of Lee-Krimm improper interactions
    for a given conformer.

    The energy is ``V2 * phi**2 + V4 * phi**4`` where
    ``phi = |h|^t / (1 - |h|^s)`` and *h* is the signed pyramid height of the
    central atom above the plane of its three neighbours.

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

    h = smee.geometry.compute_pyramid_heights(conformer, particle_idxs)

    v2 = parameters[:, potential.parameter_cols.index("V2")]
    v4 = parameters[:, potential.parameter_cols.index("V4")]
    t = parameters[:, potential.parameter_cols.index("t")]
    s = parameters[:, potential.parameter_cols.index("s")]

    abs_h = torch.abs(h)
    phi = abs_h.pow(t) / (1.0 - abs_h.pow(s))

    return (v2 * phi**2 + v4 * phi**4).sum(-1)


@smee.potentials.potential_energy_fn(
    smee.PotentialType.HARMONIC_ANGLE_OOP, smee.EnergyFn.OOP_HARMONIC_ANGLE
)
def compute_oop_harmonic_angle_energy(
    system: smee.TensorSystem,
    potential: smee.TensorPotential,
    conformer: torch.Tensor,
) -> torch.Tensor:
    """Compute the potential energy [kcal / mol] of Wilson-Decius out-of-plane
    harmonic angle interactions for a given conformer.

    The energy is ``1/2 * k * (theta - theta0) ** 2`` where *theta* is the
    bond-plane angle (the angle between the out-of-plane bond and the plane of the
    remaining two bonds).

    Particle ordering per term: ``(p1_center, p2_oop, p3_plane, p4_plane)``.
    Three such terms are generated per improper center.

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

    theta = smee.geometry.compute_oop_angles(conformer, particle_idxs)

    k = parameters[:, potential.parameter_cols.index("k")]
    theta0 = parameters[:, potential.parameter_cols.index("theta0")]

    return (0.5 * k * (theta - theta0) ** 2).sum(-1)

=======
    theta = smee.geometry.compute_angles(conformer, particle_idxs)
    k1 = parameters[:, potential.parameter_cols.index("k1")]
    k2 = parameters[:, potential.parameter_cols.index("k2")]
    a1 = parameters[:, potential.parameter_cols.index("angle1")]
    a2 = parameters[:, potential.parameter_cols.index("angle2")]
    k0 = k1 + k2
    a0 = (k1 * a1 + k2 * a2) / k0
    return (0.5 * k0 * (theta - a0) ** 2).sum(-1)
>>>>>>> 0d0b42d2096b27ea518bc24c18979b40de5ce93e
