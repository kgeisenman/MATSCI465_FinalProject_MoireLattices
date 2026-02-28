"""

This Python script contains a function generate_twisted_bilayers.
It returns an ASE Atoms object which can be used for visualization or for implementation in multislice simulations.

"""

import numpy as np
import ase
import abtem

def crop_center_square(structure, shrink_factor=0.8):
    """
    Crops the supercell of an Atoms object by scaling it down by a user-specified factor.

    Parameters:
        - structure (Atoms object in ASE): Input structure
        - shrink_factor (float): Percent factor that the original structure will be shrunken down to

    Returns:
        - cropped (Atoms object in ASE): Output structure of cropped supercell
    """
    
    cell = structure.get_cell()
    Lx, Ly = cell.lengths()[:2]

    new_Lx = shrink_factor * Lx
    new_Ly = shrink_factor * Ly

    center = np.array([Lx/2, Ly/2])

    pos = structure.get_positions()

    mask = (
        (abs(pos[:,0] - center[0]) <= new_Lx/2) &
        (abs(pos[:,1] - center[1]) <= new_Ly/2)
    )

    cropped = structure[mask].copy()

    cropped.set_cell([
        [new_Lx, 0, 0],
        [0, new_Ly, 0],
        [0, 0, cell[2,2]]
    ])

    cropped.center(axis=(0,1))

    return cropped

def generate_twisted_bilayers(f, twist_angle=0.0, scale=10):
    """
    Generates an Atoms object for twisted bilayers of 2D materials.

    Parameters:
        - f (str): Input structure file (i.e. CIF, xyz, POSCAR, etc.)
            * NOTE: This will only work for structure files that only contain two layers within the unit cell.
        - twist_angle (float): Twist angle (in degrees) of the bilayers about the z-axis
        - scale (int): Number of repetitions of the default supercell

    Returns:
        - full_structure (Atoms object in ASE): Output structure file of twisted bilayers
    """

    # Read the cif file + extend the cell along the x- and y-directions
    structure = ase.io.read(f, index=-1) * (scale, scale, 1)

    # cif files from Materials Project typically contain two layers, only one is necessary
    # 1. Create a mask to extract only one of the layers
    # 2. Reduce the vacuum space left behind by masking operation
    mask = structure.get_scaled_positions()[:,2] < 0.5
    monolayer = structure[mask]
    monolayer.center(vacuum=-2, axis=2)
        
    # Make sure the supercell is NOT periodic along the z-direction
    monolayer.set_pbc([1, 1, 0])
    
    # Orthogonalize the supercell
    orthogonal_monolayer = abtem.orthogonalize_cell(monolayer)
    
    # Determine offset distance between the two layers
    z_separation = structure.get_positions()[1,2] - structure.get_positions()[0,2] # distance (in Angstroms) between layers
    z_original = orthogonal_monolayer.get_positions()[0,2] # original z-position of the monolayer
    vacuum_layer_height = orthogonal_monolayer.get_cell().lengths()[2] / 2 # remaining vacuum space above the monolayer in the supercell
    z_offset = z_original + z_separation - vacuum_layer_height

    # Create two copies of monolayers
    full_structure = orthogonal_monolayer.copy()
    second_monolayer = orthogonal_monolayer.copy()

    # Offset and twist each monolayer
    second_monolayer.positions[:,2] += z_offset
    second_monolayer.euler_rotate(phi=twist_angle/2, center='COU') # twist along the center of the supercell
    full_structure.euler_rotate(phi=-twist_angle/2, center='COU')

    # Append the two monolayers
    full_structure.extend(second_monolayer)
    full_structure.center(vacuum=2.0, axis=2)
    full_structure = crop_center_square(full_structure)
    full_structure.set_pbc([1, 1, 0])
    full_structure.wrap()
    
    return full_structure