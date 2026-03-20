# MATSCI465_FinalProject_MoireLattices
Final Project of Arami Chang, Katie Eisenman, and Zek Kelly for MSE 465 in the department of Materials Science and Engineering at Northwestern University.

This project utilizes abTEM and ASE python libraries to simulate Moire superlattice structures with varying twist angles and create diffraction patterns and 4D STEM images of hexagonal boron nitride (hBN), a common 2D material. 

The twisted_bilayers.py file is used to generate two twisted bilayers of material based on CIF files contained in the example CIFs folder. 

Use of this file is demonstrated in the generate_twisted_bilayers.ipynb notebook and the diffraction_pattern.ipynb notebook. 

The diffraction_pattern.ipynb notebook demonstrates a pipeline to take a twisted bilayer and use the multislice method to generate diffraction patterns of hBN. This file contains code to create movies that show how the structure changes as a function of twist angle, which can be viewed in the animation files. 

The 4dstem_simulation.ipynb file visualizes CBED patterns and HAADF images from a generated 4D STEM dataset of hBN. This code also performs center-of-mass calculations and generates center-of-mass, electric field, and charge density maps.