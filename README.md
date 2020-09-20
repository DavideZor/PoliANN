# PoliANN
PoliANN is a family of implemented Neural Networks used to solve Partial Differential Equations. This project belongs to the author's Master's thesis "A modern non-deterministic approach for solving Partial Differential Equations: Machine Learning applied to the Navier-Stokes equations" in Aeronautical Engineering at Politecnico di Milano.


The codes used in the Master's thesis work are here reported for the sake of completeness. In particular, they are divided according to the different families of the considered differential equations:

- **Burgers/** In this folder, the codes used to solve the viscous Burgers equation using the Finite Difference (**FDM/**), Finite Volume (**FVM/**), Finite Element (**FEM/**) and Feed Forward Neural Network Method (**FFNN/**) are listed.
- **Poisson/** In this folder, the codes used to solve the Poisson equation on the unit square (**UnitSquare/**), star (**Star/**) and Italian territory (**Italy/**) domain are listed. The Finite Element Method and the Feed Forward Neural Network approach are both used to solve such equation.
- **NavierStokes/** In this folder, the codes used to solve the Navier-Stokes equations for the Kovasznay (**Kovasznay/**), step flow (**StepFlow/**) and 2D cylinder (**2DCylinder/**) probelm are listed. The Finite Element Method and the Feed Forward Neural Network approach are both used to solve such equations.
- **MATLAB/** In this folder, the codes used to obtain the Mercator projection of the Italian territory (**Italy_XY.txt**) are presented. These are the only codes developed in the MATLAB environment exploiting the Geography Toolbox package. All the previous codes are written in Python.

For the Finite Element Method the Python library FEniCS was exploited, whereas for the Feed Forward Neural Network approach the DeepXDE library (based on TensorFlow 2) was used.
