Dataset
==================
The Datasets used for training our models will be provided by the Open Graph Benchmark (OGB).
For each Dataset, the following files will be provided:
#.Graph Lables.
#.Number of Nodes per Graph.
#.Number of Edges per Graph.
#.Edges between Nodes.
#.Node Features
#.Edge Features

Node Features 
-------------------
The Node Features will be displayed through a 9-dimensional Index List, which corresponds to:
#. Atom Number (1 to 119) (H is excluded, so C is 5 not 6)
#. Chirality (Unspecified, Tetrahedral Clockwise(CW), Tetrahedral Counter Clockwise(CCW), Other and Misc)
#. Degree (0 to 10 and misc)
#. Formal Charge (-5 to 5 and misc) (so an index of 5 would be a charge of 0)
#. Number of H Atoms (0 to 8 and misc)
#. Number of radicals (0 to 4 and misc)
#. Hybridization (SP, SP2, SP3, SP3D, SP3D2, misc)
#. Is Aromatic (False, True)
#. Is in Ring (False, True)

Edge Features
-------------------
The Edge Features will be displayed through a 3-dimensional Index List, which corresponds to:
#. Bond Type (Single, Double, Triple, Aromatic, Misc)
#. Bond Stereo (None, Z, E, Cis, Trans, Any)
#. Is Conjugated (False, True)
