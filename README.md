# &#9658;TensorMol 0.1
![](doc/images/newtitle.png)
-Title signature by Alex Graves' handwriting LSTM https://arxiv.org/abs/1308.0850

## Authors:
 Kun Yao (kyao@nd.edu), John Herr (jherr1@nd.edu),
 David Toth (dtoth1@nd.edu), Ryker McIntyre, Nicolas Casetti
 John Parkhill (john.parkhill@gmail.com)

## Model Chemistries:
 - Behler-Parrinello with electrostatics
 - Many Body Expansion
 - Bonds in Molecules NN
 - Atomwise Forces
 - Inductive Charges

## Simulation Types:
 - Optimizations
 - Molecular Dynamics (NVE,NVT Nose-Hoover)
 - Monte Carlo
 - Open/Periodic Boundary Conditions
 - Meta-Dynamics
 - Infrared spectra by propagation
 - Infrared spectra by Harmonic Approximation.
 - Nudged Elastic Band
 - Path integral simulations via interface with [I-PI](https://github.com/i-pi/i-pi) MD engine.

## License: GPLv3
By using this software you agree to the terms in COPYING

## Installation:
 - Works on OSX, Ubuntu, and Windows subsystem for Linux.
```
git clone https://github.com/jparkhill/TensorMol.git
cd TensorMol
# If you are using python2x
sudo pip install -e .
# If you are using python3x
sudo pip3 install -e .
python test.py
```

## Usage:
 - ```import TensorMol as tm```
 - TensorMol assumes a directory structure executing path which mirrors the git.
 - We are working on /doc/Tutorials, but it's sparse now.
 - There is also a lot in various test.py's
 - Please also refer to IPython notebooks in /notebooks.
 - IPI interface: start server: ~/i-pi/i-pi samples/i-pi_interface/H2O_cluster.xml > log &; run client: python test_ipi.py

## Timing Information
TensorMol is robust and fast. You can get an BP+electrostatic energy and force of this monstrous cube of 24,000 atoms
in less than 100 seconds on a 2015 MacbookPro (Core i7 2.5Ghz, 16GB mem). Periodic simulations are about 3x
more expensive.  

<img src="doc/images/monster.png" width="300">
<img src="doc/images/Timings.png" width="800">
<img src="doc/images/PeriodicTimings.png" width="800">

## Sample Results

### Chemical Reactions
Converged nudged elastic band simulations of the cyclization cascade of endiandric acid C (c.f. K. C. Nicolaou, N. A. Petasis, R. E. Zipkin, 1982, The endiandric acid cascade. Electrocyclizations in organic synthesis. 4. Biomimetic approach to endiandric acids A-G. Total synthesis and thermal studies, J. Am. Chem. Soc. 104(20):5560–5562).

<img src="doc/images/cascade.gif" width="300">

This reaction path can be found in a few minutes on an ordinary laptop. Relaxation from the linearly interpolated guess looks like this:

<img src="doc/images/neb.png" width="300">

The associated energy surface is shown below.

<img src="doc/images/pes.png" width="300">

### Dynamic Properties
- Water Trimer IR spectrum generated with david_testIR() in test.py
- The red lines are MP2(qchem) solid line is TensorMol's IR propagation.

![](doc/images/water.png)

## Requirements:
- Minimum Pre-Requisites: Python2.7x, TensorFlow
- Python3x support coming soon.
- Useful Pre-Requisites: CUDA7.5, PySCF
- To Train Minimally: ~100GB Disk 20GB memory
- To Train Realistically: 1TB Disk, GTX1070++
- To Evaluate: Normal CPU and 10GB Mem

## Acknowledgements:
 - Google Inc. (for TensorFlow)
 - NVidia Corp. (hardware)
 - von Lilienfeld Group (for GBD9)
 - Chan Group (for PySCF)

## Common Issues:
- nan during training due to bad checkpoints in /networks (clean.sh)
- Also crashes when reviving networks from disk.
- if you have these issues try re-installing or:

```
sh clean.sh
```
