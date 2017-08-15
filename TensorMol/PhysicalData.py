# The spatial unit of TensorMol is Angstrom.
# Tne energy unit of Tensormol is Hartree except in MD where it is kcal/mol
# The time unit of Tensormol is the Fs. 
#
# These should be all Caps. etc...
#
from __future__ import absolute_import
import numpy as np
from math import pi as Pi

ELEHEATFORM = {1:-0.497912, 6:-37.844411, 7:-54.581501, 8:-75.062219, 9:-99.716370}     # ref: https://figshare.com/articles/Atomref%3A_Reference_thermochemical_energies_of_H%2C_C%2C_N%2C_O%2C_F_atoms./1057643
bond_length_thresh = {"HH": 1.5, "HC": 1.5, "HN": 1.5, "HO": 1.5, "CC": 1.7, "CN": 1.7, "CO": 1.7, "NN": 1.7, "NO": 1.7, "OO": 1.7 }
#ele_U = {1:-0.500273, 6:-37.846772, 7:-54.583861, 8:-75.064579, 9:-99.718730}   # ref: https://figshare.com/articles/Atomref%3A_Reference_thermochemical_energies_of_H%2C_C%2C_N%2C_O%2C_F_atoms./1057643
ele_U = {1:-0.500273, 6:-37.8462793, 7:-54.58449,  8:-75.060612}
ele_E_david = {1: -0.5026682859, 6:-37.8387398670, 8:-75.0586028553}
atoi = {'H':1,'He':2,'Li':3,'Be':4,'B':5,'C':6,'N':7,'O':8,'F':9,'Ne':10,'Na':11,'Mg':12,'Al':13,'Si':14,'P':15,'S':16,'Cl':17,'Ar':18,'K':19,'Ca':20,'Sc':21,'Ti':22,'Si':23,'V':24,'Cr':25,'Br':35, 'Cs':55, 'Pb':82}
itoa = ['X','H','He','Li','Be','B','C','N','O','F','Ne','Na','Mg','Al','Si','P','S','Cl','Ar','K','Ca','Sc','Ti','Si','V','Cr','Br','Cs','Pb']
atoc = {1: 40, 6: 100, 7: 150, 8: 200, 9:240}
atom_valance = {1:1, 8:2, 7:3, 6:4}
bond_index = {"HH": 1, "HC": 2, "HN": 3, "HO": 4, "CC": 5, "CN": 6, "CO": 7, "NN": 8, "NO": 9, "OO": 10}
dihed_pair = {1006:1, 1007:2, 1008:3, 6006:4, 6007:5, 6008:6,  7006:7, 7007:8, 7008:9, 8006:10, 8007:11, 8008:12}  # atomic_1*1000 + atomic_2 hacky way to do that
atomic_radius = {1:53.0, 2:31.0, 3:167.0, 4:112.0, 5:87.0, 6:67.0, 7:56.0, 8:48.0, 9:42.0, 10:38.0, 11:190.0, 12:145.0, 13:118.0, 14:111.0, 15:98.0, 16:88.0, 17:79.0, 18:71.0} # units in pm, ref: https://en.wikipedia.org/wiki/Atomic_radius
atomic_radius_2 = {1:25.0, 3:145.0, 4:105.0, 5:85.0, 6:70.0, 7:65.0, 8:60.0, 9:50.0, 11:180.0, 12:150.0, 13:125.0, 14:110.0, 15:100.0, 16:100.0, 17:100.0} # units in pm, ref: https://en.wikipedia.org/wiki/Atomic_radius
atomic_vdw_radius = {1:1.001, 2:1.012, 3:0.825, 4:1.408, 5:1.485, 6:1.452, 7:1.397, 8:1.342, 9:1.287, 10:1.243} # ref: http://onlinelibrary.wiley.com/doi/10.1002/jcc.20495/epdf   unit in angstrom
C6_coff = {1:0.14, 2:0.08, 3:1.16, 4:1.61, 5:3.13, 6:1.75, 7:1.23, 8:0.70, 9:0.75, 10:0.63}  # ref: http://onlinelibrary.wiley.com/doi/10.1002/jcc.20495/epdf unit in Jnm^6/mol
S6 = {"PBE": 0.75, "BLYP":1.2, "B-P86":1.05, "TPSS":1.0, "B3LYP":1.05}  # s6 scaler of different DF of Grimmer C6 scheme
atomic_raidus_cho = {1:0.328, 6:0.754, 8:0.630} # roughly statisfy mp2 cc-pvtz equilibrium carbohydrate bonds.
GOLDENRATIO = (np.sqrt(5.)+1.0)/2.0
KAYBEETEE = 0.000950048 # At 300K
BOHRPERA = 1.889725989
ANGSTROMPERMETER = pow(10.0,10.0)
BOHRPERM = BOHRPERA*ANGSTROMPERMETER
BOHRINM = 0.52917720859*pow(10.0,-10.0)
KJPERHARTREE = 2625.499638
JOULEPERHARTREE = KJPERHARTREE*1000.0
JOULEPERKCAL = 4183.9953
KCALPERHARTREE = 627.509474
WAVENUMBERPERHARTREE = 219474.63
ELECTRONPERPROTONMASS = 1836.15267
FEMTOPERUNIT = pow(10.0,-15.0)
PICOPERUNIT = pow(10.0,-12.0)
SPEEDOFLIGHT=299792458.0 #m/s
MASSOFELECTRON = 548.579909*pow(10.0,-9.0) # In kilograms/mol
FSPERAU = 0.0241888
AVOCONST = 6.02214086*np.power(10.0,23.0)
AUPERDEBYE = 0.393456
IDEALGASR =  8.3144621 # J/molK
AMUINKG = 1.660538782*pow(10.0,-27.0)
SECPERATOMIC = 2.418884326505*pow(10.0,-17.0) # Atomic unit of time.
#Convert evals from H/(kg bohr^2) to J/(kg m^2) = 1/s^2 */
KCONVERT = (4.359744*pow(10.0,-18.0))/(BOHRINM * BOHRINM * AMUINKG);
CMCONVERT = 1.0/(2.0 * Pi * SPEEDOFLIGHT * 100.0);
ATOMICMASSESAMU = np.array([1.00794, 4.002602, 6.941, 9.012182, 10.811, 12.0107, 14.0067, 15.9994, 18.9984032, 20.1791, 22.98976928, 24.3050, 26.9815386, 28.0855, 30.973762, 32.065, 35.453, 39.948, 39.0983, 40.078, 44.955912, 47.867, 50.9415,51.9961,54.938045,55.845,58.933195,58.6934,63.546,65.38,69.723,72.63,74.92160,78.96,79.904,83.798,85.4678,87.62,88.90585,91.224,92.90638,95.96,98.,101.07,102.90550,106.42,107.8682,112.411,114.818,118.710,121.760,127.60,126.90447,131.293,132.9054519,137.327,138.90547,140.116,140.90765,144.242,145.,150.36,151.964,157.25,158.92535,162.500,164.93032,167.259,168.93421,173.054,174.9668,178.49,180.94788,183.84,186.207,190.23,192.217,195.084,196.966569,200.59,204.3833,207.2,208.98040,209.,210.,222.,223.,226.,227.,232.03806,231.03586,238.02891,237.,244.,243.,247.,247.,251.,252.,257.])
# These are in Kilograms/mol
ATOMICMASSES = 0.000999977*ATOMICMASSESAMU
