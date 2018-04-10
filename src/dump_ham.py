"""
Dumps hamiltonian and basis on grid used in current density calculation
"""

from ase import Atoms
from ase.io import read, write
from ase.visualize import view
from ase.dft.kpoints import monkhorst_pack
from ase.io.trajectory import Trajectory

from gpaw import GPAW, FermiDirac
from gpaw import setup_paths
from gpaw.lcao.tools import get_lcao_hamiltonian, get_lead_lcao_hamiltonian
from gpaw.lcao.tools import dump_hamiltonian_parallel, get_bfi2

import pickle as pickle

import numpy as np
from numpy import ascontiguousarray as asc

from new2 import plot_basis, plot_eig
from utils import *

"""ARGPARSE"""
import argparse
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument(
    '--path',
    default='../data/c8/',
    help='path to data folder')
parser.add_argument(
    '--xyzname',
    default='c8.xyz',
    help='name of the xyz file')
parser.add_argument('--basis',
                    default='sz',
                    help='basis (sz, dzp, ...)')
args = parser.parse_args()

import os
basis = args.basis
path = os.path.abspath(args.path) + "/"
xyzname = args.xyzname

"""
Constants
"""
xc = 'PBE'
FDwidth = 0.1
kpts = (1, 1, 1)
mode = 'lcao'
h = 0.20
vacuum = 4
basis_full = {'H': 'sz', 'C': basis, 'Si': basis, 'Ge': basis}

"""
Read molecule
"""
molecule = read(path + xyzname)
view(molecule)

"""
Identify end atoms and align according to z-direction

atoms the furthers from one another
"""
atoms = identify_and_align(molecule)

np.save(path + "positions.npy", atoms.get_positions())
np.save(path + "symbols.npy", atoms.get_chemical_symbols())
atoms.write(path + "central_region.xyz")


"""
Run and converge calculation
"""
calc = GPAW(h=h,
            xc=xc,
            basis=basis_full,
            occupations=FermiDirac(width=FDwidth),
            kpts=kpts,
            mode=mode,
            symmetry={'point_group': False, 'time_reversal': False})
atoms.set_calculator(calc)
atoms.get_potential_energy()  # Converge everything!
Ef = atoms.calc.get_fermi_level()

wfs = calc.wfs
kpt = monkhorst_pack((1, 1, 1))

basename = "basis_{0}__xc_{1}__h_{2}__fdwithd_{3}__kpts_{4}__mode_{5}__vacuum_{6}__".format(
    basis, xc, h, FDwidth, kpts, mode, vacuum)

dump_hamiltonian_parallel(path + 'scat_' + basename, atoms, direction='z')

a_list = range(0, len(atoms))
symbols = atoms.get_chemical_symbols()
bfs = get_bfi2(symbols, basis_full, range(len(a_list)))
rot_mat = np.diag(v=np.ones(len(bfs)))
c_fo_xi = asc(rot_mat.real.T)  # coefficients
phi_xg = calc.wfs.basis_functions.gd.zeros(len(c_fo_xi))
wfs = calc.wfs
gd0 = calc.wfs.gd
calc.wfs.basis_functions.lcao_to_grid(c_fo_xi, phi_xg, -1)
np.save(path + basename + "ao_basis_grid", [phi_xg, gd0])
plot_basis(atoms, phi_xg, ns=len(bfs), folder_name=path + "basis/ao")

