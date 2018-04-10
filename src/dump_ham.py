"""IMPORTS"""
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

setup_paths.insert(
    0, '/Users/andersjensen/steno/local_gpaw/data/benzene/gpaw_basis')
setup_paths.insert(0, '/kemi/aj/local_gpaw/data/benzene/gpaw_basis')
setup_paths.insert(0, '/Users/andersjensen/')

"""ARGPARSE"""
import argparse
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument(
    '--path',
    default='/Users/andersjensen/steno/local_gpaw/data/allene5/',
    help='path to data folder')
parser.add_argument(
    '--xyzname',
    default='/Users/andersjensen/steno/local_gpaw/data/allene5/',
    help='name of the xyz file')
parser.add_argument('--basis',
                    default='sz',
                    help='basis (sz, dzp, ...)')
args = parser.parse_args()

basis = args.basis
path = args.path
trajname = args.xyzname

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
molecule = read(path + trajname)
view(molecule)

"""
Identify end atoms and align according to z-direction

atoms the furthers from one another
"""

import seaborn as sns
import matplotlib.pyplot as plt


pos = molecule.get_positions()


def distance(pos1, pos2):
    dis = np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] -
                                            pos2[1])**2 + (pos1[2] - pos2[2])**2)
    return dis


def distance_matrix(pos):
    dM = ([[distance(pos[i], pos[j]) for i in range(pos.shape[0])]
           for j in range(pos.shape[0])])
    sns.heatmap(dM)
    plt.show()
    return dM


dM = distance_matrix(pos)
m = np.unravel_index(pos.argmax(), pos.shape)

print(m)

# print(dM[endatom1,endatom2])
# print(m)
# print()

# dis_test = distance(pos[0],pos[0])
# print(dis_test
# exit()

# #electorde
sI = endatom1
eI = endatom2

print(sI,eI)

po = (molecule[endatom1].position)
lo = (molecule[endatom2].position)
v = lo - po
z = [0, 0, 1]
molecule.rotate(v, z)
# molecule.rotate('z', np.pi + np.pi / 4.0)

molecule.center(vacuum=vacuum)

elec1 = Atoms('H', positions=[molecule[sI].position])
elec2 = Atoms('H', positions=[molecule[eI].position])
if sI > eI:
    del molecule[sI]
    del molecule[eI]
else:
    del molecule[eI]
    del molecule[sI]

atoms = elec1 + molecule + elec2

atoms.center(vacuum=vacuum)
atoms.set_pbc([1, 1, 1])

view(atoms)
exit()
"""
Save data
"""
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

H_kin = wfs.T_qMM[0]
print(H_kin)
np.save(path + "H_kin.npy", H_kin)
# exit()

# path = "/kemi/aj/local_gpaw/data/H20/"
basename = "basis_{0}__xc_{1}__fdwithd_{2}__kpts_{3}__mode_{4}__vacuum_{5}__".format(
    basis, xc, FDwidth, kpts, mode, vacuum)

# basename = "basis_{0}__xc_{1}__a_{2}__c_{3}__d_{4}__h_{5}__fdwithd_{6}__kpts_{7}__mode_{8}__vacuum_{9}__".format(basis,xc,a,c,d,h,FDwidth,kpts,mode,vacuum)

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
print(len(bfs))
plot_basis(atoms, phi_xg, ns=len(bfs), folder_name=path + "basis/ao")

summ = np.zeros(phi_xg[0, :, :, :].shape)
ns = np.arange(len(bfs))
for n, phi in zip(ns, phi_xg.take(ns, axis=0)):
    summ += abs(phi) * abs(phi)
write(path + "basis/ao/sum.cube", atoms, data=summ)

"""Lowdin"""
dump_hamiltonian_parallel(path + 'scat_' + basename, atoms, direction='z')
atoms.write(path + basename + ".traj")

print("lowdin")
H_ao, S_ao = pickle.load(open(path + 'scat_' + basename + '0.pckl', 'rb'))
# print "done loading"
H_ao = H_ao[0, 0]
S_ao = S_ao[0]
n = len(S_ao)
eig, rot = np.linalg.eig(S_ao)
rot = np.dot(rot / np.sqrt(eig), rot.T.conj())
r_mat = np.identity(n)
r_mat[:] = np.dot(r_mat, rot)
A = r_mat
U = np.linalg.inv(A)
rot_mat = A
c_fo_xi = asc(rot_mat.real.T)  # coefficients
lowdin_phi_xg = calc.wfs.basis_functions.gd.zeros(len(c_fo_xi))
wfs = calc.wfs
gd0 = calc.wfs.gd
calc.wfs.basis_functions.lcao_to_grid(c_fo_xi, lowdin_phi_xg, -1)
np.save(path + basename + "lowdin_basis", [lowdin_phi_xg])
np.save(path + basename + "lowdin_U", U)
plot_basis(atoms, lowdin_phi_xg, ns=len(bfs),
           folder_name=path + "basis/lowdin")

# MO - basis
eig, vec = np.linalg.eig(np.dot(np.linalg.inv(S_ao), H_ao))
order = np.argsort(eig)
eig = eig.take(order)
vec = vec.take(order, axis=1)
S_mo = np.dot(np.dot(vec.T.conj(), S_ao), vec)
vec = vec / np.sqrt(np.diag(S_mo))
S_mo = np.dot(np.dot(vec.T.conj(), S_ao), vec)
H_mo = np.dot(np.dot(vec.T, H_ao), vec)

rot_mat = vec
c_fo_xi = asc(rot_mat.real.T)  # coefficients
mo_phi_xg = calc.wfs.basis_functions.gd.zeros(len(c_fo_xi))
wfs = calc.wfs
gd0 = calc.wfs.gd
calc.wfs.basis_functions.lcao_to_grid(c_fo_xi, mo_phi_xg, -1)
np.save(path + basename + "mo_energies", eig)
np.save(path + basename + "mo_basis", mo_phi_xg)
plot_basis(atoms, mo_phi_xg, ns=len(bfs), folder_name=path + "basis/mo")


# eigenchannels
from new import ret_gf_ongrid, calc_trans, fermi_ongrid,\
    lesser_gf_ongrid, lesser_se_ongrid, ret_gf_ongrid2,\
    lesser_gf_ongrid2, retarded_gf2
gamma = 1e0
H_cen = H_ao
S_cen = S_ao
n = len(H_cen)
GamL = np.zeros([n, n])
GamR = np.zeros([n, n])
GamL[0, 0] = gamma
GamR[n - 1, n - 1] = gamma

# #C8
# bfs = get_bfi2(symbols, basis, [23,24])
# print bfs, "left"
# GamL[bfs[0],bfs[0]] = GamL[bfs[1],bfs[1]] = gamma
# symbols = atoms.get_chemical_symbols()
# bfs = get_bfi2(symbols, basis, [21,22])
# print bfs, "right"
# GamR[bfs[0],bfs[0]] = GamR[bfs[1],bfs[1]] = gamma


print(eig, H_mo[24, 24])

for ef in [eig[22], 0, eig[27]]:
    """Current approx at low temp"""
    Sigma_r = -1j / 2 * (GamL + GamR)
    # Gr_approx = retarded_gf2(H_cen, S_cen, ef, Sigma_r)
    Gr_approx = retarded_gf2(H_cen, S_cen, ef, Sigma_r)
    # print Gr_approx, "Gr_approx"

    # from new import calc_trans, ret_gf_ongrid
    # e_grid = np.arange(eig[15],eig[30],0.001)
    # Gamma_L = [GamL for en in range(len(e_grid))]
    # Gamma_R = [GamR for en in range(len(e_grid))]
    # Gamma_L = np.swapaxes(Gamma_L, 0, 2)
    # Gamma_R = np.swapaxes(Gamma_R, 0, 2)
    # Gr = ret_gf_ongrid(e_grid, H_cen, S_cen, Gamma_L, Gamma_R)
    # trans = calc_trans(e_grid,Gr,Gamma_L,Gamma_R)

    # np.save("/Users/andersjensen/Desktop/trans.npy",np.array([e_grid,trans]))

    # Tt = GamL.dot(Gr_approx).dot(GamR).dot(Gr_approx.T.conj())
    # print Tt.trace(), "transmission"

    def get_left_channels(Gr, S, GamL, GamR, nchan=1):
        g_s_ii = Gr
        lambda_l_ii = GamL
        lambda_r_ii = GamR

        s_mm = S
        s_s_i, s_s_ii = np.linalg.eig(s_mm)
        s_s_i = np.abs(s_s_i)
        s_s_sqrt_i = np.sqrt(s_s_i)  # sqrt of eigenvalues
        s_s_sqrt_ii = np.dot(s_s_ii * s_s_sqrt_i, s_s_ii.T.conj())
        s_s_isqrt_ii = np.dot(s_s_ii / s_s_sqrt_i, s_s_ii.T.conj())

        lambdab_r_ii = np.dot(np.dot(s_s_isqrt_ii, lambda_r_ii), s_s_isqrt_ii)
        a_l_ii = np.dot(np.dot(g_s_ii, lambda_l_ii), g_s_ii.T.conj())  # AL
        ab_l_ii = np.dot(np.dot(s_s_sqrt_ii, a_l_ii),
                         s_s_sqrt_ii)  # AL in lowdin
        lambda_i, u_ii = np.linalg.eig(ab_l_ii)  # lambda and U
        ut_ii = np.sqrt(lambda_i / (2.0 * np.pi)) * u_ii  # rescaled
        m_ii = 2 * np.pi * np.dot(np.dot(ut_ii.T.conj(), lambdab_r_ii), ut_ii)
        T_i, c_in = np.linalg.eig(m_ii)
        T_i = np.abs(T_i)

        channels = np.argsort(-T_i)
        c_in = np.take(c_in, channels, axis=1)
        T_n = np.take(T_i, channels)
        v_in = np.dot(np.dot(s_s_isqrt_ii, ut_ii), c_in)
        return T_n, v_in

    T_n, v_in = get_left_channels(Gr_approx, S_cen, GamL, GamR)
    # print(T_n)

    def get_eigenchannels(Gr, GamR):
        """
        Calculate the eigenchannels from
        G Gamma G
        """
        A = Gr.dot(GamL).dot(Gr.T.conj())
        Teigs, Veigs = np.linalg.eig(A)
        order = np.argsort(Teigs)
        Teigs = Teigs.take(order)
        Veigs = Veigs.take(order, axis=1)

        return Teigs, Veigs

    # T_n, v_in = get_eigenchannels(Gr_approx,GamL)

    # print T_n
    # eig, vec = np.linalg.eig(Tt)
    # order = np.argsort(eig)
    # eig = eig.take(order)
    # vec = vec.take(order, axis=1)

    # print v_in
    # print np.abs(v_in)

    # print eig, "eigs"

    # rot_mat = np.abs(v_in)
    rot_mat = v_in.real
    # rot_mat = vec
    # print rot_mat
    c_fo_xi = asc(rot_mat.T)  # coefficients
    # print c_fo_xi
    # print c_fo_xi.max(), c_fo_xi.min()
    # exit()
    # c_fo_xi = asc(rot_mat)#coefficients
    teig_phi_xg = calc.wfs.basis_functions.gd.zeros(len(c_fo_xi))
    wfs = calc.wfs
    gd0 = calc.wfs.gd
    calc.wfs.basis_functions.lcao_to_grid(c_fo_xi, teig_phi_xg, -1)
    np.save(path + basename +
            "eigenchannels__ef_{0}.npy".format(ef), teig_phi_xg)
    np.save(path + basename + "Teig__ef_{0}.npy".format(ef), eig)
    plot_eig(atoms, teig_phi_xg, ns=2, folder_name=path +
             "basis/eigchan", ext="real_ef_{0}".format(ef))

    # rot_mat = np.abs(v_in)
    rot_mat = v_in.imag
    # rot_mat = vec
    # print rot_mat
    c_fo_xi = asc(rot_mat.T)  # coefficients
    # print c_fo_xi
    # print c_fo_xi.max(), c_fo_xi.min()
    # exit()
    # c_fo_xi = asc(rot_mat)#coefficients
    teig_phi_xg = calc.wfs.basis_functions.gd.zeros(len(c_fo_xi))
    wfs = calc.wfs
    gd0 = calc.wfs.gd
    calc.wfs.basis_functions.lcao_to_grid(c_fo_xi, teig_phi_xg, -1)
    np.save(path + basename +
            "eigenchannels__ef_{0}.npy".format(ef), teig_phi_xg)
    np.save(path + basename + "Teig__ef_{0}.npy".format(ef), eig)
    plot_eig(atoms, teig_phi_xg, ns=2, folder_name=path +
             "basis/eigchan", ext="imag_ef_{0}".format(ef))
