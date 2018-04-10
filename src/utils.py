import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from ase import Atoms
from ase.io import read, write
from ase.visualize import view

from gpaw import GPAW, FermiDirac
from gpaw import setup_paths
from gpaw.lcao.tools import get_lcao_hamiltonian, get_lead_lcao_hamiltonian
from gpaw.lcao.tools import dump_hamiltonian_parallel, get_bfi2

from numpy import ascontiguousarray as asc

import pickle
from tqdm import tqdm

def plot_basis(atoms, phi_xG, ns, folder_name='./basis'):
    """
    r: coefficients of atomcentered basis functions
    atoms: Atoms-object 
    ns: indices of bfs functions to plot. 
    """
    # for n, phi in zip(ns, phi_xG.take(ns, axis=0)):
    n=0
    for phi in phi_xG:
        # print "writing %d of %d" %(n, len(ns)), 
        write('%s/%d.cube' % (folder_name,n), atoms, data=phi)
        n += 1

def distance(pos1, pos2):
    dis = np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] -
                                            pos2[1])**2 + (pos1[2] - pos2[2])**2)
    return dis


def distance_matrix(pos):
    dM = np.array([[distance(pos[i], pos[j]) for i in range(pos.shape[0])]
           for j in range(pos.shape[0])])
    #sns.heatmap(dM)
    #plt.show()
    return dM


def identify_and_align(molecule):
    vacuum = 4
    
    pos = molecule.get_positions()

    dM = distance_matrix(pos)
    m = np.unravel_index(np.argmax(dM, axis=None), dM.shape)

    endatom1, endatom2 = m

    # #electorde
    sI = endatom1
    eI = endatom2

    #print(sI,eI)

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

    #view(atoms)
    return atoms

def all_this(atoms,calc,path,basis,basis_full,xc,FDwidth,kpts,mode,h,vacuum=4):
    wfs = calc.wfs
    from ase.dft.kpoints import monkhorst_pack
    from ase.io import write as ase
    kpt = monkhorst_pack((1, 1, 1))

    H_kin = wfs.T_qMM[0]
    np.save(path + "H_kin.npy", H_kin)
    # exit()

    # path = "/kemi/aj/local_gpaw/data/H20/"
    # basename = "basis_{0}__xc_{1}__fdwithd_{3}__kpts_{4}__mode_{5}__vacuum_{6}__".format(basis, xc, FDwidth, kpts, mode, vacuum)

    basename = "basis_{0}__xc_{1}__h_{2}__fdwithd_{3}__kpts_{4}__mode_{5}__vacuum_{6}__".format(basis, xc, h, FDwidth, kpts, mode, vacuum)

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
    plot_basis(atoms, phi_xg, ns=len(bfs), folder_name=path + "basis/ao")

    summ = np.zeros(phi_xg[0, :, :, :].shape)
    ns = np.arange(len(bfs))
    for n, phi in zip(ns, phi_xg.take(ns, axis=0)):
        summ += abs(phi) * abs(phi)
    write(path + "basis/ao/sum.cube", atoms, data=summ)

    """Lowdin"""
    dump_hamiltonian_parallel(path + 'scat_' + basename, atoms, direction='z')
    atoms.write(path + basename + ".traj")

    H_ao, S_ao = pickle.load(open(path + 'scat_' + basename + '0.pckl', 'rb'))
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
    from new2 import ret_gf_ongrid, calc_trans, fermi_ongrid,\
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


    # for ef in [eig[22], 0, eig[27]]:
    for ef in [0]:
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
    #         print rot_mat
            c_fo_xi = asc(rot_mat.T)  # coefficients
    #         print c_fo_xi
    #         print c_fo_xi.max(), c_fo_xi.min()
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
    #         print rot_mat
            c_fo_xi = asc(rot_mat.T)  # coefficients
    #         print c_fo_xi
    #         print c_fo_xi.max(), c_fo_xi.min()
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

def plot_transmission(energy_grid, trans, save_name):
    """
    plots the transmission
    """
    plt.plot(energy_grid, trans)
    plt.yscale('log')
    plt.ylabel(r'Transmission')
    plt.xlabel(r'E-E$_F$ (eV)')
    plt.savefig(save_name)
    plt.close()

def calc_trans(energy_grid, gret, gamma_left, gamma_right):
    """
    Landauer Transmission
    """
    trans = np.array([np.dot(np.dot(np.dot(\
                    gamma_left[:, :, en], gret[:, :, en]),\
                    gamma_right[:, :, en]), gret[:, :, en].T.conj())\
                    for en in range(len(energy_grid))])
    trans_trace = np.array([trans[en, :, :].trace() for en in range(len(energy_grid))])
    return trans_trace

def calc_trans_mo(energy_grid, gret, gamma_left, gamma_right):
    """
    Landauer Transmission
    """
    trans = np.array([np.dot(np.dot(np.dot(\
                    gamma_left[:, :, en], gret[:, :, en]),\
                    gamma_right[:, :, en]), gret[:, :, en].T.conj())\
                    for en in range(len(energy_grid))])
    trans_trace = np.array([trans[en, :, :].trace() for en in range(len(energy_grid))])
    return trans, trans_trace

def plot_complex_matrix(matrix, save_name):
    fig, axes = plt.subplots(nrows=1, ncols=2)
    for dat, axis in zip([matrix.real, matrix.imag], axes.flat):
        image = axis.matshow(dat, cmap='seismic', vmin=dat.min(), vmax=dat.max())
        cb = fig.colorbar(image, ax=axis, orientation='horizontal')
        cb.formatter.set_powerlimits((0, 0))
        cb.update_ticks()
        plt.cm.get_cmap('seismic')
    plt.savefig(save_name)
    plt.close()

def plot_real_matrix(matrix, save_name):
    fig, axes = plt.subplots(nrows=1, ncols=1)
    image = axes.matshow(matrix, cmap='seismic', vmin=matrix.min(), vmax=matrix.max())
    cb = fig.colorbar(image, ax=axes, orientation='horizontal')
    cb.formatter.set_powerlimits((0, 0))
    cb.update_ticks()
    plt.cm.get_cmap('seismic')
    plt.savefig(save_name)
    plt.close()

def retarded_gf(h_ao, s_ao, energy, gamma_left, gamma_right):
    """
    Retarded Gf using approx
    """
    return np.linalg.inv(energy*s_ao-h_ao+(1j/2.)*(gamma_left+gamma_right))

def retarded_gf2(h_ao, s_ao, energy, sigma_ret):
    """
    Retarded Gf
    """
    eta = 1e-10
    return np.linalg.inv(-sigma_ret+(energy+eta*1.j)*s_ao-h_ao)
    # return np.linalg.inv(energy*s_ao-h_ao-sigma_ret)

def ret_gf_ongrid(energy_grid, h_ao, s_ao, gamma_left, gamma_right):
    """
    Put the retarded gf on an energy grid
    """
    ret_gf = np.array([retarded_gf(h_ao, s_ao, energy_grid[en],\
                        gamma_left[:, :, en], gamma_right[:, :, en])\
                        for en in range(len(energy_grid))])
    ret_gf = np.swapaxes(ret_gf, 0, 2)
    return ret_gf

def ret_gf_ongrid2(energy_grid, h_ao, s_ao, sigma_ret):
    """
    Put the retarded gf on an energy grid
    """
    ret_gf = np.array([retarded_gf2(h_ao, s_ao, energy_grid[en],\
                        sigma_ret[:, :, en])\
                        for en in range(len(energy_grid))])
    ret_gf = np.swapaxes(ret_gf, 0, 2)
    return ret_gf


def fermi(energy, mu_):
    """
    Fermi-Dirac distribution
    """
    kbt_ = 25e-6
    return 1./(np.exp((energy-mu_)/kbt_)+1.)

def fermi_ongrid(energy_grid, e_f, bias):
    """
    FD on grid
    """
    f_left = [fermi(en, e_f+bias/2.)  for en in energy_grid]
    f_right = [fermi(en, e_f-bias/2.) for en in energy_grid]
    return f_left, f_right

def lesser_se_ongrid(energy_grid, gamma_left, gamma_right, f_left, f_right):
    """
    Lesser self energy on grid
    """
    sigma_les = np.array([1j*(gamma_left[:, :, en]*f_left[en] + gamma_right[:, :, en]*f_right[en])\
                          for en in range(len(energy_grid))])
    sigma_les = np.swapaxes(sigma_les, 0, 2)
    return sigma_les

def lesser_gf_ongrid(energy_grid, ret_gf, sigma_les):
    """
    Lesser gf on grid
    """
    lesser_gf = np.array([np.dot(np.dot(\
                    ret_gf[:, :, en], sigma_les[:, :, en]), ret_gf[:, :, en].T.conj())\
                    for en in range(len(energy_grid))])
    lesser_gf = np.swapaxes(lesser_gf, 0, 2)
    return lesser_gf


def lesser_gf_ongrid2(energy_grid, ret_gf, gammaL):
    """
    Lesser gf on grid Evers way
    """
    lesser_gf = np.array([np.dot(np.dot(\
                    ret_gf[:, :, en], 1j*gammaL[:, :, en]), ret_gf[:, :, en].T.conj())\
                    for en in range(len(energy_grid))])
    lesser_gf = np.swapaxes(lesser_gf, 0, 2)
    return lesser_gf

def plot_dos(energy_grid, s_ao, les_gf, ret_gf, save_name):
    """
    plotting dos/spectral function
    """
    dos1 = [(1/(2.*np.pi))*np.trace(np.dot(les_gf[:, :, en].imag, s_ao))\
             for en in range(len(energy_grid))]
    dos2 = [(-1/(np.pi))*np.trace(np.dot(ret_gf[:, :, en].imag, s_ao))\
             for en in range(len(energy_grid))]

    plt.plot(energy_grid, dos1, '--', label=r"G$^<$")
    plt.plot(energy_grid, dos2, label=r"G$^{ret}$")
    plt.legend()
    # plt.yscale('log')
    plt.ylabel(r'DOS')
    plt.xlabel(r'E-E$_F$ (eV)')
    plt.savefig(save_name)
    plt.close()


def plot_gradient(atoms, phi_xG, gd0, wfs, ns, folder_name='./grad', vacuum=3.0, h=0.20):
    """
    Plot of gradients
    """
    au2A = 0.529177249
    
    for n in ns:
        x_grid, y_grid, z_grid = gradient(phi_xG, gd0, wfs, n)
        x_cor = coords(gd0, c=0)
        y_cor = coords(gd0, c=1)
        z_cor = coords(gd0, c=2)
        
        s = 4
        x_cor = x_cor[::s]
        y_cor = y_cor[::s]
        z_cor = z_cor[::s]

        x_grid = x_grid[::s, ::s, ::s]
        y_grid = y_grid[::s, ::s, ::s]
        z_grid = z_grid[::s, ::s, ::s]
        # print len(y_cor), y_grid.shape
        a = 0
        amp = 1e-3
        name = "{0}.spt".format(n)
        with open(folder_name+"/"+name, "w") as text_file:
            for ix, x in enumerate(x_cor):
                for iy, y in enumerate(y_cor):
                    for iz, z in enumerate(z_cor):
                        norm = np.sqrt(x_grid[ix, iy, iz]**2+\
                                y_grid[ix, iy, iz]**2+z_grid[ix, iy, iz]**2)
                        if norm > 1e-30:
                            # print x, y, z, norm, np.sqrt(x_grid[ix, iy, iz]**2+y_grid[ix, iy, iz]**2+z_grid[ix, iy, iz]**2)
                            # print x_grid[ix, iy, iz], y_grid[ix, iy, iz], z_grid[ix, iy, iz]
                            text_file.write(\
                             "draw arrow{0} arrow color [1,0,0] diameter {7} {{ {1},{2},{3} }} {{ {4},{5},{6} }} \n".\
                             format(a, x*au2A, y*au2A, z*au2A,\
                             (x+x_grid[ix, iy, iz]/norm)*au2A,\
                             (y+y_grid[ix, iy, iz]/norm)*au2A,\
                             (z+z_grid[ix, iy, iz]/norm)*au2A,\
                             norm*amp))
                        a += 1

def gradient(phi_xG, gd0, index):
    """
    gradient
    """
    phi = phi_xG.take(index, axis=0)
    gradx = Gradient(gd0, v=0)
    x_grid = gd0.zeros()
    gradx.apply(phi, x_grid)

    grady = Gradient(gd0, v=1)
    y_grid = gd0.zeros()
    grady.apply(phi, y_grid)

    gradz = Gradient(gd0, v=2)
    z_grid = gd0.zeros()
    gradz.apply(phi, z_grid)

    return x_grid, y_grid, z_grid

def gradient_ae(t, gd0, ps, index):
    """
    gradient
    """
    # print 1
    phi = ps[index]
    gradx = Gradient(gd0, v=0)
    x_grid = t.gd.zeros()
    gradx.apply(phi, x_grid)
    # print 2
    grady = Gradient(gd0, v=1)
    y_grid = t.gd.zeros()
    grady.apply(phi, y_grid)
    # print 3
    gradz = Gradient(gd0, v=2)
    z_grid = t.gd.zeros()
    gradz.apply(phi, z_grid)

    return x_grid, y_grid, z_grid

# def div(jx,jy,jz,gd0):
#     """
#     div
#     """
#     gradx = Gradient(gd0, v=0)
#     x_grid =gd0.zeros()
#     gradx.apply(jx, x_grid)

#     grady = Gradient(gd0, v=1)
#     y_grid = gd0.zeros()
#     grady.apply(jy, y_grid)

#     gradz = Gradient(gd0, v=2)
#     z_grid = gd0.zeros()
#     gradz.apply(jz, z_grid)

#     return x_grid + y_grid + z_grid

# def div(jx,jy,jz,dx,dy,dz):
#     """
#     div
#     """

#     gradx = np.gradient(jx,dx,axis=0,edge_order=2)

#     grady = np.gradient(jy,dy,axis=1,edge_order=2)

#     gradz = np.gradient(jz,dz,axis=2,edge_order=2)

#     return gradx + grady + gradz

def div(jx,jy,jz,dx,dy,dz):
    """
    div
    """
    gradx, _, _  = gradientO4(jx,dx,dy,dz)
    _, grady, _, = gradientO4(jy,dx,dy,dz)
    _, _, gradz  = gradientO4(jz,dx,dy,dz)
    return gradx + grady + gradz

def orbital(phi_xG, index):
    """
    Takes a calculator obejct and return the gradient of orbital with index =  index
    and put it onto  the grid - return that grid
    """
    return phi_xG.take(index, axis=0)

def orbital_ae(ps, index):
    """
    Takes a calculator obejct and return the gradient of orbital with index =  index
    and put it onto  the grid - return that grid
    """
    # return t.get_wave_function(index)
    return ps[index]


def orb_grad(phi_xG, gd0, i_orb, j_orb):
    psi = orbital(phi_xG, i_orb)
    x, y, z = gradient(phi_xG, gd0, j_orb)
    return psi*x, psi*y, psi*z

def orb_grad2(phi_xG, i_orb, j_orb, dx, dy, dz):
    psi = orbital(phi_xG, i_orb)
    x,y,z = gradientO4(phi_xG.take(j_orb, axis=0),dx,dy,dz)
    return psi*x, psi*y, psi*z

def orb_grad_ae(t, gd0, ps, i_orb, j_orb):
    # print "orb"
    psi = orbital_ae(ps, i_orb)
    # print "orb done, grad start"
    x, y, z = gradient_ae(t, gd0, ps, j_orb)
    # print "grad don"
    return psi*x, psi*y, psi*z


def plot_orb_grad(atoms, ns, basis, folder_name='./grad',vacuum=3.0, h = 0.20):
    """
    Takes a calculator obejct and return the gradient of orbital with index =  index
    and put it onto  the grid - return that grid
    """
    if exists(folder_name)==False:
            # print "making folder for basis functions"
            call('mkdir %s'%folder_name, shell=True)

    
    for n in ns:
        for m in ns:
            x_grid,y_grid,z_grid = orb_grad(calc,atoms,n,m)
            au2A = 0.529177249
            x_cor = coords(gd0,c=0)
            y_cor = coords(gd0,c=1)
            z_cor = coords(gd0,c=2)

            s = 4
            x_cor = x_cor[::s]
            y_cor = y_cor[::s]
            z_cor = z_cor[::s]

            x_grid = x_grid[::s,::s,::s]
            y_grid = y_grid[::s,::s,::s]
            z_grid = z_grid[::s,::s,::s]

            dim = len(x_cor)
            data = np.array([[0, 0, 0, 0, 0, 0, 0]])
            a = 0 
            
            name = "{0}_{1}.spt".format(n,m)
            with open(folder_name+"/"+name, "w") as text_file:
                for ix, x in enumerate(x_cor):
                    for iy, y in enumerate(y_cor):
                        for iz, z in enumerate(z_cor):
                            # print a, (len(x_cor[::10])*len(y_cor[::10])*len(z_cor[::10]))
                            # data = np.concatenate((data,[[x,y,z,x_grid2[ix,iy,iz],y_grid2[ix,iy,iz],z_grid2[ix,iy,iz],1.]]), axis=0)
                            if np.sqrt(x_grid[ix,iy,iz]**2+y_grid[ix,iy,iz]**2+z_grid[ix,iy,iz]**2) > 5e-4:
            #                     print("draw arrow{0} arrow color [1,0,0] diameter 0.02 {{ {1},{2},{3} }} {{ {4},{5},{6} }}".format(a,x*au2A,y*au2A,z*au2A,(x+x_grid[ix,iy,iz])*au2A,(y+y_grid[ix,iy,iz])*au2A,(z+z_grid[ix,iy,iz])*au2A))
                                text_file.write("draw arrow{0} arrow color [1,0,0] diameter 0.02 {{ {1},{2},{3} }} {{ {4},{5},{6} }} \n".format(a,x*au2A,y*au2A,z*au2A,(x+x_grid[ix,iy,iz])*au2A,(y+y_grid[ix,iy,iz])*au2A,(z+z_grid[ix,iy,iz])*au2A))   
                            a += 1



def coords(gd0, c, pad=True):
    """Return coordinates along one of the three axes.
    Useful for plotting::

        import matplotlib.pyplot as plt
        plt.plot(gd.coords(0), data[:, 0, 0])
        plt.show()
    """
    L = np.linalg.norm(gd0.cell_cv[c])
    N = gd0.N_c[c]
    h = L / N
    p = gd0.pbc_c[c] or pad
    return np.linspace((1 - p) * h, L, N - 1 + p, False)

def plot_total_current(path,co,energy_grid,trans,fL,fR, atoms,h,basis,vacuum,xc,basename):
    current2, x_cor, y_cor, z_cor =\
        np.load(path+"/data/"+basename+"current.npy")
    
    current2 = 2*abs(current2) # spin
    dE = energy_grid[1]-energy_grid[0]
    eV2au = 27.211
    current = (1/eV2au)*(1/np.pi)*np.array([trans[en].real*(fL[en]-fR[en])*dE for en in range(len(energy_grid))]).sum()
    # print current, current2.max()
    # print current2
    np.save(path+'/data/'+basename+'trans.npy',[energy_grid,trans])
    pos = atoms.get_positions()[:,2]
    # print pos
    plt.scatter(pos,[abs(current2).max() for x in pos],c='r',label="atoms")
    plt.plot(z_cor, [current for x in range(len(z_cor))], label="Normal")
    plt.plot(z_cor, current2,label="integral")
    plt.legend()
    # plt.yscale('log')
    # plt.ylim([current2.min(),current 2.max()])
    plt.ylim([current2.min(),current.max()*1.1])
    plt.ylabel(r'I (a.u.)')
    plt.xlabel(r'z ($\AA$)')
    plt.savefig(path+'/plots/'+basename+'current.png')
    plt.close()
    np.save(path+'/data/'+basename+'trans.npy',[energy_grid,trans])


def get_ae_wf(t,atoms,n_basis):
    ps = []
    gd0 = t.gd
    for i in range(n_basis):
        ps.append(t.get_wave_function(i))
    return ps

# def get_lowdin_basis(H,S,GamL,GamR,path,basename):
#     """ Lowdin """
#     # n = len(S)
#     # eig, rot = np.linalg.eig(S)
#     # rot = np.dot(rot / np.sqrt(eig), rot.T.conj())
#     # r_mat = np.identity(n)
#     # r_mat[:] = np.dot(r_mat, rot)
#     # A = r_mat
#     # U = np.linalg.inv(A)
#     # Lowdin_H = np.dot(np.dot(A.T.conj(), H), A)
#     # Lowdin_S = np.dot(np.dot(A.T.conj(), S), A)

#     # Lowdin_GAML = np.dot(np.dot(A.T, GamL), A) 
#     # Lowdin_GAMR = np.dot(np.dot(A.T, GamR), A) 

#     # Lowdin_GAML2 = [np.dot(np.dot(A.T,GAML2[:, :, en]), A) for en in range(len(EGRID))]
#     # Lowdin_GAMR2 = [np.dot(np.dot(A.T,GAMR2[:, :, en]), A) for en in range(len(EGRID))]
#     # Lowdin_GAML2 = np.swapaxes(Lowdin_GAML2, 0, 2)
#     # Lowdin_GAMR2 = np.swapaxes(Lowdin_GAMR2, 0, 2)

#     # Lowdin_GR = ret_gf_ongrid(EGRID, Lowdin_H, Lowdin_S, Lowdin_GAML2, Lowdin_GAMR2)

#     # Lowdin_Sigma_r = -1j/2 * (Lowdin_GAML + Lowdin_GAMR)

#     # Lowdin_Sigmales = lesser_se_ongrid(EGRID, Lowdin_GAML2, Lowdin_GAMR2, FL, FR)
#     # Lowdin_Gles = lesser_gf_ongrid(EGRID, Lowdin_GR, Lowdin_Sigmales)
#     """ """



    # return lowdin_phi_xg, U

def get_divJ(Gr, Sigma_r, GamL, GamR, U, bias, gd0, lowdin_phi_xg):
    n = len(Gr)
    I = np.diag(v=np.ones(n))
    first_term = U.dot(I+Sigma_r.dot(Gr)).dot(GamL)\
                  .dot(Gr.T.conj()).dot(np.linalg.inv(U))\
                            
    second_term = np.linalg.inv(U.T.conj()).dot(Gr).dot(GamL)\
                  .dot(I+Gr.T.conj().dot(Sigma_r.T.conj())).dot(U.T.conj())\
                            

    # test1 = U.dot(I).dot(GamL).dot(Gr.T.conj()).dot(np.linalg.inv(U))
    # test2 = (1j/2)*U.dot(GamL).dot(Gr).dot(GamL).dot(Gr.T.conj()).dot(np.linalg.inv(U))
    # test3 = (1j/2)*U.dot(GamR).dot(Gr).dot(GamL).dot(Gr.T.conj()).dot(np.linalg.inv(U))
    # test4 = U.dot(Sigma_r.dot(Gr)).dot(GamL).dot(Gr.T.conj()).dot(np.linalg.inv(U))\

    # plot_complex_matrix(test1,"1-ft")
    # plot_complex_matrix(test2,"2-ft")  
    # plot_complex_matrix(test3,"3-ft") 
    # plot_complex_matrix(test4,"2+3-ft") 

    divJij = -1j * 1/(2*np.pi) * bias * (first_term - second_term)
    plot_complex_matrix(divJij,"divJij")
    # plot_complex_matrix(first_term,"ft")  
    # plot_complex_matrix(second_term,"st")              
    divJ = 1j*gd0.zeros(1)[0]
    bfs = np.arange(0,n,1)
    for i in range(len(bfs)):
        for j in range(len(bfs)):
            divJ += divJij[i,j] * orbital(lowdin_phi_xg,i) * orbital(lowdin_phi_xg,j)

    return divJ

def put_matrix_onto_grid(mat,basis,gd0):
    n = len(mat)
    bfs = np.arange(0,n,1)
    x_cor = gd0.coords(0)
    y_cor = gd0.coords(1)
    z_cor = gd0.coords(2)
    dx = x_cor[1]-x_cor[0]
    dy = y_cor[1]-y_cor[0]
    dz = z_cor[1]-z_cor[0]
    dA = (x_cor[1]-x_cor[0])*(y_cor[1]-y_cor[0])
    
    res = 1j*gd0.zeros(1)[0]
    for i in range(len(bfs)):
        for j in range(len(bfs)):
            res += mat[i,j] * orbital(basis,i) * orbital(basis,j)*dA
    
    return res

def calc_production(Sigma_l,Gr,Sigma_r,Gles,basis,gd0,e_grid):
    n = len(Gr)
    S = 0j*np.zeros(Gr.shape)
    S1 = 0j*np.zeros(Gr.shape)
    S2 = 0j*np.zeros(Gr.shape)
    S3 = 0j*np.zeros(Gr.shape)
    S4 = 0j*np.zeros(Gr.shape)
    dE = e_grid[1]-e_grid[0]
    for i in range(len(e_grid)):
        S1[:,:,i] = (Sigma_l[:,:,i].dot(Gr[:,:,i].T.conj()))*dE/2*np.pi
        S2[:,:,i] = (Sigma_r[:,:,i].dot(Gles[:,:,i]))*dE/2*np.pi
        S3[:,:,i] = (Gr[:,:,i].dot(Sigma_l[:,:,i]))*dE/2*np.pi
        S4[:,:,i] = (Gles[:,:,i].dot(Sigma_r[:,:,i].T.conj()))*dE/2*np.pi
        S[:,:,i] = (Sigma_l[:,:,i].dot(Gr[:,:,i].T.conj())+Sigma_r[:,:,i].dot(Gles[:,:,i])\
                    -Gr[:,:,i].dot(Sigma_l[:,:,i])-Gles[:,:,i].dot(Sigma_r[:,:,i].T.conj()))\
                    *dE/2*np.pi
    
    S_tot = S.sum(axis=2)
    S1_sum = S1.sum(axis=2)
    S2_sum = S2.sum(axis=2)
    S3_sum = S3.sum(axis=2)
    S4_sum = S4.sum(axis=2)
    Sigma_r_sum = Sigma_r.sum(axis=2)*dE/2*np.pi

    S_on_grid = put_matrix_onto_grid(S_tot,basis,gd0)
    np.save("S_on_grid.npy",S_on_grid)
    Sr_on_grid = put_matrix_onto_grid(Sigma_r_sum,basis,gd0)
    np.save("Sr_on_grid.npy",Sr_on_grid)
    S1_on_grid = put_matrix_onto_grid(S1_sum,basis,gd0)
    np.save("S1_on_grid.npy",S1_on_grid)
    S2_on_grid = put_matrix_onto_grid(S2_sum,basis,gd0)
    np.save("S2_on_grid.npy",S2_on_grid)
    S3_on_grid = put_matrix_onto_grid(S3_sum,basis,gd0)
    np.save("S3_on_grid.npy",S3_on_grid)
    S4_on_grid = put_matrix_onto_grid(S4_sum,basis,gd0)
    np.save("S4_on_grid.npy",S4_on_grid)

    x_cor = gd0.coords(0)
    y_cor = gd0.coords(1)
    z_cor = gd0.coords(2)
    dx = x_cor[1]-x_cor[0]
    dy = y_cor[1]-y_cor[0]
    dz = z_cor[1]-z_cor[0]
    dA = (x_cor[1]-x_cor[0])*(y_cor[1]-y_cor[0])

    bfs = np.arange(0,n,1)
    Sgrid = 1j*gd0.zeros(1)[0]
    for i in range(len(bfs)):
        for j in range(len(bfs)):
            Sgrid += S_tot[i,j] * orbital(basis,i) * orbital(basis,j)*dA


    return Sgrid

# def get_production(Gr, Sigma_r, Gles, Sigma_l, gd0, lowdin_phi_xg):
#     n = len(Gr)
#     S = Sigma_l.dot(Gr.T.conj())+Sigma_r.dot(Gles)\
#         -Gr.dot(Sigma_l)-Gles.dot(Sigma_r.T.conj())
                            
#     bfs = np.arange(0,n,1)
#     Sgrid = 1j*gd0.zeros(1)[0]
#     for i in range(len(bfs)):
#         for j in range(len(bfs)):
#             Sgrid += S[i,j] * orbital(lowdin_phi_xg,i) * orbital(lowdin_phi_xg,j)

#     return Sgrid


def Jc_current(Gles,path,data_basename,fname):
    Mlt = 1j*Gles/(4*np.pi)
    n = len(Mlt)
    np.save(path+data_basename+"Gles_dV.npy", Mlt)
    basis_data = np.load(path+fname+"ao_basis_grid.npy")
    phi_xg, gd0 = basis_data

    jx = gd0.zeros(1)[0]
    jy = gd0.zeros(1)[0]
    jz = gd0.zeros(1)[0]

    x_cor = gd0.coords(0)
    y_cor = gd0.coords(1)
    z_cor = gd0.coords(2)
    dx = x_cor[1]-x_cor[0]
    dy = y_cor[1]-y_cor[0]
    dz = z_cor[1]-z_cor[0]
    bf_list = np.arange(0,n,1)
    # bf_list =  np.arange(6,10,1) # np.arange(37,41,1)
    # print bf_list, "bf_list"

    for k, i in enumerate(tqdm(bf_list,desc="Outer loop current")):
        for l, j in enumerate(bf_list):
            # x1,y1,z1 = orb_grad(phi_xg, gd0,i,j)      
            x1,y1,z1 = orb_grad2(phi_xg,i,j,dx,dy,dz)      
            # x2,y2,z2 = orb_grad(phi_xg, gd0,j,i) 

            jx += 2*Mlt[k,l].real*x1
            jy += 2*Mlt[k,l].real*y1
            jz += 2*Mlt[k,l].real*z1
            # jx += Mlt[k,l].real*(x1-x2)
            # jy += Mlt[k,l].real*(y1-y2)
            # jz += Mlt[k,l].real*(z1-z2)

    dA = (x_cor[1]-x_cor[0])*(y_cor[1]-y_cor[0])
    current = jz.sum(axis=(0,1))*dA       
    
    return current, jx, jy, jz, x_cor, y_cor, z_cor, gd0
      
def gradientO4(f, *varargs):
    """Calculate the fourth-order-accurate gradient of an N-dimensional scalar function.
    Uses central differences on the interior and first differences on boundaries
    to give the same shape.
    Inputs:
      f -- An N-dimensional array giving samples of a scalar function
      varargs -- 0, 1, or N scalars giving the sample distances in each direction
    Outputs:
      N arrays of the same shape as f giving the derivative of f with respect
       to each dimension.
    """
    N = len(f.shape)  # number of dimensions
    n = len(varargs)
    if n == 0:
        dx = [1.0]*N
    elif n == 1:
        dx = [varargs[0]]*N
    elif n == N:
        dx = list(varargs)
    else:
        raise SyntaxError("invalid number of arguments")


    # use central differences on interior and first differences on endpoints

    #print dx
    outvals = []

    # create slice objects --- initially all are [:, :, ..., :]
    slice0 = [slice(None)]*N
    slice1 = [slice(None)]*N
    slice2 = [slice(None)]*N
    slice3 = [slice(None)]*N
    slice4 = [slice(None)]*N

    otype = f.dtype.char
    if otype not in ['f', 'd', 'F', 'D']:
        otype = 'd'

    for axis in range(N):       
        # select out appropriate parts for this dimension
        out = np.zeros(f.shape, f.dtype.char)

        slice0[axis] = slice(2, -2)
        slice1[axis] = slice(None, -4)
        slice2[axis] = slice(1, -3)
        slice3[axis] = slice(3, -1)
        slice4[axis] = slice(4, None)
        # 1D equivalent -- out[2:-2] = (f[:4] - 8*f[1:-3] + 8*f[3:-1] - f[4:])/12.0
        out[slice0] = (f[slice1] - 8.0*f[slice2] + 8.0*f[slice3] - f[slice4])/12.0

        slice0[axis] = slice(None, 2)
        slice1[axis] = slice(1, 3)
        slice2[axis] = slice(None, 2)
        # 1D equivalent -- out[0:2] = (f[1:3] - f[0:2])
        out[slice0] = (f[slice1] - f[slice2])

        slice0[axis] = slice(-2, None)
        slice1[axis] = slice(-2, None)
        slice2[axis] = slice(-3, -1)
        ## 1D equivalent -- out[-2:] = (f[-2:] - f[-3:-1])
        out[slice0] = (f[slice1] - f[slice2])


        # divide by step size
        outvals.append(out / dx[axis])

        # reset the slice object in this dimension to ":"
        slice0[axis] = slice(None)
        slice1[axis] = slice(None)
        slice2[axis] = slice(None)
        slice3[axis] = slice(None)
        slice4[axis] = slice(None)

    if N == 1:
        return outvals[0]
    else:
        return outvals

def nabla(u,dx,dy,dz):
    dxi2,dyi2,dzi2 = dx**(-2),dy**(-2),dz**(-2)
    nx, ny, nz = u.shape
    res = np.zeros(u.shape)
    for i in range(0,nx):
        for j in range(0,ny):
            for k in range(0,nz):
                tmp = 0
                if i == 0:
                    tmp += (u[i+1,j,k] -2*u[i,j,k] + u[i+1,j,k]) * dxi2 
                if i == nx-1:
                    tmp += (u[i-1,j,k] -2*u[i,j,k] + u[i-1,j,k]) * dxi2 
                if i not in (0,nx-1):
                    tmp += (u[i+1,j,k] -2*u[i,j,k] + u[i-1,j,k]) * dxi2 

                if j == 0:
                    tmp += (u[i,j+1,k] -2*u[i,j,k] + u[i,j+1,k]) * dyi2
                if j == ny-1:
                    tmp += (u[i,j-1,k] -2*u[i,j,k] + u[i,j-1,k]) * dyi2
                if j not in (0,ny-1):
                    tmp += (u[i,j+1,k] -2*u[i,j,k] + u[i,j-1,k]) * dyi2

                if k == 0:
                    tmp += (u[i,j,k+1] -2*u[i,j,k] + u[i,j,k+1]) * dzi2
                if k == nz-1:
                    tmp += (u[i,j,k-1] -2*u[i,j,k] + u[i,j,k-1]) * dzi2
                if k not in (0,nz-1):
                    tmp += (u[i,j,k+1] -2*u[i,j,k] + u[i,j,k-1]) * dzi2
                
                res[i,j,k] = tmp
        
    return res


def plot_local_currents(atoms, basis, K , energies, transmission, cutoff=0.1, ignore=[], direction=2):
    from ase.io import write
    from gpaw.lcao.tools import get_bfi2
    dic={}
    for atom in atoms: 
        dic[atom.index]=get_bfi2(atoms.get_chemical_symbols(), basis, [atom.index])
    mask = np.ones( (len(atoms), len(atoms)))
    for bf1 in ignore:
        for bf2 in ignore:
            mask[bf1, bf2]=0.0
    arrows = np.zeros( len(atoms),len(atoms) ) 

    f = open('flux_e%.2f.dat'%energies[0], 'w')
    f.write('wireframe off \n background white; \n')
    #get arrows
    for n, atom1 in enumerate(atoms):
        bfs1 = dic[atom1.index]
        for m, atom2 in enumerate(atoms):
            if m==n:
                arrows[ n,m]=0.0
                continue
            if n in ignore or m in ignore:
                arrows[ n,m]=0.0
                continue
            bfs2 = dic[atom2.index]
            arrows[n,m] = K[:,:].take(bfs1, axis=0).take(bfs2, axis=1).sum(axis= (0,1))
    norm = np.abs(np.multiply(mask, arrows[:,:]) ).max()
    arrows2 = arrows[:,:]/norm                    
    for atom1 in atoms:
        for atom2 in atoms:
            # print direction, "direction"
            exit()
            if atom2.position[direction]-atom1.position[direction]<0.0:
                continue
            if np.abs(arrows[e, atom1.index, atom2.index])<cutoff*transmission[e]:
#                        print "arrow too small, continuing"
                continue
            p2 = np.sign(arrows[e, atom1.index, atom2.index])#positive or negative contribution
            if p2>0.0:
                c = 'red'
                o1 = atom1.position
                o2 = atom2.position
            else:
                c = 'blue'
                o1 =atom2.position
                o2 = atom1.position
            diam=np.abs(arrows2[atom1.index, atom2.index])
#                    print atom1.index, atom2.index, c, diam
            # print "hi"
            f.write('draw arrow%d_%d arrow {%f %f %f} {%f %f %f} diameter %.6f color %s \n'%(atom1.index,atom2.index, o1[0],o1[1], o1[2], o2[0],o2[1], o2[2], diam, c ) ) 
    np.save('arrows.npy', arrows)
    return 

# def get_local_currents(atoms, h, s, ef, G0L,\
                    #    basis, indices, cutoff = 0.10, direction=2, ignore = [], trans_real=None):
def get_local_currents(path,ext,atoms, h, s, ef, G0r, GamL,GamR, bias,\
                       basis, indices, cutoff = 0.10, direction=2, ignore = [], trans_real=None):
        """
        Calculate and plot local currents. 
        The Green's functions and selfenergies must have been dumped on an energy grid. 
        Parameters:
            atoms: ase atoms object
                Atom you wish the local currents calculated for. 
            basis: basis specification
                GPAW style. String or library. Example: 'dzp', {H:'dzp'} etc. 
            indices: list
                indices of energies at which to calculate local currents. 
            dump: Boolean
                if dump=False the local current are not plotted. Instead the tranmission across a surface is calculated. 
            cutoff: float or 1D array
                if local current between atom is smaller than cutoff no arrow is drawn. If integer, transmission is first calculated.   
            direction: integer
                transport direction
            ignore: list:
                indices for atoms to have have plotted local currents. Useful if atom are e.g. close to leads where current is not conserved.    
         """
        dim = len(h)
        K = np.empty( (dim, dim)) 
        arrows = np.zeros( (len(atoms), len(atoms) )   )

        Tt = GamL.dot(G0r).dot(GamR).dot(G0r.T.conj())
        # print Tt.trace(), trans_real
        # exit()
        energy = ef
        # Sigma_el_L= Sigma_el_L_left + Sigma_el_L_right
        G0L = 1j*G0r.dot(GamL).dot(G0r.T.conj()) 
        # G0L *= bias
        # print G0L
         #get lesser Greens function
        # G0L = dot3( G0r , Sigma_el_L , G0r.T.conj() )#np.dot(G0r[:,:,e],np.dot(Sigma_el_L[:,:,e],dagger(G0r[:,:,e])))#
        V = h-energy*s#Sigma_el_r[:,:,index]
        K_mn = np.multiply(V, G0L.T)-np.multiply(V.T, G0L)
        K_mn[range(dim), range(dim)] = 0        #set diagonal zero
        K[:,:]=np.real(K_mn)

        # print K_mn
        # print np.real(K).max()
        # exit()
 #       np.set_printoptions(precision=2, suppress=True)

        #get currents between atoms
        from gpaw.lcao.tools import get_bfi2
        # print "getting arrows..."
        trans = 0
        dic={}
        for atom in atoms: 
            dic[atom.index]=get_bfi2(atoms.get_chemical_symbols(), basis, [atom.index])

        for n, atom1 in enumerate(atoms):
            # print n, "of ", len(atoms)
            bfs1 = dic[atom1.index]
            for m, atom2 in enumerate(atoms):
                if m==n:
                    # print m, "m"
                    continue
                if n in ignore or m in ignore:
                    # print n, "n"
                    continue
                bfs2 = dic[atom2.index]
                arrows[n,m] = K[:,:].take(bfs1, axis=0).take(bfs2, axis=1).sum(axis= (0,1))
#                print arrows[n,m,:]
                if atom1.position[2]<10.0 and atom2.position[2]>10.0:
                    # print arrows[n,m]
                    trans+=arrows[n,m]
        np.save('arrows.npy', arrows)
        # print trans, Tt.trace(), trans_real
        # exit()

        
        f = open(path+ext, 'w')
        f.write('wireframe off \n background white; \n')
        f.write('load "file://{0}/central_region.xyz" \n'.format(path))
        # text_file.write("set defaultdrawarrowscale 0.1 \n")
        norm = np.abs(arrows[:,:]).max()
        # print trans_real
        # print arrows
        # exit()
        arrows2 = arrows/norm                    
        for atom1 in atoms:
            for atom2 in atoms:
                # print np.real(trans_real)
                # if atom2.position[direction]-atom1.position[direction]<0.0:
                #     print "hi"
                #     continue
                # print np.abs(arrows[atom1.index, atom2.index]), cutoff*np.real(trans_real)
                if np.abs(arrows[atom1.index, atom2.index]) < cutoff*np.real(trans_real):
                    # print "arrow too small, continuing"
                    continue
                p2 = np.sign(arrows[atom1.index, atom2.index])#positive or negative contribution
                p2 = atom1.position[2]-atom2.position[2]#positive or negative contribution
                if p2>0.0:
                    c = 'red'
                    o1 = atom1.position
                    o2 = atom2.position
                else:
                    c = 'red'
                    o1 =atom2.position
                    o2 = atom1.position
                diam=np.abs(arrows2[atom1.index, atom2.index])/2
                # print atom1.index, atom2.index, c, diam
                f.write('draw arrow%d_%d arrow {%f %f %f} {%f %f %f} diameter %.6f color %s \n'%(atom2.index,atom1.index, o2[0],o2[1], o2[2], o1[0],o1[1], o1[2], diam, c ) )   
        f.write('rotate 90 \n')
        f.write('background white \n')
        f.write('write file:/{0}plots/interatomic_current.png'.format(path))
        # print "got arrows"
        # exit()
        # plot_local_currents(atoms,basis=basis,K=K,energies=[ef],transmission=trans_real)
        # print trans, "Trans"
        return trans

def plot_current(jx,jy,jz,x,y,z,savename,s,amp,co,path):
    import numpy as np
    a = 0 
    au2A = 0.529177249
    x = x[::s]*au2A
    y = y[::s]*au2A
    z = z[::s]*au2A
    jz = jz[::s,::s,::s]
    jy = jy[::s,::s,::s]      
    jx = jx[::s,::s,::s]   
    with open(savename, "w") as text_file:
        text_file.write('load "file://{0}central_region.xyz" \n'.format(path))
        for ix, x2 in enumerate(x):
            for iy, y2 in enumerate(y):
                for iz, z2 in enumerate(z):
                    norm2 = np.sqrt(jx[ix,iy,iz]**2+jy[ix,iy,iz]**2+jz[ix,iy,iz]**2)
                    norm = np.sqrt(jz[ix,iy,iz]**2)
                    if norm2 > co: #and norm < co*1000:
#                     if norm < 1e-12:
#                     if norm > co:
                        #  print norm, "norm"
                        if jz[ix,iy,iz] > 0.0:
                            color = [1,0,0]
                        else:
                            color = [0,0,1]
                        text_file.write(\
                         "draw arrow{0} arrow color {8} diameter {7} {{ {1},{2},{3} }} {{ {4},{5},{6} }} \n".\
                         format(a,x2,y2,z2,\
                         (x2+jx[ix,iy,iz]/(3*norm2)),\
                         (y2+jy[ix,iy,iz]/(3*norm2)),\
                         (z2+jz[ix,iy,iz]/(3*norm2)),\
                         norm*amp,\
                         color))
                    a += 1
        text_file.write("set defaultdrawarrowscale 0.1 \n")
        text_file.write('rotate 90 \n')
        text_file.write('background white \n')

