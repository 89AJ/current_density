import matplotlib
import pickle
from gpaw import setup_paths
import numpy as np
from tqdm import tqdm
from ase.io import read
import os
import argparse

from utils import *
#from my_poisson_solver import solve_directly, minus_gradient, solve_with_multigrid


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--path',
                    default='../data/c8/',
                    help='path')
parser.add_argument('--ef',
                    default=0.,
                    help='fermi')
args = parser.parse_args()
path = os.path.abspath(args.path) + "/"
ef = float(args.ef)

def main(path = path):
    """
    Calculates the current density for a molecule with different basis set.
    """
    calc(path=path,h=0.2,basis="sz",inv="",grid_size=0,wideband=True,electrode_type="H",ef=ef)

def calc(path, h=0.20, co=1e-10, basis='sz', vacuum=4, xc='PBE',\
         bias=1e-3, ef=0, grid_size=0.2/2.,\
         gamma=1e0, estart=-8, eend=8, es=4e-3, inv="", mode='lcao',\
         wideband=True ,electrode_type="H",kpts=(1,1,1), FDwidth=0.1,\
         correction=False):

    """
    Calculate the current 
    """
    # constants
    eV2au = 1/27.211
    au2eV = 27.211
    au2A = 0.529177249
    # n6 = get_nbasis_electrode(electrode_type,basis)

    a = 25.  # Size of unit cell (Angstrom)
    c = a / 2
    d = 0.74
    # fname = "basis_{0}__xc_{1}__fdwithd_{2}__kpts_{3}__mode_{4}__vacuum_{5}__".format(basis,xc,FDwidth,kpts,mode,vacuum)
    fname = "basis_{0}__xc_{1}__h_{2}__fdwithd_{3}__kpts_{4}__mode_{5}__vacuum_{6}__".format(basis,xc,h,FDwidth,kpts,mode,vacuum)

    basename = "__basis_{0}__h_{1}__cutoff_{2}__xc_{3}__gridsize_{4:.2f}__bias_{5}__ef_{6}__gamma_{7}__e_grid_{8}_{9}_{10}__muliti_grid__type__"\
                .format(basis, h, co, xc, grid_size, bias, ef, gamma, estart, eend, es)
    plot_basename = "plots/"+basename
    data_basename = "data/"+basename

    bias *= eV2au
    ef *= eV2au
    estart *= eV2au
    eend *= eV2au
    es *= eV2au
    gamma *= eV2au

    H_ao, S_ao = pickle.load(open(path+'scat_'+fname+'0.pckl', 'rb'))
    H_ao = H_ao[0, 0]
    S_ao = S_ao[0]
    n = len(H_ao)

    H_cen = H_ao *eV2au
    S_cen = S_ao
    GamL = np.zeros([n,n])
    GamR = np.zeros([n,n])
    GamL[0,0] = gamma
    GamR[n-1,n-1] = gamma



    print("Calculating transmission")
    """transmission"""
    e_grid = np.arange(estart, eend, es)
    Gamma_L = [GamL for en in range(len(e_grid))]
    Gamma_R = [GamR for en in range(len(e_grid))]
    Gamma_L = np.swapaxes(Gamma_L, 0, 2)
    Gamma_R = np.swapaxes(Gamma_R, 0, 2)
    Gr = ret_gf_ongrid(e_grid, H_cen, S_cen, Gamma_L, Gamma_R)
    trans = calc_trans(e_grid, Gr, Gamma_L, Gamma_R)
    plot_transmission(e_grid, trans, path+inv+plot_basename+"trans.png")
    np.save(path+inv+data_basename+'trans_full.npy',[e_grid,trans]) 
    
    # MO - basis
    eig, vec = np.linalg.eig(np.dot(np.linalg.inv(S_cen),H_cen))
    order = np.argsort(eig)
    eig = eig.take(order)
    vec = vec.take(order, axis=1)
    S_mo = np.dot(np.dot(vec.T.conj(), S_cen),vec)
    vec = vec/np.sqrt(np.diag(S_mo))
    S_mo = np.dot(np.dot(vec.T.conj(), S_cen), vec)    
    H_mo = np.dot(np.dot(vec.T, H_cen), vec)
    GamL_mo = np.dot(np.dot(vec.T, GamL), vec)
    GamR_mo = np.dot(np.dot(vec.T, GamR), vec)
    Gamma_L_mo = [GamL_mo for en in range(len(e_grid))]
    Gamma_R_mo = [GamR_mo for en in range(len(e_grid))]
    Gamma_L_mo = np.swapaxes(Gamma_L_mo, 0, 2)
    Gamma_R_mo = np.swapaxes(Gamma_R_mo, 0, 2)


    Gr_mo = ret_gf_ongrid(e_grid, H_mo, S_mo, Gamma_L_mo, Gamma_R_mo)
    trans_mo, trans_mo_trace = calc_trans_mo(e_grid, Gr_mo, Gamma_L_mo, Gamma_R_mo)
    plot_transmission(e_grid, trans_mo_trace, path+inv+plot_basename+"trans_mo.png")
    np.save(path+inv+data_basename+'trans_full_mo.npy',[e_grid,trans_mo_trace])          


    """Current with fermi functions"""
    fR, fL = fermi_ongrid(e_grid, ef, bias)
    dE = e_grid[1]-e_grid[0]
    current_trans = (1/(2*np.pi))*np.array([trans[en].real*(fL[en]-fR[en])*dE for en in range(len(e_grid))]).sum()

    np.save(path+inv+data_basename+"current_trans.npy", current_trans) 

    Sigma_lesser = lesser_se_ongrid(e_grid, Gamma_L, Gamma_R, fL, fR)
    G_lesser = lesser_gf_ongrid(e_grid, Gr, Sigma_lesser)
    G_lesser2 = lesser_gf_ongrid2(e_grid, Gr, Gamma_L)


    np.save(path+inv+data_basename+"matrices.npy",[H_cen,S_cen,Gr,G_lesser,e_grid])

    """Current approx at low temp"""
    Sigma_r = -1j/2. *(GamL + GamR) #+ V_pot

    plot_complex_matrix(Sigma_r, path+inv+"Sigma_r")

    Gr_approx = retarded_gf2(H_cen, S_cen, ef, Sigma_r)


    Sigma_r = 1j*np.zeros(Gamma_L.shape)
    for i in range(len(e_grid)):
        Sigma_r[:,:,i] = -1j/2. * (Gamma_L[:,:,i] + Gamma_R[:,:,i]) #+ V_pot

    basis = np.load(path+fname+"ao_basis_grid.npy") 
    Gles = Gr_approx.dot(GamL).dot(Gr_approx.T.conj()) 
    Gles *= bias

    Sigma_r_mo = -1j/2. *(GamL_mo + GamR_mo)
    Gr_approx_mo = retarded_gf2(H_mo, S_mo, ef, Sigma_r_mo)
    Gles_mo = Gr_approx_mo.dot(GamL_mo).dot(Gr_approx_mo.T.conj())

    plot_complex_matrix(Gles, path+inv+"Gles")

    Tt = GamL.dot(Gr_approx).dot(GamR).dot(Gr_approx.T.conj())
    Tt_mo = GamL_mo.dot(Gr_approx_mo).dot(GamR_mo).dot(Gr_approx_mo.T.conj())
    current_dV = (bias/(2*np.pi))*Tt.trace()

    np.save(path+inv+data_basename+"matrices_dV.npy",[Gr_approx,Gles,GamL])
    np.save(path+inv+data_basename+"matrices_mo_dV.npy",[Gr_approx_mo,Gles_mo,GamL_mo])
    np.save(path+inv+data_basename+"trans_dV.npy", [ef,Tt.trace()])
    np.save(path+inv+data_basename+"trans_mo_dV.npy", [ef,Tt_mo.trace()])
    np.save(path+inv+data_basename+"current_dV.npy", current_dV) 

    basis_data = np.load(path+fname+"ao_basis_grid.npy")
    phi_xg, gd0 = basis_data
    x_cor = gd0.coords(0)
    y_cor = gd0.coords(1)
    z_cor = gd0.coords(2)
    
    """Non corrected current"""
    current_c, jx_c, jy_c, jz_c, x_cor, y_cor, z_cor, gd0 = Jc_current(Gles,path,data_basename,fname)
    np.save(path+inv+data_basename+"current_c_all.npy",np.array([jx_c, jy_c, jz_c,x_cor,y_cor,z_cor]) )
    np.save(path+inv+data_basename+"current_c.npy",np.array([current_c,x_cor,y_cor,z_cor]) )

    dx = (x_cor[1]-x_cor[0])
    dy = (y_cor[1]-y_cor[0])
    dz = (z_cor[1]-z_cor[0])
    
    SI = 31
    EI = -31
    j_z_cut = jz_c[:,:,SI:EI]
    multiplier = 1/(3*j_z_cut[::2,::2,::2].max())
    cut_off = j_z_cut[::2,::2,::2].max()/20.

    plot_current(jx_c,jy_c,jz_c,x_cor,y_cor,z_cor,path+"current.spt",2,multiplier,cut_off,path)

    if correction == True:
        dx = (x_cor[1]-x_cor[0])
        dy = (y_cor[1]-y_cor[0])
        dz = (z_cor[1]-z_cor[0])
        dA = dx*dy
        divJc = div(jx_c,jy_c,jz_c,dx,dy,dz)
    
        divJcz = divJc.sum(axis=(0,1))*dA
        np.save(path+inv+data_basename+"divJcz.npy",np.array([divJcz,x_cor,y_cor,z_cor]) )

        # print "Importing lowdin basis"
        """lowdin"""
        lowdin_phi_xg = np.load(path+fname+"lowdin_basis.npy")
        U = np.load(path+fname+"lowdin_U.npy")

        Sigma_r = -1j/2. *(GamL + GamR)
        divJ = get_divJ(Gr_approx,Sigma_r,GamL,GamR,U,bias,gd0,lowdin_phi_xg[0])
        divJz = divJ.sum(axis=(0,1))*dA

        """ Solving the poisson equation"""
        rho_n = divJ - divJc
        rhoz = rho_n.sum(axis=(0,1))*dA

        tol = 3e-12

        sol = solve_with_multigrid(rho_n.real,x_cor,y_cor,z_cor,tol)

        np.save(path+inv+data_basename+"sol_all_{0}.npy".format(tol),np.array([sol,x_cor,y_cor,z_cor]) )
        solz = sol.sum(axis=(0,1))*dA 

        jx2, jy2, jz2 = gradientO4(sol,dx,dy,dz)
        jz2 *= -1
        jy2 *= -1
        jx2 *= -1

        current_nl_my = jz2.sum(axis=(0,1))*dA  

        divJnl = div(jx2,jy2,jz2,dx,dy,dz)
        divJnlz = divJnl.sum(axis=(0,1))*dA

        np.save(path+inv+data_basename+"divJ.npy",divJ)
        np.save(path+inv+data_basename+"rhoz.npy",np.array([rhoz,x_cor,y_cor,z_cor]) )
        np.save(path+inv+data_basename+"rho_all.npy",np.array([rho_n,x_cor,y_cor,z_cor]) )
        np.save(path+inv+data_basename+"divJz.npy",np.array([divJz,x_cor,y_cor,z_cor]) )
        np.save(path+inv+data_basename+"divJnl_{0}.npy".format(tol),np.array([divJnl]) )
        np.save(path+inv+data_basename+"divJnlz_{0}.npy".format(tol),np.array([divJnlz,x_cor,y_cor,z_cor]) )
        np.save(path+inv+data_basename+"current_all_{0}.npy".format(tol),np.array([jx2,jy2,jz2]) )
        np.save(path+inv+data_basename+"poisson_{0}__solz.npy".format(tol),np.array([solz,x_cor,y_cor,z_cor]) )
        np.save(path+inv+data_basename+"poisson_{0}__current_nl_my.npy".format(tol),np.array([current_nl_my,x_cor,y_cor,z_cor]) )
    else:
        pass

if __name__ == '__main__':
    main()


