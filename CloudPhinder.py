#!/usr/bin/env python
"""                                                                            
: Variant of Volker Springel's SUBFIND algorithm that identifies the largest possible self-gravitating structures of a certain particle type. THIS ASSUMES PHYSICAL UNITS IN THE SNAPSHOT!

Usage: CloudPhinder.py <files> ... [options]

Options:                                                                       
   -h --help                  Show this screen.
   --ptype=<N>                GIZMO particle type to analyze [default: 0]
   --G=<G>                    Gravitational constant to use; should be consistent with what was used in the simulation. [default: 4.301e4]
   --boxsize=<L>              Box size of the simulation; for neighbour-search purposes. [default: None]
   --cluster_ngb=<N>          Length of particle's neighbour list. [default: 32]
   --nmin=<n>                 Minimum particle number density to cut at, in cm^-3 [default: 1]
   --softening=<L>            Force softening for potential, if species does not have adaptive softening. [default: 1e-5]
   --fuzz=<L>                 Randomly perturb particle positions by this small fraction to avoid problems with particles at the same position in 32bit floating point precision data [default: 0]
   --np=<N>                   Number of snapshots to run in parallel [default: 1]
"""


alpha_crit = 2
#potential_mode = False


import h5py
from numba import jit, vectorize
from joblib import Parallel, delayed
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from matplotlib import pyplot as plt
import numpy as np
from sys import argv
import meshoid
from docopt import docopt
from collections import OrderedDict
import pykdgrav
from pykdgrav.bruteforce import BruteForcePotential
from os import path
from natsort import natsorted

@jit
def TotalEnergy(xc, mc, vc, hc, uc):
    phic = Potential(xc, mc, hc)
    v_well = vc - np.average(vc, weights=mc,axis=0)
    vSqr = np.sum(v_well**2,axis=1)
    return np.sum(mc*(0.5*vSqr + 0.5*phic + uc))

@jit
def PotentialEnergy(xc, mc, vc, hc, uc):
    phic = Potential(xc, mc, hc)
    return np.sum(mc*0.5*phic)

@jit
def KineticEnergy(xc, mc, vc, hc, uc):
#    phic = Potential(xc, mc, hc)
    v_well = vc - np.average(vc, weights=mc,axis=0)
    vSqr = np.sum(v_well**2,axis=1)
    return np.sum(mc*(0.5*vSqr + uc))

@jit
def Potential(xc,mc,hc):
    if len(xc)==1: return -2.8*mc/hc
    if len(xc) > 10000:
#        phic = pykdgrav.Potential(xc, mc, hc, G=4.301e4, parallel=True)
        phic = pykdgrav.Potential(xc, mc, hc, G=4.301e4, parallel=True)
    else:
#        phic = BruteForcePotential(xc, mc, hc, G=4.301e4)
        phic = BruteForcePotential(xc, mc, hc, G=4.301e4)
    return phic

@jit
def VirialParameter(c):
    xc, mc, vc, hc = x[c], m[c], v[c], h[c]
    phic = Potential(xc,mc, hc)
    v_well = vc - np.average(vc, weights=mc,axis=0)
    vSqr = np.sum(v_well**2,axis=1)
    return np.abs(2*(0.5*vSqr.sum() + u[c].sum())/phic.sum())

@jit
def EnergyIncrement(i, c, m, x, v, u, v_com):
    phi = -4.301e4 * np.sum(m[c]/cdist([x[i],],x[c]))
    vSqr = np.sum((v[i]-v_com)**2)
    M = m[c].sum()
    mu = m[i]*M/(m[i]+M)
    return 0.5*mu*vSqr + m[i]*u[i] + m[i]*phi

@jit
def KE_Increment(i, c, m, x, v, u, v_com):
    vSqr = np.sum((v[i]-v_com)**2)
    M = m[c].sum()
    mu = m[i]*M/(m[i]+M)
    return 0.5*mu*vSqr + m[i]*u[i]

@jit
def PE_Increment(i, c, m, x, v, u, v_com):
    phi = -4.301e4 * np.sum(m[c]/cdist([x[i],],x[c]))
    return m[i]*phi

def SaveArrayDict(path, arrdict):
    """Takes a dictionary of numpy arrays with names as the keys and saves them in an ASCII file with a descriptive header"""
    header = ""
    offset = 0
    
    for i, k in enumerate(arrdict.keys()):
        if type(arrdict[k])==list: arrdict[k] = np.array(arrdict[k])
        if len(arrdict[k].shape) == 1:
            header += "(%d) "%offset + k + "\n"
            offset += 1
        else:
            header += "(%d-%d) "%(offset, offset+arrdict[k].shape[1]-1) + k + "\n"
            offset += arrdict[k].shape[1]
            
    data = np.column_stack([b for b in arrdict.values()])
    data = data[(-data[:,0]).argsort()] 
    np.savetxt(path, data, header=header,  fmt='%.15g', delimiter='\t')


#@jit 
def ParticleGroups(x, m, rho, phi, h, u, v, zz, ids, cluster_ngb=32):
    phi = -rho
#    plt.hist(np.log10(-phi)); plt.show()
    order = phi.argsort()
    phi[:] = phi[order]
    x[:], m[:], v[:], h[:], u[:], rho[:], ids[:], zz[:] = x[order], m[order], v[order], h[order], u[order], rho[order], ids[order], zz[order]

    ngbdist, ngb = cKDTree(x).query(x,min(cluster_ngb, len(x)))
#    print(ngbdist, ngb)
#    sigma = np.log10(meshoid.meshoid(x,m).SurfaceDensity(size=1000.,center=[0,0,0],res=4000).T*1e4)/3
#    sigma = np.clip(sigma,0,1)
#    plt.pcolormesh(np.linspace(-500,500,4000),np.linspace(-500,500,4000),sigma)
#    plt.show()
    
    groups = {}
    group_energy = {}
    group_KE = {}
#    group_PE = {}
    COM = {}
    v_COM = {}
    masses = {}
    bound_groups = {}
    bound_subgroups = {}
    assigned_group = -np.ones(len(x),dtype=np.int32)
    for i in range(len(x)):
        #if assigned_group[i] > 0: continue
        if not i%1000: 
            print(i)
#            print([len(b) for b in bound_groups.values()])
#                 len(b)
#            for k in groups.keys():
                # should first check if virial param is < ~ 10, otherwise we will eat up all our CPU time computing potential
                # for the low-density contours that surround everything
#                group_energy[k] = TotalEnergy(groups[k])
#                if group_energy[k] < 0:
#                    bound_groups[k] = groups[k][:]
#        print(ngbdist)
#        print(3*h[ngb[i]])
        ngbi = ngb[i][(ngbdist[i] < 3*h[i]) + (ngbdist[i] < 3*h[ngb[i]])]
#        ngbi = ngb[i]
        lower = phi[ngbi] < phi[i]
        nlower = lower.sum()
        if nlower == 0:
            groups[i] = [i,]
            assigned_group[i] = i
            group_energy[i] = m[i]*u[i] -2.8*m[i]**2/h[i]
            group_KE[i] = m[i]*u[i]
            if np.abs(group_energy[i]-group_KE[i]) and 2*group_KE[i]/np.abs(group_energy[i] - group_KE[i]) < alpha_crit: bound_groups[i] = [i,]
            v_COM[i] = v[i]
            COM[i] = x[i]
            masses[i] = m[i]
        elif nlower == 1:
            assigned_group[i] = assigned_group[ngbi[lower][0]]
            groups[assigned_group[i]].append(i)
        else:
            a, b = ngbi[lower][:2]
            group_index_a, group_index_b = assigned_group[a], assigned_group[b]
            if group_index_a == group_index_b: 
                assigned_group[i] = group_index_a
            else:
            #OK, we're at a saddle point, so we need to consider merging the groups if energetically favourable,
            # and if not then add the particle to the more energetically favourable group
                group_a, group_b = groups[group_index_a], groups[group_index_b]
                ma, mb = masses[group_index_a], masses[group_index_b]
                xa, xb = COM[group_index_a], COM[group_index_b] #np.average(x[group_a],axis=0,weights=ma), np.average(x[group_b],axis=0,weights=mb)
                va, vb = v_COM[group_index_a], v_COM[group_index_b]
                dE = .5 * ma*mb/(ma+mb) * np.sum((va-vb)**2) - 4.3e4 * ma*mb/np.sum((xa-xb)**2)**0.5 #energy created by merging groups from relative motion and mutual binding energy
                Ea, Eb = group_energy[group_index_a], group_energy[group_index_b]
                group_ab = group_a + group_b
                groups[group_index_a] = group_ab
                group_KE[group_index_a] = KineticEnergy(x[group_ab], m[group_ab], v[group_ab], h[group_ab], u[group_ab])
                group_energy[group_index_a] = group_KE[group_index_a] + PotentialEnergy(x[group_ab], m[group_ab], v[group_ab], h[group_ab], u[group_ab]) #Ea + Eb + dE
                
                COM[group_index_a] = (ma*xa + mb*xb)/(ma+mb)
                v_COM[group_index_a] = (ma*va + mb*vb)/(ma+mb)
                masses[group_index_a] = ma + mb
                groups.pop(group_index_b,None)
                assigned_group[i] = group_index_a
                assigned_group[assigned_group==group_index_b] = group_index_a
                # if this new group is bound, we can delete the old bound group
                if 2*group_KE[group_index_a]/np.abs(group_energy[group_index_a] - group_KE[group_index_a]) < alpha_crit:
                    bound_groups[group_index_a] = group_ab[:]
                    bound_groups.pop(group_index_b,None)
                
            groups[group_index_a].append(i)

        if nlower > 0:
            g = assigned_group[i]
            mgroup = masses[g]
            group_KE[g] += KE_Increment(i, groups[g][:-1], m, x, v, u, v_COM[g])
            group_energy[g] += EnergyIncrement(i, groups[g][:-1], m, x, v, u, v_COM[g])
            if 2*group_KE[g]/np.abs(group_energy[g] - group_KE[g]) < alpha_crit:
                bound_groups[g] = groups[g][:]
            v_COM[g] = (m[i]*v[i] + mgroup*v_COM[g])/(m[i]+mgroup)
            masses[g] += m[i]
        
#    print("initial grouping complete")
    # OK, now make a pass through the bound groups to absorb any subgroups within a larger group 
    assigned_bound_group = -np.ones(len(x),dtype=np.int32)
    largest_assigned_group = -np.ones(len(x),dtype=np.int32)
    
    for k, g in bound_groups.items():
        for i in g:
            if len(g) > largest_assigned_group[i]: 
                assigned_bound_group[i] = k
                largest_assigned_group[i] = len(g)
#    print("secondary grouping complete")
    bound_groups = {}

    for i in range(len(assigned_bound_group)):
#        if not i%1000: print(i, len(bound_groups.keys()))
        a = assigned_bound_group[i]
        if a < 0: continue

        if a in bound_groups.keys(): bound_groups[a].append(i)
        else: bound_groups[a] = [i,]
#    print("tertiary grouping complete")
    return groups, bound_groups, assigned_group

    
def ComputeClouds(filename, options):
    n = filename.split("_")[1].split(".")[0]
    nmin = float(options["--nmin"])
    if path.isfile("bound_%s_%g.dat"%(n,nmin)): return
    print(filename)
    cluster_ngb = int(float(options["--cluster_ngb"]) + 0.5)
    G = float(options["--G"])
    boxsize = options["--boxsize"]
    ptype = "PartType"+ options["--ptype"]

#    recompute_potential = options["--recompute_potential"]
    softening = float(options["--softening"])
    if boxsize != "None":
        boxsize = float(boxsize)
    else:
        boxsize = None
    fuzz = float(options["--fuzz"])


    if path.isfile(filename):
        F = h5py.File(filename,'r')
    else:
        print("Could not find "+filename)
        return
    if not ptype in F.keys():
        print("Particles of desired type not found!")
    
    m = np.array(F[ptype]["Masses"])
    criteria = np.ones(len(m),dtype=np.bool)

    if len(m) < cluster_ngb:
        print("Not enough particles for meaningful cluster analysis!")
        return

    x = np.array(F[ptype]["Coordinates"])

    ids = np.array(F[ptype]["ParticleIDs"])
    u = (np.array(F[ptype]["InternalEnergy"]) if ptype=="PartType0" else np.zeros_like(m))
    if "Density" in F[ptype].keys():
        rho = np.array(F[ptype]["Density"])
    else:
        rho = meshoid.meshoid(x,m,des_ngb=cluster_ngb).Density()
    print(rho)
        #ngbdist = meshoid.meshoid(x,m,des_ngb=cluster_ngb).ngbdist
    zz = np.array(F[ptype]["Metallicity"])
#    rho = meshoid.meshoid(x,m).KernelAverage(rho)
#    c = np.average(x,axis=0,weights=rho**2)
#    x = x - c
#    print(rho.max())
 
    criteria *= (rho*145.7 > nmin) # only look at dense gas (>nmin cm^-3)
#    criteria *= np.max(np.abs(x),axis=1) < 50.
    print("%g particles denser than %g cm^-3" %(criteria.sum(),nmin))  #(np.sum(rho*147.7>nmin), nmin))
    if not criteria.sum(): return
    m = m[criteria]
    x = x[criteria]
    u = u[criteria]
    v = np.array(F[ptype]["Velocities"])[criteria]
    rho = rho[criteria]
    ids = ids[criteria]
    zz = zz[criteria]
#    ngbdist, ngb = ngbdist[criteria]
    if fuzz: x += np.random.normal(size=x.shape)*x.std()*fuzz

    if "AGS-Softening" in F[ptype].keys():
        h_ags = np.array(F[ptype]["AGS-Softening"])[criteria]
    elif "SmoothingLength" in F[ptype].keys():
        h_ags = np.array(F[ptype]["SmoothingLength"])[criteria]
    else:
       # h_ags = meshoid.meshoid(x,m,des_ngb=cluster_ngb).SmoothingLength() #np.ones_like(m)*softening #(m/rho * cluster_ngb)**(1./3) #
        h_ags = np.ones_like(m)*softening
    if "Potential" in F[ptype].keys(): # and not recompute_potential:
        phi = np.array(F[ptype]["Potential"])[criteria]
    else:
        phi = Potential(x, m, h_ags)

#    phi = np.ones_like(rho)
    
    x, m, rho, phi, h_ags, u, v, zz = np.float64(x), np.float64(m), np.float64(rho), np.float64(phi), np.float64(h_ags), np.float64(u), np.float64(v), np.float64(zz)

    groups, bound_groups, assigned_group = ParticleGroups(x, m, rho, phi, h_ags, u, v, zz, ids, cluster_ngb=cluster_ngb)
    print("Done grouping. Computing group properties...")
    groupmass = np.array([m[c].sum() for c in bound_groups.values() if len(c)>10])
    groupid = np.array([c for c in bound_groups.keys() if len(bound_groups[c])>10])
    groupid = groupid[groupmass.argsort()[::-1]]
    bound_groups = OrderedDict(zip(groupid, [bound_groups[i] for i in groupid]))
#    exit()

    # Now we analyze the clouds and dump their properties

    bound_data = OrderedDict()
    bound_data["Mass"] = []
    bound_data["Center"] = []
    bound_data["PrincipalAxes"] = []
    bound_data["Reff"] = []
    bound_data["HalfMassRadius"] = []
    bound_data["NumParticles"] = []
#    bound_data["SigmaEff"] = []
    


    Fout = h5py.File("Clouds_%s_%g.hdf5"%(n,nmin), 'w')

    i = 0
    for k,c in bound_groups.items():
        bound_data["Mass"].append(m[c].sum())
        bound_data["NumParticles"].append(len(c))
        bound_data["Center"].append(np.average(x[c], weights=m[c], axis=0))
        dx = x[c] - bound_data["Center"][-1]
        eig = np.linalg.eig(np.cov(dx.T))[0]
        bound_data["PrincipalAxes"].append(np.sqrt(eig))
        bound_data["Reff"].append(np.prod(np.sqrt(eig))**(1./3))
        r = np.sum(dx**2, axis=1)**0.5
        bound_data["HalfMassRadius"].append(np.median(r))
#        sigma_eff = meshoid.meshoid(x[c],m[c],h_ags[c]).SurfaceDensity(size=4*bound_data["HalfMassRadius"][-1],center=bound_data["Center"][-1], res=400)
        
#        bound_data["SigmaEff"].append(np.average(sigma_eff,weights=sigma_eff)*1e4)

        cluster_id = "Cloud"+ ("%d"%i).zfill(int(np.log10(len(bound_groups))+1))

        N = len(c)

        Fout.create_group(cluster_id)
        fids = np.array(F[ptype]["ParticleIDs"])
        idx = np.in1d(fids, ids[c])
        for k in F[ptype].keys():
            Fout[cluster_id].create_dataset(ptype+"/"+k, data = np.array(F[ptype][k])[idx])
#            Fout[cluster_id].create_dataset("Coordinates", data=x[c])
            # Fout[cluster_id].create_dataset("Velocities", data=v[c])
            # Fout[cluster_id].create_dataset("Metallicity", data=zz[c])
            # Fout[cluster_id].create_dataset("Masses", data=m[c])
            # Fout[cluster_id].create_dataset("InternalEnergy", data=u[c])
            # Fout[cluster_id].create_dataset("Density", data=rho[c])
            # Fout[cluster_id].create_dataset("SmoothingLength", data=h_ags[c])
            # Fout[cluster_id].create_dataset("ParticleIDs", data=ids[c])
        i += 1
        

    print("Done grouping bound clusters!")

#                print "Reduced chi^2: %g Relative mass error: %g"%(EFF_rChiSqr,EFF_dm/mc[bound].sum())
#    cluster_masses = np.array(bound_data["Mass"])
#    bound_clusters = [b for b in bound_groups.values()]
     #np.array(bound_clusters)[np.array(bound_data["Mass"]).argsort()[::-1]]
    
    # write to Clusters_xxx.hdf5
#    for i, c in enumerate(bound_clusters):

        
    Fout.close()
    F.close()
    
    #now save the ascii data files
    SaveArrayDict("bound_%s_%g.dat"%(n,nmin), bound_data)
#    SaveArrayDict(filename.split("snapshot")[0] + "unbound_%s.dat"%n, unbound_data)

    
    
def main():
    options = docopt(__doc__)
    nproc=int(options["--np"])
    if nproc==1:
        for f in options["<files>"]:
            print(f)
            ComputeClouds(f, options)
    else:
        print(natsorted(options["<files>"]))
        Parallel(n_jobs=nproc)(delayed(ComputeClouds)(f,options) for f in options["<files>"])

if __name__ == "__main__": main()
