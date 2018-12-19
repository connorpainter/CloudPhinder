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
   --alpha_crit=<f>           Critical virial parameter to be considered bound [default: 2.]
   --np=<N>                   Number of snapshots to run in parallel [default: 1]
   --ntree=<N>                Number of particles in a group above which PE will be computed via BH-tree [default: 10000]
"""


#alpha_crit = 2
#potential_mode = False


import h5py
from time import time
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
from pykdgrav.treewalk import GetPotential
from pykdgrav.bruteforce import BruteForcePotential, BruteForcePotential2
from os import path
from natsort import natsorted
import cProfile



#@jit
#def TotalEnergy(xc, mc, vc, hc, uc):
#    phic = Potential(xc, mc, hc)
#    v_well = vc - np.average(vc, weights=mc,axis=0)
#    vSqr = np.sum(v_well**2,axis=1)
#    return np.sum(mc*(0.5*vSqr + 0.5*phic + uc))

#@jit
def PotentialEnergy(xc, mc, vc, hc, uc, tree=None, particles_not_in_tree=None, x=None, m=None, h=None):
    if len(xc) > 1e5: return 0 # this effective sets a cutoff in particle number so we don't waste time on things that are clearly too big to be a GMC
    if len(xc)==1: return -2.8*mc/hc**2 / 2
    if tree:
        phic = pykdgrav.Potential(xc, mc, hc, tree=tree, G=4.301e4)
        if particles_not_in_tree: # have to add the potential from the stuff that's not in the tree
            phic += BruteForcePotential2(xc, x[particles_not_in_tree], m[particles_not_in_tree], h=h[particles_not_in_tree], G=4.301e4)
    else:
        phic = BruteForcePotential(xc, mc, hc, G=4.301e4)
    return np.sum(mc*0.5*phic)

#@profile
def InteractionEnergy(x,m,h, group_a, tree_a, particles_not_in_tree_a, group_b, tree_b, particles_not_in_tree_b):
    xb, mb, hb = x[group_b], m[group_b], h[group_b]    
#    potential_energy = 0.
    if tree_a:
        phi = GetPotential(xb, tree_a, G=4.301e4,theta=.7)
        xa, ma, ha = np.take(x, particles_not_in_tree_a,axis=0), np.take(m, particles_not_in_tree_a,axis=0), np.take(h, particles_not_in_tree_a,axis=0)
        phi += BruteForcePotential2(xb, xa, ma, ha,G=4.301e4)
    else:
        xa, ma, ha = x[group_a], m[group_a], h[group_a]        
        phi = BruteForcePotential2(xb, xa, ma, ha,G=4.301e4)
    potential_energy = (mb*phi).sum()
#    if tree_b:
#        phi = GetPotential(xa, tree_b, G=4.301e4,theta=0.7)
#        phi += BruteForcePotential2(xa, x[particles_not_in_tree_b], m[particles_not_in_tree_b], h[particles_not_in_tree_b],G=4.301e4)
#    else:
#        phi = BruteForcePotential2(xa, xb, mb, hb,G=4.301e4)
#    potential_energy += (ma*phi).sum()
#    potential_energy /= 2
    return potential_energy


#@jit
def KineticEnergy(xc, mc, vc, hc, uc):
#    phic = Potential(xc, mc, hc)
    v_well = vc - np.average(vc, weights=mc,axis=0)
    vSqr = np.sum(v_well**2,axis=1)
    return np.sum(mc*(0.5*vSqr + uc))

#@jit
#def Potential(xc,mc,hc, tree=None):
#    if len(xc)==1: return -2.8*mc/hc
#    if len(xc) > 10000:
#        phic = pykdgrav.Potential(xc, mc, hc, G=4.301e4, parallel=True)
#        phic = pykdgrav.Potential(xc, mc, hc, G=4.301e4, parallel=True)
#    else:
#        phic = BruteForcePotential(xc, mc, hc, G=4.301e4)
#    if tree: phic = 
#    phic = BruteForcePotential(xc, mc, hc, G=4.301e4)
#    return phic

#@jit
def KE(c, x, m, h, v, u):
    xc, mc, vc, hc = x[c], m[c], v[c], h[c]
    v_well = vc - np.average(vc, weights=mc,axis=0)
    vSqr = np.sum(v_well**2,axis=1)
    return (mc*(vSqr/2 + u[c])).sum()

def PE(c, x, m, h, v, u):
    phic = pykdgrav.Potential(x[c], m[c], h[c], G=4.301e4, theta=0.7)
    return 0.5*(phic*m[c]).sum()
    
def VirialParameter(c, x, m, h, v, u):
    ke, pe = KE(c,x,m,h,v,u), PE(c,x,m,h,v,u)
    return(np.abs(ke/pe))
#    xc, mc, vc, hc = x[c], m[c], v[c], h[c]
 #   phic = pykdgrav.Potential(xc,mc, hc, G=4.301e4)
 #   v_well = vc - np.average(vc, weights=mc,axis=0)
 #   vSqr = np.sum(v_well**2,axis=1)
 #   return np.abs(2*(0.5*vSqr.sum() + u[c].sum())/phic.sum())

#@jit
#@profile
def EnergyIncrement(i, c, m, M, x, v, u, h, v_com, tree=None, particles_not_in_tree = None):
    phi = 0.
    if particles_not_in_tree:
        xa, ma, ha = np.take(x,particles_not_in_tree,axis=0), np.take(m,particles_not_in_tree,axis=0), np.take(h,particles_not_in_tree,axis=0)
        phi += BruteForcePotential2(np.array([x[i],]), xa, ma, h=ha, G=4.301e4)[0]
    if tree:
        phi += 4.301e4 * pykdgrav.PotentialWalk(x[i], 0., tree, theta=0.7)
    vSqr = np.sum((v[i]-v_com)**2)
    mu = m[i]*M/(m[i]+M)
    return 0.5*mu*vSqr + m[i]*u[i] + m[i]*phi

#@jit
def KE_Increment(i,m,  v, u, v_com, mtot):
    vSqr = np.sum((v[i]-v_com)**2)
#    M = m[c].sum()
    mu = m[i]*mtot/(m[i]+mtot)
    return 0.5*mu*vSqr + m[i]*u[i]

#@jit
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
#@profile
def ParticleGroups(x, m, rho, phi, h, u, v, zz, ids, cluster_ngb=32):
#    print(u)
    phi = -rho
#    plt.hist(np.log10(-phi)); plt.show()
    order = phi.argsort()
    phi[:] = phi[order]
    x[:], m[:], v[:], h[:], u[:], rho[:], ids[:], zz[:] = x[order], m[order], v[order], h[order], u[order], rho[order], ids[order], zz[order]

    ngbdist, ngb = cKDTree(x).query(x,min(cluster_ngb, len(x)), distance_upper_bound=0.1)#)

#    print((ngbdist/h[:,np.newaxis])[ngbdist[:,-1] > 3*h])
#    exit()
    max_group_size = 0
    groups = {}
    particles_since_last_tree = {}
    group_tree = {}
    group_energy = {}
    group_KE = {}
    group_PE = {}
    COM = {}
    v_COM = {}
    masses = {}
    positions = {}
    softenings = {}
    bound_groups = {}
    bound_subgroups = {}
    assigned_group = -np.ones(len(x),dtype=np.int32)
    for i in range(len(x)): # do it one particle at a time, in decreasing order of density
        if not i%10000: 
#            print(i,len(x),max_group_size)
            max_group_size=0
#            if masses.values(): print(max(masses.values()))
        ngbi = ngb[i]#[(ngbdist[i] < 3*h[i]) * (ngbdist[i] < 3*h[ngb[i]])]
        ngbi = ngbi[ngbi < len(x)] # , ngb[ngbi < len(x)]
        

        lower = phi[ngbi] < phi[i]
        nlower = lower.sum()
        if nlower == 0: # if this is the densest particle in the kernel, let's create our own group with blackjack and hookers
            groups[i] = [i,]
            group_tree[i] = None
            assigned_group[i] = i
            group_energy[i] = m[i]*u[i]# -2.8*m[i]**2/h[i] / 2 # kinetic + potential energy
            group_KE[i] = m[i]*u[i]
#            if np.abs(group_energy[i]-group_KE[i]) and 2*group_KE[i]/np.abs(group_energy[i] - group_KE[i]) < alpha_crit: bound_groups[i] = [i,]
            v_COM[i] = v[i]
            COM[i] = x[i]
            masses[i] = m[i]
            positions[i] = x[i]
            softenings[i] = h[i]
            particles_since_last_tree[i] = [i,]
        elif nlower == 1: # if there is only one denser particle, we belong to its group
            assigned_group[i] = assigned_group[ngbi[lower][0]]
            groups[assigned_group[i]].append(i)
        else: # o fuck we're at a saddle point, let's consider both respective groups
            a, b = ngbi[lower][:2]
            group_index_a, group_index_b = assigned_group[a], assigned_group[b]
            if masses[group_index_a] < masses[group_index_b]: group_index_a, group_index_b = group_index_b, group_index_a # make sure group a is the bigger one, switching labels if needed

            if group_index_a == group_index_b:  # if both dense boyes belong to the same group, that's the group for us too
                assigned_group[i] = group_index_a
            else:
            #OK, we're at a saddle point, so we need to merge those groups
                group_a, group_b = groups[group_index_a], groups[group_index_b]
#                print(len(group_a))                
                ma, mb = masses[group_index_a], masses[group_index_b]
                xa, xb = COM[group_index_a], COM[group_index_b] 
                va, vb = v_COM[group_index_a], v_COM[group_index_b]
#                Ea, Eb = group_energy[group_index_a], group_energy[group_index_b]
                group_ab = group_a + group_b
#                if(len(group_a) > 1 and len(group_b)>1): print(group_energy[group_index_a] - group_KE[group_index_a],  PE(group_a, x, m, h, v, u), group_energy[group_index_b] - group_KE[group_index_b], PE(group_b,x,m,h,v,u))
                groups[group_index_a] = group_ab
                
                #group_energy[group_index_a] = group_KE[group_index_a] + group_KE[group_index_b]
                group_energy[group_index_a] += group_energy[group_index_b]
                group_KE[group_index_a] += group_KE[group_index_b]
                group_energy[group_index_a] += 0.5*ma*mb/(ma+mb) * np.sum((va-vb)**2) # energy due to relative motion: 1/2 * mu * dv^2
                group_KE[group_index_a] += 0.5*ma*mb/(ma+mb) * np.sum((va-vb)**2)
                group_energy[group_index_a] += InteractionEnergy(x,m,h, group_a, group_tree[group_index_a], particles_since_last_tree[group_index_a], group_b, group_tree[group_index_b], particles_since_last_tree[group_index_b]) # mutual interaction energy; we've already counted their individual binding energies

     #           group_energy[group_index_a] += PE(group_ab, x, m, h, v, u)

                if len(group_a) > ntree: # we've got a big group, so we should probably do stuff with the tree
                    if len(group_b) > 128: # if the smaller of the two is also large, let's build a whole new tree, and a whole new adventure
                        
                        group_tree[group_index_a] = pykdgrav.ConstructKDTree(np.take(x,group_ab,axis=0), np.take(m,group_ab), np.take(h,group_ab))
                        particles_since_last_tree[group_index_a][:] = []
                    else:  # otherwise we want to keep the old tree from group a, and just add group b to the list of particles_since_last_tree
                        particles_since_last_tree[group_index_a] += group_b
                else:
                    particles_since_last_tree[group_index_a][:] = group_ab[:]
                    
                if len(particles_since_last_tree[group_index_a]) > ntree:
                    group_tree[group_index_a] = pykdgrav.ConstructKDTree(np.take(x,group_ab,axis=0), np.take(m,group_ab), np.take(h,group_ab))
                    particles_since_last_tree[group_index_a][:] = []                    
        
#                group_KE[group_index_a] = gKineticEnergy(np.take(x,group_ab,axis=0), np.take(m,group_ab), v[group_ab], np.take(h,group_ab), u[group_ab])
#                group_energy[group_index_a] = KE(group_ab, x, m, h, v, u) + PE(group_ab, x, m, h, v, u)
                #+ PotentialEnergy(np.take(x,group_ab,axis=0), np.take(m,group_ab), v[group_ab], np.take(h,group_ab), u[group_ab], group_tree[group_index_a], particles_since_last_tree[group_index_a], x, m,h) #Ea + Eb + dE
                
                COM[group_index_a] = (ma*xa + mb*xb)/(ma+mb)
                v_COM[group_index_a] = (ma*va + mb*vb)/(ma+mb)
                masses[group_index_a] = ma + mb
                groups.pop(group_index_b,None)
                assigned_group[i] = group_index_a
                assigned_group[assigned_group==group_index_b] = group_index_a
                # if this new group is bound, we can delete the old bound group
                if abs(2*group_KE[group_index_a]/np.abs(group_energy[group_index_a] - group_KE[group_index_a]) - alpha_crit) < 0.1:
#                    print(masses[group_index_a], VirialParameter(group_ab, x, m, h, v, u))
                    bound_groups[group_index_a] = group_ab[:]
                    bound_groups.pop(group_index_b,None)
                    #-group_KE[group_index_a]
##                    print(len(group_ab), group_KE[group_index_a], KE(group_ab, x, m, h, v, u), group_energy[group_index_a]-group_KE[group_index_a], PE(group_ab, x, m, h, v, u))         
                for d in groups, particles_since_last_tree, group_tree, group_energy, group_KE, COM, v_COM, masses, bound_groups, bound_subgroups:
                    d.pop(group_index_b, None)
                
            groups[group_index_a].append(i)
            max_group_size = max(max_group_size, len(particles_since_last_tree[group_index_a]))
        if nlower > 0: # assuming we've added a particle to an existing group, we have to update stuff
            g = assigned_group[i]
            mgroup = masses[g]
            group_KE[g] += KE_Increment(i, m, v, u, v_COM[g], mgroup)
            group_energy[g] += EnergyIncrement(i, groups[g][:-1], m, mgroup, x, v, u, h, v_COM[g], group_tree[g], particles_since_last_tree[g])
            if abs(2*group_KE[g]/np.abs(group_energy[g] - group_KE[g]) - alpha_crit) < 0.1:
                bound_groups[g] = groups[g][:]
            v_COM[g] = (m[i]*v[i] + mgroup*v_COM[g])/(m[i]+mgroup)
            masses[g] += m[i]
            particles_since_last_tree[g].append(i)
            if len(particles_since_last_tree[g]) > ntree:
                group_tree[g] = pykdgrav.ConstructKDTree(x[groups[g]], m[groups[g]], h[groups[g]])
                particles_since_last_tree[g][:] = []
            max_group_size = max(max_group_size, len(particles_since_last_tree[g]))
#            print(group_KE[g], KE(groups[g], x, m, h, v, u), group_energy[g]-group_KE[g], PE(groups[g], x, m, h, v, u))

        
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
    datafile_name = "bound_%s_n%g_alpha%g.dat"%(n,nmin,alpha_crit)
#    if path.isfile(datafile_name): return
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
#    print(rho)
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
        phi = pykdgrav.Potential(x, m, h_ags)

#    phi = np.ones_like(rho)
    
    x, m, rho, phi, h_ags, u, v, zz = np.float64(x), np.float64(m), np.float64(rho), np.float64(phi), np.float64(h_ags), np.float64(u), np.float64(v), np.float64(zz)

    t = time()
    groups, bound_groups, assigned_group = ParticleGroups(x, m, rho, phi, h_ags, u, v, zz, ids, cluster_ngb=cluster_ngb)
    t = time() - t
    print("Time: %g"%t)
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
    SaveArrayDict(datafile_name, bound_data)
#    SaveArrayDict(filename.split("snapshot")[0] + "unbound_%s.dat"%n, unbound_data)

    
alpha_crit = float(docopt(__doc__)["--alpha_crit"])
ntree = int(docopt(__doc__)["--ntree"])

def main():
    options = docopt(__doc__)
    print(options)
    nproc=int(options["--np"])
    if nproc==1:
        for f in options["<files>"]:
            print(f)
            ComputeClouds(f, options)
#            cProfile.runctx("ComputeClouds('%s',options)"%f, globals(), locals(),filename=None)
    else:
        print(natsorted(options["<files>"]))
        Parallel(n_jobs=nproc)(delayed(ComputeClouds)(f,options) for f in options["<files>"])

if __name__ == "__main__": main()
