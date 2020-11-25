#!/usr/bin/env python
"""                                                                            
Algorithm that identifies the largest possible self-gravitating structures of a certain particle type. This, newer version relies on the load_from_snapshot routine from GIZMO.

Usage: CloudPhinder.py <snapshots> ... [options]

Options:                                                                       
   -h --help                  Show this screen.

   --outputfolder=<name>      Specifies the folder to save the outputs to, None defaults to the same location as the snapshot [default: None]
   --ptype=<N>                GIZMO particle type to analyze [default: 0]
   --G=<G>                    Gravitational constant to use; should be consistent with what was used in the simulation. [default: 4.301e4]
   --boxsize=<L>              Box size of the simulation; for neighbour-search purposes. [default: None]
   --cluster_ngb=<N>          Length of particle's neighbour list. [default: 32]
   --nmin=<n>                 Minimum H number density to cut at, in cm^-3 [default: 1]
   --softening=<L>            Force softening for potential, if species does not have adaptive softening. [default: 1e-5]
   --fuzz=<L>                 Randomly perturb particle positions by this small fraction to avoid problems with particles at the same position in 32bit floating point precision data [default: 0]
   --alpha_crit=<f>           Critical virial parameter to be considered bound [default: 2.]
   --np=<N>                   Number of snapshots to run in parallel [default: 1]
   --ntree=<N>                Number of particles in a group above which PE will be computed via BH-tree [default: 10000]
   --overwrite                Whether to overwrite pre-existing clouds files
   --units_already_physical   Whether to convert units to physical from comoving
   --max_linking_length=<L>   Maximum radius for neighbor search around a particle [default: 1e100]
"""
#   --snapdir=<name>           path to the snapshot folder, e.g. /work/simulations/outputs

#alpha_crit = 2



## from builtin
import h5py
from time import time,sleep
from os import path,getcwd,mkdir
from glob import glob
from collections import OrderedDict

from matplotlib import pyplot as plt

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from numba import jit, njit


from sys import argv
from docopt import docopt

from multiprocessing import Pool

## from github/mikegrudic
import pykdgrav
from pykdgrav.treewalk import GetPotential
from pykdgrav.kernel import *
from Meshoid import Meshoid

## from GIZMO
import load_from_snapshot #routine to load snapshots from GIZMo files

## global variables
potential_mode = False

@njit
def BruteForcePotential2(x_target,x_source, m,h=None,G=1.):
    if h is None: h = np.zeros(x_target.shape[0])
    potential = np.zeros(x_target.shape[0])
    for i in range(x_target.shape[0]):
        for j in range(x_source.shape[0]):
            dx = x_target[i,0]-x_source[j,0]
            dy = x_target[i,1]-x_source[j,1]
            dz = x_target[i,2]-x_source[j,2]
            r = np.sqrt(dx*dx + dy*dy + dz*dz)
#            if r>0: rinv = 1/r
            if r < h[j]:
                potential[i] += m[j] * PotentialKernel(r, h[j])
            else:
                if r>0: potential[i] -= m[j]/r
    return G*potential

#@jit
#def TotalEnergy(xc, mc, vc, hc, uc):
#    phic = Potential(xc, mc, hc)
#    v_well = vc - np.average(vc, weights=mc,axis=0)
#    vSqr = np.sum(v_well**2,axis=1)
#    return np.sum(mc*(0.5*vSqr + 0.5*phic + uc))

#@jit
def PotentialEnergy(xc, mc, vc, hc, uc, tree=None, particles_not_in_tree=None, x=None, m=None, h=None):
#    if len(xc) > 1e5: return 0 # this effective sets a cutoff in particle number so we don't waste time on things that are clearly too big to be a GMC
    if len(xc)==1: return -2.8*mc/hc**2 / 2
    if tree:
        phic = pykdgrav.Potential(xc, mc, hc, tree=tree, G=4.301e4)
        if particles_not_in_tree: # have to add the potential from the stuff that's not in the tree
            phic += BruteForcePotential2(xc, x[particles_not_in_tree], m[particles_not_in_tree], h=h[particles_not_in_tree], G=4.301e4)
    else:
        phic = BruteForcePotential(xc, mc, hc, G=4.301e4)
    return np.sum(mc*0.5*phic)

##@profile
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


def KineticEnergy(xc, mc, vc, hc, uc):
#    phic = Potential(xc, mc, hc)
    v_well = vc - np.average(vc, weights=mc,axis=0)
    vSqr = np.sum(v_well**2,axis=1)
    return np.sum(mc*(0.5*vSqr + uc))


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
    return(np.abs(2*ke/pe))

def EnergyIncrement(i, c, m, M, x, v, u, h, v_com, tree=None, particles_not_in_tree = None):
    phi = 0.
    if particles_not_in_tree:
        xa, ma, ha = np.take(x,particles_not_in_tree,axis=0), np.take(m,particles_not_in_tree,axis=0), np.take(h,particles_not_in_tree,axis=0)
        phi += BruteForcePotential2(np.array([x[i],]), xa, ma, h=ha, G=4.301e4)[0]
    if tree:
        phi += 4.301e4 * pykdgrav.PotentialWalk(x[i], tree,0., theta=0.7)
    vSqr = np.sum((v[i]-v_com)**2)
    mu = m[i]*M/(m[i]+M)
    return 0.5*mu*vSqr + m[i]*u[i] + m[i]*phi

def KE_Increment(i,m,  v, u, v_com, mtot):
    vSqr = np.sum((v[i]-v_com)**2)
    mu = m[i]*mtot/(m[i]+mtot)
    return 0.5*mu*vSqr + m[i]*u[i]

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


def ParticleGroups(x, m, rho, phi, h, u, v, zz, ids, cluster_ngb=32, rmax=1e100):
    if not potential_mode: phi = -rho
    ngbdist, ngb = cKDTree(x).query(x,min(cluster_ngb, len(x)), distance_upper_bound=min(rmax, h.max()))

    max_group_size = 0
    groups = {}
    particles_since_last_tree = {}
    group_tree = {}
    group_alpha_history = {}
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
    
    assigned_bound_group = -np.ones(len(x),dtype=np.int32)
    largest_assigned_group = -np.ones(len(x),dtype=np.int32)
    
    for i in range(len(x)): # do it one particle at a time, in decreasing order of density
        if not i%10000:
            print("Processed %d of %g particles; ~%3.2g%% done."%(i, len(x), 100*(float(i)/len(x))**2))
        if np.any(ngb[i] > len(x) -1):
            groups[i] = [i,]
            group_tree[i] = None
            assigned_group[i] = i
            group_energy[i] = m[i]*u[i]
            group_KE[i] = m[i]*u[i]
            v_COM[i] = v[i]
            COM[i] = x[i]
            masses[i] = m[i]
            particles_since_last_tree[i] = [i,]
            continue 
        ngbi = ngb[i][1:]

        lower = phi[ngbi] < phi[i]
        if lower.sum():
            ngb_lower, ngbdist_lower = ngbi[lower], ngbdist[i][1:][lower]
            ngb_lower = ngb_lower[ngbdist_lower.argsort()]
            nlower = len(ngb_lower)
        else:
            nlower = 0

        add_to_existing_group = False
        if nlower == 0: # if this is the densest particle in the kernel, let's create our own group with blackjack and hookers
            groups[i] = [i,]
            group_tree[i] = None
            assigned_group[i] = i
            group_energy[i] = m[i]*u[i]# - 2.8*m[i]**2/h[i] / 2 # kinetic + potential energy
            group_KE[i] = m[i]*u[i]
            v_COM[i] = v[i]
            COM[i] = x[i]
            masses[i] = m[i]
            particles_since_last_tree[i] = [i,]
        elif nlower == 1 or assigned_group[ngb_lower[0]] == assigned_group[ngb_lower[1]]: # if there is only one denser particle, or both of the nearest two denser ones belong to the same group, we belong to that group too
            assigned_group[i] = assigned_group[ngb_lower[0]]
            groups[assigned_group[i]].append(i)
            add_to_existing_group = True
        else: # o fuck we're at a saddle point, let's consider both respective groups
            a, b = ngb_lower[:2]
            group_index_a, group_index_b = assigned_group[a], assigned_group[b]
            if masses[group_index_a] < masses[group_index_b]: group_index_a, group_index_b = group_index_b, group_index_a # make sure group a is the bigger one, switching labels if needed

            if group_index_a == group_index_b:  # if both dense boyes belong to the same group, that's the group for us too
                assigned_group[i] = group_index_a
            else:
            #OK, we're at a saddle point, so we need to merge those groups
                group_a, group_b = groups[group_index_a], groups[group_index_b] 
                ma, mb = masses[group_index_a], masses[group_index_b]
                xa, xb = COM[group_index_a], COM[group_index_b] 
                va, vb = v_COM[group_index_a], v_COM[group_index_b]
                group_ab = group_a + group_b
                groups[group_index_a] = group_ab
                
                group_energy[group_index_a] += group_energy[group_index_b]
                group_KE[group_index_a] += group_KE[group_index_b]
                group_energy[group_index_a] += 0.5*ma*mb/(ma+mb) * np.sum((va-vb)**2) # energy due to relative motion: 1/2 * mu * dv^2
                group_KE[group_index_a] += 0.5*ma*mb/(ma+mb) * np.sum((va-vb)**2)
                group_energy[group_index_a] += InteractionEnergy(x,m,h, group_a, group_tree[group_index_a], particles_since_last_tree[group_index_a], group_b, group_tree[group_index_b], particles_since_last_tree[group_index_b]) # mutual interaction energy; we've already counted their individual binding energies

     #           group_energy[group_index_a] += PE(group_ab, x, m, h, v, u)

                if len(group_a) > ntree: # we've got a big group, so we should probably do stuff with the tree
                    if len(group_b) > 512: # if the smaller of the two is also large, let's build a whole new tree, and a whole new adventure
                        group_tree[group_index_a] = pykdgrav.ConstructKDTree(np.take(x,group_ab,axis=0), np.take(m,group_ab), np.take(h,group_ab))
                        particles_since_last_tree[group_index_a][:] = []
                    else:  # otherwise we want to keep the old tree from group a, and just add group b to the list of particles_since_last_tree
                        particles_since_last_tree[group_index_a] += group_b
                else:
                    particles_since_last_tree[group_index_a][:] = group_ab[:]
                    
                if len(particles_since_last_tree[group_index_a]) > ntree:
                    group_tree[group_index_a] = pykdgrav.ConstructKDTree(np.take(x,group_ab,axis=0), np.take(m,group_ab), np.take(h,group_ab))
                    particles_since_last_tree[group_index_a][:] = []                    
                
                COM[group_index_a] = (ma*xa + mb*xb)/(ma+mb)
                v_COM[group_index_a] = (ma*va + mb*vb)/(ma+mb)
                masses[group_index_a] = ma + mb
                groups.pop(group_index_b,None)
                assigned_group[i] = group_index_a
                assigned_group[assigned_group==group_index_b] = group_index_a
                # if this new group is bound, we can delete the old bound group
                avir = abs(2*group_KE[group_index_a]/np.abs(group_energy[group_index_a] - group_KE[group_index_a]))
                if avir < alpha_crit:
                    largest_assigned_group[group_ab] = len(group_ab)
                    assigned_bound_group[group_ab] = group_index_a

                for d in groups, particles_since_last_tree, group_tree, group_energy, group_KE, COM, v_COM, masses: # delete the data from the absorbed group
                    d.pop(group_index_b, None)
                add_to_existing_group = True
                
            groups[group_index_a].append(i)
            max_group_size = max(max_group_size, len(groups[group_index_a]))
            
        if add_to_existing_group: # assuming we've added a particle to an existing group, we have to update stuff
            g = assigned_group[i]
            mgroup = masses[g]
            group_KE[g] += KE_Increment(i, m, v, u, v_COM[g], mgroup)
            group_energy[g] += EnergyIncrement(i, groups[g][:-1], m, mgroup, x, v, u, h, v_COM[g], group_tree[g], particles_since_last_tree[g])
            avir = abs(2*group_KE[g]/np.abs(group_energy[g] - group_KE[g]))
            if avir < alpha_crit:
                largest_assigned_group[i] = len(groups[g])
#                assigned_bound_group[i] = g  NOTE: need to assign ALL group members to this group upon adding a particle - see below
                assigned_bound_group[groups[g]] = g
#            group_alpha_history[g].append([rho[i], avir])
            v_COM[g] = (m[i]*v[i] + mgroup*v_COM[g])/(m[i]+mgroup)
            masses[g] += m[i]
            particles_since_last_tree[g].append(i)
            if len(particles_since_last_tree[g]) > ntree:
                group_tree[g] = pykdgrav.ConstructKDTree(x[groups[g]], m[groups[g]], h[groups[g]])
                particles_since_last_tree[g][:] = []
            max_group_size = max(max_group_size, len(groups[g]))

    # Now assign particles to their respective bound groups
    print((assigned_bound_group == -1).sum() / len(assigned_bound_group))
    for i in range(len(assigned_bound_group)):
        a = assigned_bound_group[i]
        if a < 0:
#            if sfr[i] > 0: print(a)
            continue

        if a in bound_groups.keys(): bound_groups[a].append(i)
        else: bound_groups[a] = [i,]

    return groups, bound_groups, assigned_group


def ComputeClouds(filepath):
    outputfolder = options["--outputfolder"]
    if ".hdf5" in filepath: # we have a lone snapshot, no snapdir
        snapnum = int(filepath.split("_")[-1].split(".hdf5")[0].split(".")[0].replace("/",""))
        snapname = filepath.split("_")[-2].split("/")[-1]
        snapdir = "/".join(filepath.split("/")[:-1])
        if outputfolder == "None": outputfolder = snapdir #getcwd() + "/".join(filepath.split("/")[:-1])
    else: # filepath refers to the directory in which the snapshot's multiple files are stored
        snapnum = int(filepath.split("snapdir_")[-1].replace("/",""))
        print(filepath)
        snapname = glob(filepath+"/*.hdf5")[0].split("_")[-2].split("/")[-1] #"snapshot" #filepath.split("_")[-2].split("/")[-1]
        print(snapname)
        snapdir = filepath.split("snapdir")[0] + "snapdir" + filepath.split("snapdir")[1]
        if outputfolder == "None": outputfolder = getcwd() + filepath.split(snapdir)[0]

    if outputfolder == "": outputfolder = "."
    if outputfolder is not "None":
        if not path.isdir(outputfolder):
            mkdir(outputfolder)

    hdf5_outfilename = outputfolder + '/'+ "Clouds_%s_n%g_alpha%g.hdf5"%(snapnum, nmin, alpha_crit)
    dat_outfilename = outputfolder + '/' +"bound_%s_n%g_alpha%g.dat"%(snapnum, nmin,alpha_crit)
    if path.isfile(dat_outfilename) and not options["--overwrite"]: return
            
    if not snapdir:
        snapdir = getcwd()
        print('Snapshot directory not specified, using local directory of ', snapdir)

    
    fname_found, _, _ =load_from_snapshot.check_if_filename_exists(snapdir,snapnum, snapshot_name=snapname)
    if fname_found!='NULL':    
        print('Snapshot ', snapnum, ' found in ', snapdir)
    else:
        print('Snapshot ', snapnum, ' NOT found in ', snapdir, '\n Skipping it...')
        return

    cluster_ngb = int(float(options["--cluster_ngb"]) + 0.5)
    G = float(options["--G"])
    boxsize = options["--boxsize"]
    ptype = int(options["--ptype"])


    softening = float(options["--softening"])
    if boxsize != "None":
        boxsize = float(boxsize)
    else:
        boxsize = None
    fuzz = float(options["--fuzz"])
    npart = load_from_snapshot.load_from_snapshot("NumPart_Total", "Header", snapdir, snapnum, snapshot_name=snapname)[ptype]
    print(npart)
    if npart < cluster_ngb:
        print("Not enough particles for meaningful cluster analysis!")
        return
    
    #Read gas properties
    keys = load_from_snapshot.load_from_snapshot("keys",ptype,snapdir,snapnum, snapshot_name=snapname)
    if keys is 0:
        print("No keys found, noping out!")        
        return

    criteria = np.ones(npart,dtype=np.bool) # now we refine by particle density
    if "Density" in keys:
        rho = load_from_snapshot.load_from_snapshot("Density",ptype,snapdir,snapnum, snapshot_name=snapname, units_to_physical=(not options["--units_already_physical"]))
        if len(rho) < cluster_ngb:
            print("Not enough particles for meaningful cluster analysis!")
            return

    else: # we have to do a kernel density estimate for e.g. dark matter or star particles
        m = load_from_snapshot.load_from_snapshot("Masses",ptype,snapdir,snapnum, snapshot_name=snapname)
        if len(m) < cluster_ngb:
            print("Not enough particles for meaningful cluster analysis!")
            return
        x = load_from_snapshot.load_from_snapshot("Coordinates",ptype,snapdir,snapnum, snapshot_name=snapname)
        print("Computing density...")
        rho = Meshoid(x,m,des_ngb=cluster_ngb).Density()
        print("Density done!")
        criteria = np.arange(len(rho))[(rho*404 > nmin)] # only look at dense gas (>nmin cm^-3)
        
    criteria = np.arange(len(rho))[rho*404 > nmin] # only look at dense gas (>nmin cm^-3)
    print("%g particles denser than %g cm^-3" % (criteria.size,nmin))  #(np.sum(rho*147.7>nmin), nmin))
    if not criteria.size > cluster_ngb:
        print('Not enough dense particles, exiting...')
        return
    rho = np.take(rho, criteria, axis=0)
    rho_order = (-rho).argsort()
    rho = rho[rho_order]
    particle_data = {"Density": rho} # now let's store all particle data that satisfies the criteria
    for k in keys:
        if not k in particle_data.keys():
            particle_data[k] = load_from_snapshot.load_from_snapshot(k,ptype,snapdir,snapnum, snapshot_name=snapname, particle_mask=criteria, units_to_physical=(not options["--units_already_physical"]))[rho_order]
    m = particle_data["Masses"]
    x = particle_data["Coordinates"]
    ids = particle_data["ParticleIDs"] #load_from_snapshot.load_from_snapshot("ParticleIDs",ptype,snapdir,snapnum, particle_mask=criteria)
    u = (particle_data["InternalEnergy"] if ptype == 0 else np.zeros_like(m))
    if "MagneticField" in keys:
        energy_density_code_units = np.sum(particle_data["MagneticField"]**2,axis=1) / 8 / np.pi * 5.879e9
        specific_energy = energy_density_code_units / rho
        u += specific_energy
        
    zz = (particle_data["Metallicity"] if "Metallicity" in keys else np.zeros_like(m))
    v = particle_data["Velocities"]
    if "StarFormationRate" in keys: sfr = particle_data["StarFormationRate"]
    else: sfr = np.zeros_like(m)

    if "AGS-Softening" in keys:
        hsml = particle_data["AGS-Softening"]
    elif "SmoothingLength" in keys:
        hsml = particle_data["SmoothingLength"] 
    else:
        hsml = np.ones_like(m)*softening
        
    if "Potential" in keys: # potential doesn't get used anymore, so this is moot
        phi = particle_data["Potential"] #load_from_snapshot.load_from_snapshot("Potential",ptype,snapdir,snapnum, particle_mask=criteria)
    else:
        phi = np.zeros_like(m)
    
    x, m, rho, phi, hsml, u, v, zz = np.float64(x), np.float64(m), np.float64(rho), np.float64(phi), np.float64(hsml), np.float64(u), np.float64(v), np.float64(zz)
    while len(np.unique(x,axis=0)) < len(x): # make sure no two particles are at the same position
        x *= 1+ np.random.normal(size=x.shape) * 1e-8

    t = time()
    groups, bound_groups, assigned_group = ParticleGroups(x, m, rho, phi, hsml, u, v, zz, ids, cluster_ngb=cluster_ngb, rmax=float(options["--max_linking_length"]))
    t = time() - t
    print("Time: %g"%t)
    print("Done grouping. Computing group properties...")
    groupmass = np.array([m[c].sum() for c in bound_groups.values() if len(c)>3])
    groupsfr = np.array([sfr[c].sum() for c in bound_groups.values() if len(c)>3])
    print("Total SFR in clouds: ",  groupsfr.sum())
    groupid = np.array([c for c in bound_groups.keys() if len(bound_groups[c])>3])
    groupid = groupid[groupmass.argsort()[::-1]]
    bound_groups = OrderedDict(zip(groupid, [bound_groups[i] for i in groupid]))

    # Now we analyze the clouds and dump their properties

    bound_data = OrderedDict()
    bound_data["Mass"] = []
    bound_data["Center"] = []
    bound_data["PrincipalAxes"] = []
    bound_data["Reff"] = []
    bound_data["HalfMassRadius"] = []
    bound_data["NumParticles"] = []
    bound_data["VirialParameter"] = []
    
    
    print(hdf5_outfilename)
    Fout = h5py.File(hdf5_outfilename, 'w')

    i = 0
    fids = ids
    #Store all keys in memory to reduce I/O load
#    print '\t Reading all data for Particle Type ', ptype

#    print '\t Reading done, iterating over clouds...'
    for k,c in bound_groups.items():
 #       print(len(c), len(np.unique(c)))
        bound_data["Mass"].append(m[c].sum())
        bound_data["NumParticles"].append(len(c))
        bound_data["Center"].append(np.average(x[c], weights=m[c], axis=0))
        dx = x[c] - bound_data["Center"][-1]
        eig = np.linalg.eig(np.cov(dx.T))[0]
        bound_data["PrincipalAxes"].append(np.sqrt(eig))
        r = np.sum(dx**2, axis=1)**0.5
        bound_data["HalfMassRadius"].append(np.median(r))
        bound_data["Reff"].append(np.sqrt(5./3 * np.average(r**2,weights=m[c])))
        bound_data["VirialParameter"].append(VirialParameter(c, x, m, hsml, v, u))

        cluster_id = "Cloud"+ ("%d"%i).zfill(int(np.log10(len(bound_groups))+1))

        N = len(c)

        Fout.create_group(cluster_id)
        for k in keys: #range(len(keys)):
            Fout[cluster_id].create_dataset('PartType'+str(ptype)+"/"+k, data = particle_data[k].take(c,axis=0))
        i += 1

    print("Done grouping bound clusters!")

       
    Fout.close()
    
    SaveArrayDict(dat_outfilename, bound_data)

def func(path):
    """Necessary for the multiprocessing pickling to work"""
    return ComputeClouds(path, docopt(__doc__))

def main(input):
    global options; options=input
    global nmin; nmin = float(options["--nmin"])
    global alpha_crit; alpha_crit = float(options["--alpha_crit"])
    global overwrite; overwrite =  options["--overwrite"]
    global ntree; ntree = int(options["--ntree"])
    global nproc; nproc=int(options["--np"])

    snappaths = [p  for p in options["<snapshots>"]] 
    if nproc==1:
        for f in snappaths:
            print(f)
            ComputeClouds(f)
    else:
        Pool(nproc).map(func, snappaths,chunksize=1)

def make_input(snapshots="snapshot_000.hdf5", outputfolder='None',ptype=0, G=4.301e4, boxsize='None', cluster_ngb=32,nmin=1,softening=1e-5, fuzz=0, alpha_crit=2,np=1,ntree=10000, overwrite=False, units_already_physical=False,max_linking_length=1e100):
    if (not isinstance(snapshots, list)):
        snapshots=[snapshots]
    arguments={
        "<snapshots>": snapshots,
        "--outputfolder": outputfolder,
        "--ptype": ptype,
        "--G": G,
        "--boxsize": boxsize,
        "--cluster_ngb": cluster_ngb,
        "--nmin": nmin,
        "--softening": softening,
        "--fuzz": fuzz,
        "--alpha_crit": alpha_crit,
        "--np": np,
        "--ntree": ntree,
        "--overwrite": overwrite,
        "--units_already_physical": units_already_physical,
        "--max_linking_length": max_linking_length
        }
    return arguments

if __name__ == "__main__": 
    options = docopt(__doc__)
    main(options)
    
