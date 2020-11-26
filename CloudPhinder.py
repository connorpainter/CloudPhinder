#!/usr/bin/env python
"""                                                                            
Algorithm that identifies the largest possible self-gravitating structures of a certain particle type. This, newer version relies on the load_from_snapshot routine from GIZMO.

Usage: CloudPhinder.py <snapshots> ... [options]

Options:                                                                       
   -h --help                  Show this screen.

   --outputfolder=<name>      Specifies the folder to save the outputs to, None defaults to the same location as the snapshot [default: None]
   --ptype=<N>                GIZMO particle type to analyze [default: 0]
   --G=<G>                    Gravitational constant to use; should be consistent with what was used in the simulation. [default: 4.301e4]
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

## from builtin
from time import time,sleep
from os import path

import numpy as np

from docopt import docopt

from multiprocessing import Pool
import itertools

## from here
from .io_tools import parse_filepath, make_input, read_particle_data, parse_particle_data, computeAndDump, SaveArrayDict
from .clump_tools import ComputeGroups

def CloudPhind(filepath,options,particle_data=None):
    ## parses filepath and reformats outputfolder if necessary
    snapnum, snapdir, snapname, outputfolder = parse_filepath(filepath,options["--outputfolder"])

    ## skip if the file was not parseable
    if snapnum is None: return

    ## generate output filenames
    nmin = float(options["--nmin"])
    alpha_crit = float(options["--alpha_crit"])

    hdf5_outfilename = outputfolder + '/'+ "Clouds_%s_n%g_alpha%g.hdf5"%(snapnum, nmin, alpha_crit)
    dat_outfilename = outputfolder + '/' +"bound_%s_n%g_alpha%g.dat"%(snapnum, nmin,alpha_crit)    

    ## check if output already exists, if we aren't being asked to overwrite, short circuit
    overwrite = options["--overwrite"]
    if path.isfile(dat_outfilename) and not overwrite: return 

    ## read particle data from disk and apply dense gas cut
    ##  also unpacks relevant variables
    ptype = int(options["--ptype"])
    cluster_ngb = int(float(options["--cluster_ngb"]) + 0.5)

    if particle_data is None:
        particle_data = read_particle_data(
            snapnum,
            snapdir,
            snapname,
            ptype=ptype,
            softening=float(options["--softening"]),
            units_already_physical=bool(options["--units_already_physical"]),
            cluster_ngb=cluster_ngb)

    (x,m,rho,
    phi,hsml,u,
    v,zz,sfr) = parse_particle_data(particle_data)

    ## skip this snapshot, there probably weren't enough particles
    if x is None: return

    ## call the cloud finder itself
    groups, bound_groups, assigned_groups = ComputeGroups(
        x,m,rho,
        phi,hsml,u,
        v,zz,sfr,
        cluster_ngb=cluster_ngb,
        max_linking_length=float(options["--max_linking_length"]),
        nmin=nmin,
        ntree = int(options["--ntree"]),
        alpha_crit=alpha_crit,
        )

    ## compute some basic properties of the clouds and dump them and
    ##  the particle data to disk
    computeAndDump(
        particle_data,
        ptype,
        bound_groups,
        assigned_groups,
        hdf5_outfilename,
        dat_outfilename,
        overwrite)

def main(options):

    nproc=int(options["--np"])

    snappaths = [p  for p in options["<snapshots>"]] 
    if nproc==1:
        for f in snappaths:
            print(f)
            CloudPhind(f,options)
    else:
        argss = zip(snappaths,itertools.repeat(options)) 
        with Pool(nproc) as my_pool:
            my_pool.starmap(CloudPhind, argss, chunksize=1)

if __name__ == "__main__": 
    options = docopt(__doc__)
    main(options)
    
