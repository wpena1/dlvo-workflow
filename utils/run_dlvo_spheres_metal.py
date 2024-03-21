"""
Code written by Gaurav Mitra and team members
used here with permission
"""
import time
import hoomd
import hoomd.md
import numpy as np
import os
import sys
import re
import os.path
import argparse
import random
from copy import copy
import gsd.hoomd

def sample_spherical(npoints, ndim=3):
#simple points on a sphere from here https://stackoverflow.com/questions/33976911/generate-a-random-sample-of-points-distributed-on-the-surface-of-a-unit-sphere
    vec = np.random.randn(ndim, npoints)
    vec /= np.linalg.norm(vec, axis=0)
    return vec.reshape((-1,3))

def sphere_fibonacci_grid_points (ng,r):
# Output, real xgB(3,ng): the grid points.
  phi = ( 1.0 + np.sqrt ( 5.0 ) ) / 2.0
  theta = np.zeros ( ng )
  sphi = np.zeros ( ng )
  cphi = np.zeros ( ng )
  for i in range ( 0, ng ):
    i2 = 2 * i - ( ng - 1 )
    theta[i] = 2.0 * np.pi * float ( i2 ) / phi
    sphi[i] = float ( i2 ) / float ( ng )
    cphi[i] = np.sqrt ( float ( ng + i2 ) * float ( ng - i2 ) ) / float ( ng )
  xg = np.zeros ( ( ng, 3 ) )
  for i in range ( 0, ng ) :
    xg[i,0] = cphi[i] * np.sin ( theta[i] ) * r
    xg[i,1] = cphi[i] * np.cos ( theta[i] ) * r
    xg[i,2] = sphi[i] * r
  return xg

def read_xyz_file(seed_file, particle_type_list):
    fh = open(seed_file,'r')
    line = fh.readline()
    config = []
    type_dict = {}
    type_list = []
    n_atoms = int(line)
    #skip a blank line
    line = fh.readline()
    line = fh.readline()
    while line:
        type, x, y, z = line.split()
        #if not type in type_dict:
        #    type_dict[type] = len(type_dict)
        #type_list.append(type_dict[type])
        type_list.append(particle_type_list.index(type))
        config.append(np.array((x,y,z)).astype(float))
        line = fh.readline()
    config = np.array(config)
    assert len(config) == n_atoms, "xyz reader- Number of atoms doesn't match the number of lines read in"
    return config, type_list


##DEFINING THE DLVO POTENTIAL
def screened_potential_shifted_force(r,rmin,rmax,radius_sum,steric_prefactor,electrostatic_prefactor,H,d):

    #d=debye length,H=2*brush_length

    #a and b are set so that the force and energy goes to zero at rmax
    #make assumption that rmax > H

    #separate distances for electrostatic and hard sphere

    l = r - radius_sum
    lcut = rmax - radius_sum

    a = electrostatic_prefactor/d*np.exp(-lcut/d)
    b = -electrostatic_prefactor*np.exp(-lcut/d) - lcut*a

    p_term1 = electrostatic_prefactor*np.exp(-l/d) + a*l + b
    p_term2 = steric_prefactor*(28*((H/l)**.25-1) + (20./11)*(1-(l/H)**2.75)+ 12*(l/H-1))

    if type(p_term2) == np.ndarray: p_term2[np.isnan(p_term2)]=0

    potential = p_term1 + p_term2*(l<H)

    # F = -dU/dr
    f_term1 = electrostatic_prefactor/d*np.exp(-l/d) - a
    f_term2 = -steric_prefactor/H*(12-7*(H/l)**(1.25)-5*(l/H)**(1.75))
    if type(f_term2) == np.ndarray: f_term2[np.isnan(f_term2)]=0
    force = f_term1 + (f_term2)*(l<H)

    return potential, force

if __name__=="__main__":
    import yaml
    '''
    parser=argparse.ArgumentParser()
    #parser.add_argument("-i","--inputfile",default=None,help="Input file (gsd file, optional)",type=str)
    parser.add_argument("--gpu",default=False,action="store_true")
    parser.add_argument("--closedwalls",default=False,help="Turn on closed walls along x/y/z directions (default: False)",action="store_true")
    parser.add_argument("--gravity",default=0.5,help="Turn on downward gravity constant force in z by setting to a positive number (default: 0)",type=float)
    parser.add_argument("--chargewall",default=0,help="Turn on wall perp to z attracting -(>0) or +(<0) particles (default: False)",type=float)
    parser.add_argument("-RN","--radiusN",default=250.0,help="Radius of colloids (default: %(default)s)",type=float)
    parser.add_argument("-RP","--radiusP",default=250.0,help="Radius of colloids (default: %(default)s)",type=float)
    parser.add_argument("-B","--brush_length",help="Brush length (default: %(default)s)",default=10,type=float)
    parser.add_argument("-s","--brush_density",help="Brush density (default: %(default)s)",default=0.09,type=float)
    parser.add_argument("-d","--debye_length",help="Debye length (default: %(default)s)",default=4,type=float)
    parser.add_argument("--surface_potentialN",help="Surface potential N in mV (default: %(default)s)",default=-50,type=float)
    parser.add_argument("--surface_potentialP",help="Surface potential P in mV (default: %(default)s)",default=50,type=float)
    parser.add_argument("--dielectric_constant",help="Dielectric constant (default: %(default)s)",default=80,type=float)
    parser.add_argument("--gamma",help="Drag coefficient (default: %(default)s)",default=0.01,type=float)
    parser.add_argument("--massP",help="Mass of a positive particle",default=1.0,type=float)
    parser.add_argument("--massN",help="Mass of a negative particle",default=None,type=float)
    parser.add_argument("--seed",help="Random seed (default: %(default)s)",default=1,type=int)
    parser.add_argument("--fraction_positive",help="Fraction positive charge (default: %(default)s)",default=0.1,type=float)
    parser.add_argument("-a","--lattice_spacing",help="Lattice spacing in terms of positive particle diameter (default: %(default)s)",default=1.5,type=float)
    parser.add_argument("--lattice_repeats",default=5,type=int,help="times to repliacte the system in each direction (default: %(default)s)") #no of times we want to replicate in one direction
    parser.add_argument("--lattice_type",default="bcc",help="Lattice type (bcc, sc, fcc) (default: %(default)s)")
    parser.add_argument("--orbit_factor",default=1.3,type=float,help="Factor beyond sum of radii and brush at which to start N particles (default: %(default)s)")
    parser.add_argument("--dt",help="Simulation time step (default: %(default)s)",type=float,default=0.1)
    parser.add_argument("--seed_file",help="xyz file specifying some seed coordinates",default=None)
    parser.add_argument("--scale_by",default="max_diameter",help="Scale xyz by this (default: %(default)s), options: max_diameter, max_radius")
    #parser.add_argument("-o","--outputprefix",help="Output prefix (required)",type=str,required=True)
    #parser.add_argument("-n","--nsteps",help="Number of steps to run",type=int,required=True)
    #parser.add_argument("-T","--temperature",help="Temperature at which to run, or list of tuples specifying annealing schedule",default="1.0")
    #parser.add_argument("-m","--mode",help="Integrator/dynamics scheme. Allowed: Minimize, Langevin, NVT (default: %(default)s)",default="Langevin")
    args = parser.parse_args()
    #locals().update(vars(args))
    '''

    print("Remote: Loading the yaml input file")
    # running process must be in run directory
    with open ('inputs_new.yaml') as f:
            parameters = yaml.load(f,Loader=yaml.FullLoader)
    locals().update(parameters)

    print("Remote: ======= ALL PARAMETERS =======")
    print(parameters)

    # DECIDE WHETHER RUNNING ON GPU OR ON CPU
    if gpu=='True':
        hoomd.context.initialize("--mode=gpu")
        print("Running on the GPU")
    else:
        hoomd.context.initialize("--mode=cpu")
        print("Running on the CPU")

    np.random.seed(parameters['seed'])
    expected_num_frames = 1000
    dump_frequency=int(nsteps/expected_num_frames)

    ##DEFINE ALL THE RELEVANT QUANTITIES

    mV_to_kBT = 25.7
    joule_to_kBT = 4.11e-21

    #N=negative, P=positive
    #use harmonic average from Derjaguin approximation: https://en.wikipedia.org/wiki/Derjaguin_approximation
    ionic_radius = 2./(1./radiusN + 1./radiusP)
    repulsion_radius = (radiusN+radiusP)/2.
    max_radius = np.max(( radiusN, radiusP))
    radius_list = [radiusP, radiusN, ionic_radius]
    print("Particle radii and radii (Derjaguin, avg)",radius_list)

    brush2 = brush_length*2
    steric_prefactor = 16*repulsion_radius*(brush2**2)*(brush_density**(3./2))/35.
    print("Steric prefactor is: %f"%steric_prefactor)

    permitivity = 8.85e-12 #Farad/M
    ionic_radius_in_m = 1e-9*ionic_radius
    radiusN_in_m = 1e-9*radiusN
    radiusP_in_m = 1e-9*radiusP
    surface_potentialN_in_V = surface_potentialN/1000
    surface_potentialP_in_V = surface_potentialP/1000
    electrostatic_prefactors = [ 2*np.pi*dielectric_constant*permitivity*radiusP_in_m/joule_to_kBT*surface_potentialP_in_V*surface_potentialP_in_V, 2*np.pi*dielectric_constant*permitivity*radiusN_in_m/joule_to_kBT*surface_potentialN_in_V*surface_potentialN_in_V, 2*np.pi*dielectric_constant*permitivity*ionic_radius_in_m/joule_to_kBT*surface_potentialN_in_V*surface_potentialP_in_V ]
    print("Electrostatic prefactors are: ",electrostatic_prefactors)


    ##SET UP THE SIMULATION TO BE RESTARTABLE IF A PROGRESSFILE EXISTS

    progressfile = outputprefix+'.progress.txt'
    if os.path.exists(progressfile):
        continuesim=True
        prevsteps = int(open(progressfile,'r').readlines()[0])
        inputfile = outputprefix+'.run.%i.gsd'%prevsteps
        totalsteps = prevsteps + nsteps
    else:
        continuesim = False
        prevsteps = 0
        totalsteps = prevsteps + nsteps

    outputgsdfile = outputprefix+'.run.%i.gsd'%totalsteps
    outputlogfile = outputprefix+'.run.%i.log'%totalsteps

    ##SNAPSHOT CREATION

    #inputfile=args.inputfile
    if(continuesim==True):
        system=hoomd.init.read_gsd(inputfile,frame=-1)  #read from last frame of previous gsd file (if present)
        snapshot = system.take_snapshot()
        particle_type_list = system.particles.types
        print("Particle types:",particle_type_list)
        #half of the box lengths in x, y, z directions
        max_z = snapshot.box.Lz/2.
        max_x = snapshot.box.Lx/2.
        max_y = snapshot.box.Ly/2.
        print("Box size:")
        print(system.box)
        mod_maxX = system.box.Lx/2.
        mod_maxY = system.box.Ly/2.
        mod_maxZ = system.box.Lz/2.
    else:
        print("Lattice_type: ",lattice_type)
        if lattice_type == "fcc":
            lattice = hoomd.lattice.fcc
        elif lattice_type == "bcc":
            lattice = hoomd.lattice.bcc
        elif lattice_type == "sc":
            lattice = hoomd.lattice.sc
        else:
            print("Lattice type %s is not supported (only fcc, bcc, sc)"%lattice_type)
            sys.exit(1)
        if(radiusN>radiusP):
            latticesite='N'
            satellite='P'
            radius_latticesite=radiusN
            radius_satellite=radiusP
        else:
            latticesite='P'
            satellite='N'
            radius_latticesite=radiusP
            radius_satellite=radiusN

        print("Lattice site is:",latticesite)
        orbit_distance = (radiusP+radiusN+brush2)*orbit_factor
        print("Putting satellite particles at a distance of %f from center particles"%orbit_distance)
        num_central = lattice_repeats**3
        num_satellite_per_center = int((1-fraction_positive)/fraction_positive)
        print("num_satellite_per_center:",num_satellite_per_center)
        center_unit_cell = lattice(a=lattice_spacing*2*radius_latticesite)
        center_in_uc = center_unit_cell.N
        satellite_in_uc = center_in_uc*num_satellite_per_center
        print("center_in_uc:",center_in_uc)
        print("satellite_in_uc:",satellite_in_uc)
        total_in_uc = satellite_in_uc + center_in_uc
        print("Total number of particles in each unit cell:",total_in_uc)
        print("Total lattice spacing:",lattice_spacing*2*radius_latticesite)

        #scale mass if not set
        if massN is None:
            massN = massP/(radiusP/radiusN)**3
            #massN=massP
            print("Setting mass N: %f (mass P: %f)"%(massN,massP))

        mass_dict = {}
        mass_dict['N'] = massN
        mass_dict['P'] = massP

        particle_positions = center_unit_cell.position

        all_positions = []
        all_masses = []
        all_types = []
        all_diameters = []

        for i in range(center_in_uc):
            xyz = particle_positions[i]
            all_positions.append(xyz)
            all_masses.append(mass_dict[latticesite])
            all_types.append(latticesite)
            all_diameters.append(2*radius_latticesite)

            #sattelite_positions = sample_spherical(num_negative_per_center)*orbit_distance
            sattelite_positions = sphere_fibonacci_grid_points (num_satellite_per_center,orbit_distance)
            for xyz_s in sattelite_positions:
                all_positions.append(xyz_s+xyz)
                all_masses.append(massN)
                all_types.append(satellite)
                all_diameters.append(2*radius_satellite)

        uc = hoomd.lattice.unitcell(N = total_in_uc,
                        a1 = center_unit_cell.a1,
                        a2 = center_unit_cell.a2,
                        a3 = center_unit_cell.a3,
                        dimensions = center_unit_cell.dimensions,
                        position = all_positions,
                        type_name = all_types,
                        mass = all_masses,
                        diameter = np.array(all_diameters),)

        snapshot = uc.get_snapshot()
        snapshot.replicate(lattice_repeats,lattice_repeats,lattice_repeats)
        snapshot.particles.types=[latticesite,satellite]
        N=len(snapshot.particles.typeid)
        print("Total number of lattice sites:",N)
        particle_type_list=snapshot.particles.types

        #half of the box lengths in x, y, z directions
        max_z = snapshot.box.Lz/2.
        max_x = snapshot.box.Lx/2.
        max_y = snapshot.box.Ly/2.

        system=hoomd.init.read_snapshot(snapshot)

        print("Initial Box size:")
        print(system.box)

        if(closedwalls=='True'):
            hoomd.update.box_resize(Lx = 2.*max_x+8*max_radius, Ly = 2.*max_y+8*max_radius, Lz=2.*max_z+8*max_radius, period=None,scale_particles=False)
            print("Updated box size in presence of closed walls:")
            print(system.box)

        mod_maxX = system.box.Lx/2.
        mod_maxY = system.box.Ly/2.
        mod_maxZ = system.box.Lz/2.

    ##ADDING INTERACTIONS IN THE SYSTEM

    ##Creating neighborlist

    nl = hoomd.md.nlist.cell()
    nl.reset_exclusions(exclusions = ['1-2', 'body','constraint'])

    ##Implementing gravity

    if gravity>0:
        typeN = hoomd.group.type(type='N')
        typeP = hoomd.group.type(type='P')
        fgrav = -gravity
        gravity_force_P = hoomd.md.force.constant(fvec=[0,0,fgrav],group=typeP)
        #gravity_force_N = hoomd.md.force.constant(fvec=[0,0,fgrav*massN],group=N)

    ##Implementing closed walls on the 6 faces of the simulation box to prevent particles going out of the box

    if closedwalls=='True':
        print("Turning on closed walls along x/y/z with effective max_z:",mod_maxZ)

        #create 6 repulsive walls
        upper_wall_x = hoomd.md.wall.plane(origin=(mod_maxX,0,0),normal=(-1,0,0),inside=True)
        lower_wall_x = hoomd.md.wall.plane(origin=(-mod_maxX,0,0),normal=(1,0,0),inside=True)
        upper_wall_y = hoomd.md.wall.plane(origin=(0,mod_maxY,0),normal=(0,-1,0),inside=True)
        lower_wall_y = hoomd.md.wall.plane(origin=(0,-mod_maxY,0),normal=(0,1,0),inside=True)
        upper_wall_z = hoomd.md.wall.plane(origin=(0,0,mod_maxZ),normal=(0,0,-1),inside=True)
        lower_wall_z = hoomd.md.wall.plane(origin=(0,0,-mod_maxZ),normal=(0,0,1),inside=True)

        wall_group = hoomd.md.wall.group(upper_wall_x,lower_wall_x,upper_wall_y,lower_wall_y,upper_wall_z,lower_wall_z)
        wall_force = hoomd.md.wall.slj(wall_group,r_cut=max_radius*(2.0**(1.0/6.0)))
        wall_force.force_coeff.set('N',epsilon=1,sigma=radiusN,alpha=0,r_cut=radiusN*(2.0**(1.0/6.0)))
        wall_force.force_coeff.set('P',epsilon=1,sigma=radiusP,alpha=0,r_cut=radiusP*(2.0**(1.0/6.0)))

    ##Implementing a charged attractive wall at the bottom of the simulation box

    # if chargewall!=0 and not closedwalls:
    if chargewall!=0:
        eps=np.abs(chargewall)
        z_chgwall = -mod_maxZ + 0.5*max_radius
        #upper_wall = hoomd.md.wall.plane(origin=(0,0,0),normal=(0,0,-1),inside=True)
        lower_wall = hoomd.md.wall.plane(origin=(0,0,z_chgwall),normal=(0,0,1),inside=True)
        wall_group = hoomd.md.wall.group(lower_wall)

        if chargewall < 0:
            attractive_wall_force = hoomd.md.wall.lj(wall_group,r_cut=radiusP*5)
            attractive_wall_force.force_coeff.set('N',epsilon=0,sigma=radiusN*2)
            attractive_wall_force.force_coeff.set('P',epsilon=eps,sigma=radiusP*2)

        if chargewall > 0:
            attractive_wall_force = hoomd.md.wall.lj(wall_group,r_cut=radiusN*5)
            attractive_wall_force.force_coeff.set('N',epsilon=eps,sigma=radiusN*2)
            attractive_wall_force.force_coeff.set('P',epsilon=0,sigma=radiusP*2)

    ##Implementing tabulated DLVO potential with polymer brush repulsion

    table = hoomd.md.pair.table(width=5000,nlist=nl)
    minima_list=[]

    pair_type=0

    for i in range(len(particle_type_list)):
        typei = particle_type_list[i]
        for j in range(i,len(particle_type_list)):
            typej = particle_type_list[j]
            key=(typei,typej)
            if i == j:
                my_radius = radius_list[i]
                radius_sum = 2*my_radius
                pair_type = i
            else:
                my_radius = radius_list[-1]
                radius_sum = radius_list[0]+radius_list[1]
                pair_type = 2
            sum_brush_lengths=brush2
            print("*******************************************************************************************************************************************************************************")
            print("Setting potential for pair ("+str(i)+","+str(j)+") with radius_sum "+str(radius_sum))
            potential_range = radius_sum + 20*debye_length
            table.pair_coeff.set(typei,typej, func=screened_potential_shifted_force,rmin=1.00005*radius_sum, rmax=potential_range,coeff=dict(radius_sum=radius_sum, steric_prefactor=steric_prefactor,electrostatic_prefactor=electrostatic_prefactors[pair_type],H=sum_brush_lengths,d=debye_length),)

            pot_min=1.00005*radius_sum
            pot_max = radius_sum+20*debye_length
            dpot = (pot_max-pot_min)/5000.
            test_range = np.arange(pot_min,pot_max+dpot,dpot)
            pot,force = screened_potential_shifted_force(test_range,rmin=pot_min, rmax=pot_max, radius_sum=radius_sum, steric_prefactor=steric_prefactor,electrostatic_prefactor=electrostatic_prefactors[pair_type],H=sum_brush_lengths,d=debye_length)
            np.savetxt(outputprefix+".pairpotential_%i,%i.txt"%(i,j), np.concatenate( (test_range.reshape(-1,1),pot.reshape(-1,1),force.reshape(-1,1)),axis=-1),header="Distance                   Potential                   Force" )
            minima_list.append((i,j,key,np.min(pot),test_range[np.argmin(pot)]))
            pair_type=pair_type+1


    print("Potential energy minima information:")
    print("i,j,key,potential_min,potential_min_location")


    for min_info in minima_list:
        print(min_info)


    ##RUN MD SIMULATION

    if type(temperature) is str and temperature.find(':')>0:
        temp_list = np.array( temperature.split(':') ,dtype=float)
        assert len(temp_list)%2==0 and len(temp_list)>=0,"must give an even number of temperature arguments"
        temp_list = [tuple(x) for x in temp_list.reshape((-1,2))]
        temperature = hoomd.variant.linear_interp(points=temp_list)
    else:
        temperature = float(temperature)
    all = hoomd.group.all()


    hoomd.analyze.log(filename=outputlogfile,
                      quantities=['potential_energy', 'temperature'],
                      period=10000,
                      overwrite=True)

    if mode.lower() == "minimize":
        fire = hoomd.md.integrate.mode_minimize_fire(dt=dt,Etol=1e-7,min_steps=100,group=all)
        hoomd.dump.gsd(outputprefix+'.gsd', period=500, group=all, overwrite=True)
        while not(fire.has_converged()):
            hoomd.run(500)
        sys.exit()
    elif mode.lower() == "langevin":
        print("mode:",mode)
        hoomd.md.integrate.mode_standard(dt=dt)
        ld=hoomd.md.integrate.langevin(group=all, kT=temperature, seed=seed)
        print("Using gamma value as:", gamma)
        for particle_type in particle_type_list:
                ld.set_gamma(particle_type,gamma)
    elif mode.upper() == "NVT":
        hoomd.md.integrate.mode_standard(dt=dt)
        integrator = hoomd.md.integrate.nvt(group=all, kT=temperature, tau=1/gamma)
        integrator.randomize_velocities(seed=seed)
    elif mode.upper() == "NPT":
        hoomd.md.integrate.mode_standard(dt=dt)
        integrator1 = hoomd.md.integrate.nvt(group=all, kT=temperature, tau=1/gamma)
        eq = hoomd.dump.gsd(outputprefix+'.eq.gsd', period=dump_frequency, group=all, overwrite=True)
        integrator1.randomize_velocities(seed=seed)
        hoomd.run(100000)
        eq.disable()
        integrator1.disable()
        integrator = hoomd.md.integrate.npt(group=all, kT=1.0, tau=100*dt, tauP=100*dt, P=1e-8)
        #integrator.randomize_velocities(seed=seed)
    else:
        print("Mode '%s' not supported"%mode)
        sys.exit(1)


    hoomd.dump.gsd(outputgsdfile, period=dump_frequency, group=all, overwrite=True)
    hoomd.run(nsteps)


    #very end of simulation, assuming successful end: write total number of steps to a progressfile
    fh = open(progressfile,'w')
    fh.write("%i\n"%totalsteps)
    fh.close()


    # Analysis starts below

    import clustering_analysis as CA
    filename = outputgsdfile
    traj = gsd.hoomd.open(filename, "r")
    numsnap = len(traj)
    if numsnap >= expected_num_frames:
        analyze_results = CA.analyze_cluster(filename,203,outputprefix)
        figure_path = CA.plot_clustering_time(analyze_results[0], analyze_results[1], analyze_results[2], outputprefix)
    else:
        print("Looks like simulations did not reach completion")
    sys.exit()