## AST425 Rebound Simulation
## Author: Chris Simbulan

## This file creates new simulations for a given seed range (first argument is start seed,
## second argument is the end seed. 

## Import Libraries -------------------------------------------------------------------------------------
import numpy as np
import rebound as reb
import reboundx as rebx
import copy
from itertools import combinations
import random
import os.path
import matplotlib.pyplot as plt
import time as ti
import sys
from os import system

## Blank array to save to empty text file
emp = np.array([])

## Choose seed
seed_start = int(sys.argv[1])
seed_end = int(sys.argv[2])

## Choose planet mass label
p_mass = str(sys.argv[3])

## This is to adjust the min.dt on the simulation if needed
## IAS15 value
##value = 0.0#float(sys.argv[3])
##tv = 0.0#float(sys.argv[4])

## Create function to setup bodies with initial conditions
def setup(seed, p_mass, logfile):
    sim = reb.Simulation()
    ## Define units, although these are the default
    sim.units = ("AU", "yr", "Msun")
    Mj = 9.542e-4 # Mass of Jupiter
    Mst = 2.856e-4 # Mass of Saturn
    Mnpt = 5.149e-5 # Mass of Neptune
    
    ## Random Seed
    random.seed(seed)
    
    ## Generate five random starting angles
    angles = []
    masses = []
    for i in range(0,5):
        angles.append(random.random()*2*np.pi)

    ## Generate jupter masses
    if p_mass == "Jupiters":
        masses = [Mj]*5
    ## Generate saturn masses
    if p_mass == "Saturns":
        masses = [Mst]*5
    ## Generate five random masses between Mnpt - 2*Mnpt
    if p_mass == "Neptunes":
        for j in range(0,5):
            masses.append((random.random() + 1)*Mnpt)
    
    ## Add bodies
    sim.add(m=1.0, id = 0) #Add solar mass planet
    logfile.write("Solar mass star added at center.\n")    
    sim.add(m = masses[0], a=13.2, e=0, inc = 0, f=angles[0], id = 1)
    logfile.write("{0} solar mass planet added 13.2 AU away and {1} degrees.\n".format(masses[0], angles[0]*180/np.pi))
    sim.add(m = masses[1], a=32.3, e=0, inc = 0, f=angles[1], id = 2)
    logfile.write("{0} solar mass planet added 32.3 AU away and {1} degrees.\n".format(masses[1], angles[1]*180/np.pi))
    sim.add(m = masses[2], a=64.2, e=0, inc = 0, f=angles[2], id = 3)
    logfile.write("{0} solar mass planet added 64.2 AU away and {1} degrees.\n".format(masses[2], angles[2]*180/np.pi))
    sim.add(m = masses[3], a=73.7, e=0, inc = 0, f=angles[3], id = 4)
    logfile.write("{0} solar mass planet added 73.7 AU away and {1} degrees.\n".format(masses[3], angles[3]*180/np.pi))
    sim.add(m = masses[4], a=91.0, e=0, inc = 0, f=angles[4], id = 5)
    logfile.write("{0} solar mass planet added 91.0 AU away and {1} degrees.\n".format(masses[4], angles[4]*180/np.pi))
    sim.move_to_com()

    ## Change the min_dt value if needed
    ##sim.ri_ias15.min_dt = value

    ## Set the min pericenter
##    sim.exit_min_peri = 0.2
    
    ## Save checkpoint of starting point
    checkStart = "Save States/{1}/HL_Tau_Seed_{0}_Start.bin".format(seed, p_mass)
    sim.save(checkStart)
 
    ## Original positions (13.6, 33.3, 65.1, 77.3, 93.0)
    ## ALMA (13.2, 32.3, 64.2, 73.7, 91.0)
    
    return sim

## Function to merge particles
def mergeParticles(sim, time, seed, logfile):
    ## Find two closest particles
    min_d2 = 1e9 # large number
    particles = sim.particles
    for p1, p2 in combinations(particles,2):
        dx = p1.x - p2.x
        dy = p1.y - p2.y
        dz = p1.z - p2.z
        d2 = dx*dx + dy*dy + dz*dz
        if d2<min_d2:
            min_d2 = d2
            cp1 = p1
            cp2 = p2
    
    ## Merge two closest particles, as long as neither is the star
    if (cp1.id != 0 and cp2.id != 0):
        mergedPlanet = reb.Particle()
        mergedPlanet.m  = cp1.m + cp2.m
        mergedPlanet.x  = (cp1.m*cp1.x  + cp2.m*cp2.x) /mergedPlanet.m
        mergedPlanet.y  = (cp1.m*cp1.y  + cp2.m*cp2.y) /mergedPlanet.m
        mergedPlanet.z  = (cp1.m*cp1.z  + cp2.m*cp2.z) /mergedPlanet.m
        mergedPlanet.vx = (cp1.m*cp1.vx + cp2.m*cp2.vx)/mergedPlanet.m
        mergedPlanet.vy = (cp1.m*cp1.vy + cp2.m*cp2.vy)/mergedPlanet.m
        mergedPlanet.vz = (cp1.m*cp1.vz + cp2.m*cp2.vz)/mergedPlanet.m
        mergedPlanet.id = cp1.id
        id1 = cp1.id
        id2 = cp2.id
        sim.remove(id=id1)
        sim.remove(id=id2)
        sim.add(mergedPlanet)
        ## Write to log file
        logfile.write("Planets {0} and {1} have collided and merged at t = {2}.\n".format(str(id1), str(id2), time))
        print ("Planets {0} and {1} have collided and merged and became Planet {2} at t = {3}.\n".format(str(id1), str(id2), mergedPlanet.id, time))
    ## If particle 1 is the star, remove particle 2
    elif (cp1.id == 0):
        id1 = cp2.id
        sim.remove(id=id1)
        logfile.write("Star and planet {0} have collided at t = {1}.\n".format(str(id1), time))
        logfile.write("S: Seed {0}: Planet has collided with star at t = {1}.\n".format(seed, time))
        print ("Star and planet {0} have collided at t = {1}.\n".format(str(id1), time))
    ## If particle 2 is the star, remove particle 1
    elif (cp2.id == 0):
        id1 = cp1.id
        sim.remove(id=id1)
        logfile.write("Star and planet {0} have collided at t = {1}.\n".format(str(id1), time))
        logfile.write("S: Seed {0}: Planet has collided with star at t = {1}.\n".format(seed, time))
        print ("Star and planet {0} have collided at t = {1}.\n".format(str(id1), time))  
    sim.move_to_com()

## Define function to eject particles
def ejectParticle(sim, time, seed, logfile):    
    max_d2 = 0.
    for p in sim.particles:
        d2 = p.x*p.x + p.y*p.y + p.z*p.z
        if d2>max_d2:
            max_d2 = d2
            mid = p.id
    sim.remove(id=mid)
    sim.move_to_com()    
    if mid != 0:
        logfile.write("Planet {0} has been ejected at t = {1}.\n".format(str(mid), time))
        print ("Planet {0} has been ejected at t = {1}.\n".format(str(mid), time))

def run_sim(seed):

    global p_mass

    ## Strings for names of files
    dataName = "Data Files/{1}/data_seed{0}_{1}.txt".format(seed, p_mass)
    logName = "Log Files/{1}/log_seed{0}_{1}.txt".format(seed, p_mass)
    checkPoint = "Save States/{1}/HL_Tau_Seed_{0}_{1}.bin".format(seed, p_mass)
    checkPrev = "Save States/{1}/HL_Tau_Seed_{0}_{1}prev.bin".format(seed, p_mass)
    checkMinPeri = "Save States/{1}/HL_Tau_Seed_{0}_{1}minperi.bin".format(seed, p_mass)
    heartbeatPath = "HB Files/{1}/heartbeat_seed{0}_{1}.txt".format(seed, p_mass)

    print ("Seed {0} Simulation\n\n".format(seed))
    ## Check if log file exists, if not, create a blank text file
    if (os.path.exists(logName) == False):
        np.savetxt(logName, emp)

    ## Check if data file exists, if not, create a blank text file    
    if (os.path.exists(dataName) == False):
        np.savetxt(dataName, emp)
    
    ## Open the log, data and planet files for reading and writing
    logfile = open(logName, "r+")
    datafile = open(dataName, "r+")

    ## Write subheader for simulation number to log file
    logfile.write("Simulation for Seed {0}\n\n".format(seed))
    
    ## Create simulation
    sim = setup(seed, p_mass, logfile)
    
    ## Create array of pointers
    ps = sim.particles
    
    ## Set distance for collision in AU
    Rj = 0.000477894503
    Rst = 0.00038925688
    sim.exit_min_distance = Rst
    ## Set distance for ejection in AU
    sim.exit_max_distance = 1000.0
    
    ## Number of outputs
    Noutputs = 1000
    ## setting this base to this value allows to extent to 5e9
    times = np.logspace(0, 9, Noutputs, base = 11.958131745004017)

    ## Calculate initial energy
    E0 = sim.calculate_energy()
    Eold = E0
    
    ## Switch to stop recording large eccentricities and inclinations
    e_switch = True
    inc_switch = True

    ## Close all the files
    logfile.close()
    datafile.close()

    ## Integrate the simulation
    for i,time in enumerate(times):
	print time
        ## Open the log, data and planet files for reading and writing
        logfile = open(logName, "a")
        datafile = open(dataName, "a")
        try:
            sim.integrate(time)
        
        ## If a the particles come too close, merge them
        except reb.Encounter as error:
            Ei = sim.calculate_energy()
            mergeParticles(sim, time, seed, logfile)
            Ef = sim.calculate_energy()
            dE = Ef - Ei
            E0 += dE

        ## If a particle reaches past the max distance, treat is as ejected
        except reb.Escape as error:
            Ei = sim.calculate_energy()
            ejectParticle(sim, time, seed, logfile)
            Ef = sim.calculate_energy()
            dE = Ef - Ei
            E0 += dE

        ## If a particles comes too close to the star, save a checkpoint
        ##except reb.Exit_min_peri as error:
        ##    sim.save(checkMinPeri)

        E = sim.calculate_energy()
        Eerror = abs(E - E0)/abs(Eold)

        for o in sim.calculate_orbits():
            #print o
            ## Write the semi-major axis, eccentricity, and inclination to data file
            datafile.write("{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}\t{7}\n".format(time, o.a, o.e,
                                                                        o.inc, o.Omega, o.omega, o.f, Eerror))
            if (o.e > 0.999):
                if e_switch == True:
                    logfile.write("S: Seed {0}: Eccentricity over 0.999 detected at t = {1}.\n".format(seed, time))
                    e_switch = False
            if (o.inc > 0.7):
                if inc_switch == True:
                    logfile.write("S: Seed {0}: Inclination over 0.7 detected at t = {1}.\n".format(seed, time))
                    inc_switch = False
                        
        ## Close all the files
        logfile.close()
        datafile.close()

        ## Save the simulation
        sim.save(checkPoint)
	system("mv Save\ States/{1}/HL_Tau_Seed_{0}_{1}.bin Save\ States/{1}/HL_Tau_Seed_{0}_{1}prev.bin".format(seed, p_mass))

    logfile = open(logName, "a")
    datafile = open(dataName, "a")

    logfile.write("Number of planets at the end of the simulation: {0}\n\n".format(sim.N - 1))
    #print("Number of particles at the end of the simulation: %d."%sim.N)
    
    if (sim.N - 1 == 5):
        logfile.write("S: Seed {0}: All five planets remained.\n".format(seed))
    if (sim.N - 1 == 1):
        logfile.write("S: Seed {0}: Only one planet remained.\n".format(seed))

    ## Close all the files
    logfile.close()
    datafile.close()
	
    return sim

## For multiple seeds
def hltau(seed):
    
    ## Run one simulation and retrieve number of planets remaining
    sim = run_sim(seed)
    
    print "Done."

seedrange = range(seed_start, seed_end)
pool = reb.InterruptiblePool(8)
pool.map(hltau, seedrange)

