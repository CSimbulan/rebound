## AST425 Rebound Simulation
## Author: Chris Simbulan

## This file loads a simulation from a checkpoint based the seed range given as the argument.
## The program will continue integrating the simulation up until five billion years.

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
    ## Open the log, data and planet files for reading and writing
    logfile = open(logName, "a")
    datafile = open(dataName, "a")

    ## Create simulation
    sim = reb.Simulation.from_file(checkPrev)

    ## Write subheader for simulation number to log file
    logfile.write("\nSimulation for Seed {0} continuing from t = {1}\n\n".format(seed, sim.t))
    
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

    tindex = (np.where(times == sim.t)[0]) + 1
    if (tindex - 1) == times[-1]:
        return sim.N
    #times = np.linspace(sim.t, sim.t+3000, 1000)

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
    for i,time in enumerate(times[tindex:]):
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
        except reb.Exit_min_peri as error:
            sim.save(checkMinPeri)

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

