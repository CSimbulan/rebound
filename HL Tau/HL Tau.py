## AST425 Rebound Simulation
## Author: Chris Simbulan

## Import Libraries -----------------------------------------------------------------------------------------
import numpy as np
import rebound as reb
import reboundx as rebx
import copy
from itertools import combinations
import random
import os.path
import matplotlib.pyplot as plt
##import visual as vs
import time as ti
import sys

## Blank array to save to empty text file
emp = np.array([])

## Choose seed
seed_start = int(sys.argv[1])
seed_end = int(sys.argv[2])

## Check if special file exists, if not, create a blank text file   
if (os.path.exists("Special.txt") == False):
    np.savetxt("Special.txt", emp)

specialfile = open("Special.txt", "a")

## Create function to setup bodies with initial conditions
def setup(seed):
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
    for i in range(0,5):
        angles.append(random.random()*2*np.pi)
    
    ## Add bodies
    sim.add(m=1.0, id = 0) #Add solar mass planet
    logfile.write("Solar mass star added at center.\n")
    sim.add(m = Mj, a=13.2, e=0, inc = 0, f=angles[0], id = 1)
    logfile.write("Jupiter mass planet added 13.2 AU away and {0} degrees.\n".format(angles[0]*180/np.pi))
    sim.add(m = Mj, a=32.3, e=0, inc = 0, f=angles[1], id = 2)
    logfile.write("Jupiter mass planet added 32.3 AU away and {0} degrees.\n".format(angles[1]*180/np.pi))
    sim.add(m = Mj, a=64.2, e=0, inc = 0, f=angles[2], id = 3)
    logfile.write("Jupiter mass planet added 64.2 AU away and {0} degrees.\n".format(angles[2]*180/np.pi))
    sim.add(m = Mj, a=73.7, e=0, inc = 0, f=angles[3], id = 4)
    logfile.write("Jupiter mass planet added 73.7 AU away and {0} degrees.\n".format(angles[3]*180/np.pi))
    sim.add(m = Mj, a=91.0, e=0, inc = 0, f=angles[4], id = 5)
    logfile.write("Jupiter mass planet added 91.0 AU away and {0} degrees.\n".format(angles[4]*180/np.pi))
    sim.move_to_com()
    
    ## Save checkpoint of starting point
    sim.save("Save States/HL_Tau_Seed_{0}_Start.bin".format(seed))
    
    ## Original positions (13.6, 33.3, 65.1, 77.3, 93.0)
    ## ALMA (13.2, 32.3, 64.2, 73.7, 91.0)
    
    return sim

## Function to merge particles
def mergeParticles(sim, time):
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
            ## Retreive the indices of the two planets relative to the simulation particles
            index1 = particles.index(p1)
            index2 = particles.index(p2)
    
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
        count = -1
        index3 = 0
        for p in sim.particles:
            count += 1
            if p.id == mergedPlanet.id:
                index3 = count
        ## Write to log file
        logfile.write("Planets {0} and {1} have collided and merged at t = {2}.\n".format(str(id1), str(id2), time))
        print ("Planets {0} and {1} have collided and merged and became Planet {2} at t = {3}.\n".format(str(id1), str(id2), mergedPlanet.id, time))
    ## If particle 1 is the star, remove particle 2
    elif (cp1.id == 0):
        id1 = cp2.id
        sim.remove(id=id1)
        index3 = 0
        logfile.write("Star and planet {0} have collided at t = {1}.\n".format(str(id1), time))
        specialfile.write("Seed {0}: Planet has collided with star at t = {1}.\n".format(seed, time))
        print ("Star and planet {0} have collided at t = {1}.\n".format(str(id1), time))
    ## If particle 2 is the star, remove particle 1
    elif (cp2.id == 0):
        id1 = cp1.id
        sim.remove(id=id1)
        index3 = 0
        logfile.write("Star and planet {0} have collided at t = {1}.\n".format(str(id1), time))
        specialfile.write("Seed {0}: Planet has collided with star at t = {1}.\n".format(seed, time))
        print ("Star and planet {0} have collided at t = {1}.\n".format(str(id1), time))  
    sim.move_to_com()
    return index1, index2, index3

## Define function to eject particles
def ejectParticle(sim, time):    
    max_d2 = 0.
    count = -1
    i1 = 0
    for p in sim.particles:
        count += 1
        d2 = p.x*p.x + p.y*p.y + p.z*p.z
        if d2>max_d2:
            max_d2 = d2
            mid = p.id
            i1 = count
    sim.remove(id=mid)
    sim.move_to_com()    
    if mid != 0:
        logfile.write("Planet {0} has been ejected at t = {1}.\n".format(str(mid), time))
        print ("Planet {0} has been ejected at t = {1}.\n".format(str(mid), time))
    return i1

## Create visual display and visual timer
##scene = vs.display(title = "HL Tau Seed 40")
##tlabel = vs.label(pos = (0, 100, 0), text = "t = 0 yrs", box = False)

#### Define classes for visual aspect.
##
#### Class for planet trail
##class trail:
##    
##    ## Initializer, takes planet it is trailing, and size of trail
##    def __init__(self, planet, size):
##        self.t = []
##        self.size = size
##        ## Create list of smaller spheres that represent the trail
##        for i in range(0, size):
##            self.t.append(vs.sphere(pos = planet.pos, radius = 0.35, color = planet.color, opacity = 1.0 - (i/10.0)/2))
##            
##    ## Update the position of each sphere in the trail        
##    def update(self, planet):
##        ## Start from the end of the trail, update its position to the one infront
##        for i in range(self.size -1, -1, -1):
##            if i > 0:
##                self.t[i].pos = self.t[i - 1].pos
##            ## If the sphere is leading the trail, update position to the planet's position
##            elif i == 0:
##                self.t[i].pos = planet.pos
##                
##    ## Delete the trail, incase of merging and collision
##    def delete(self):
##        for tr in self.t:
##            tr.visible = False
##            del tr
##            
##    ## Change the color of the trail
##    def changecolor(self, newcolor):
##        for i in range(0, self.size):
##            self.t[i].color = newcolor
##            
#### Class for entire planet system            
##class planet_system:
##    
##    ## Initializer, takes the simulation particles to build from
##    def __init__(self, particles):
##        ## Empty lists to store planets and trails
##        self.planets = []
##        self.trails = []
##        ## Size of system (number of particles)
##        self.size = len(particles)
##        ## Color and radius options
##        self.colors = [ (1.0,1.0,1.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0), (1.0, 1.0, 0.0), (1.0, 0.0, 0.0), (0.7, 0.0, 1.0) ]
##        self.radii = [3, 1.5, 1.5, 1.5, 1.5, 1.5]
##        ## For each particle in the simulation, create a sphere and a trail 
##        for i in range(0, self.size):
##            ipos = (particles[i].x, particles[i].y, particles[i].z)
##            self.planets.append(vs.sphere(pos = ipos, radius = self.radii[i], color = self.colors[i]))
##            self.trails.append(trail(self.planets[i], 15))
##    
##    ## Update the positions of the planets and the trails
##    def update(self, particles):
##        for i in range(0, self.size):
##            ipos = (particles[i].x, particles[i].y, particles[i].z)
##            self.planets[i].pos = ipos
##            self.trails[i].update(self.planets[i])
##            
##    ## Remove a planet from the animation
##    def remove(self, index):
##        self.trails[index].delete()
##        del self.trails[index]
##        self.planets[index].visible = False
##        del self.planets[index]
##        self.size -= 1
##    
##    ## Blend two colors
##    def blend(self, color1, color2):
##        b1 = (color1[0] + color2[0])/2
##        b2 = (color1[1] + color2[1])/2
##        b3 = (color1[2] + color2[2])/2
##        return (b1, b2, b3)
##            
##    ## Merge two planets
##    def merge(self, index1, index2, index3, particles):
##        newcolor = self.blend(self.planets[index1].color, self.planets[index2].color)
##        ipos = (particles[index3].x, particles[index3].y, particles[index3].z)
##        self.planets.append(vs.sphere(pos = ipos, radius = 1.5, color = newcolor))
##        self.trails.append(trail(self.planets[-1], 15))
##        self.size += 1
          
def run_sim(seed):
    ## Create simulation
    sim = setup(seed)
    
    ## Create array of pointers
    ps = sim.particles
    
    ## Create visual system
##    system = planet_system(sim.particles)
    
    ## Set distance for collision in AU
    Rj = 0.000477894503
    Rst = 0.00038925688
    sim.exit_min_distance = Rst
    ## Set distance for ejection in AU
    sim.exit_max_distance = 1000.0
    
    ## Number of outputs
    Noutputs = 100000
    #times = np.linspace(0,1.0e6,Noutputs)
    times = np.logspace(0, 6, Noutputs)
    
    ## Count number of ejections
    NE = 0
    ## Count number of collisions
    NC = 0 
    
    ## Switch to stop recording large eccentricities and inclinations
    e_switch = True
    inc_switch = True
    
    ## Integrate the simulation
    for i,time in enumerate(times):
        try:
            sim.integrate(time)
        
        ## If a the particles come too close, merge them
        except reb.Encounter as error:
            i1, i2, i3 = mergeParticles(sim, time)
##            if (i1 != 0 and i2 != 0):
##                system.merge(i1, i2, i3, sim.particles)
##                system.remove(i1)
##                if i1 < i2:
##                    system.remove(i2-1)
##                elif i1 > i2:
##                    system.remove(i2)
##            elif i1 == 0:
##                system.planets[0].color = system.blend(system.planets[0].color, system.planets[i2].color)
##                system.remove(i2)
##            elif i2 == 0:
##                system.planets[0].color = system.blend(system.planets[0].color, system.planets[i1].color)
##                system.remove(i1)
            NC += 1
        ## If a particle reaches past the max distance, treat is as ejected
        except reb.Escape as error:
            i1 = ejectParticle(sim, time)
##            system.remove(i1)
            NE += 1
        
        ## Update the animation
##        system.update(sim.particles)
##        tlabel.text = "t = {0} yrs".format(time)
        
        if (time >= 5.0e5):
            #datafile.write("\nAt time t = {0}: ----------\n\n".format(time))  
            for o in sim.calculate_orbits():
                #print o
                ## Write the semi-major axis, eccentricity, and inclination to data file
                datafile.write("{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}\n".format(time, o.a, o.e,
                                                                            o.inc, o.Omega, o.omega, o.f))
                if (o.e > 0.999):
                    if e_switch == True:
                        specialfile.write("Seed {0}: Eccentricity over 0.999 detected at t = {1}.\n".format(seed, time))
                        e_switch = False
                if (o.inc > 0.7):
                    if inc_switch == True:
                        specialfile.write("Seed {0}: Inclination over 0.7 detected at t = {1}.\n".format(seed, time))
                        inc_switch = False
                        
        
        ## Add a short delay to smoothen out animation
##        if time >= 700 and time < 10000:
##            ti.sleep(0.0001)
##        if time >= 10000 and time < 100000:
##            ti.sleep(0.001)
##        if time >= 100000 and time < 500000:
##            ti.sleep(0.001)
            
        #ti.sleep(0.01)
    logfile.write("Number of planets at the end of the simulation: {0}\n\n".format(sim.N - 1))
    #print("Number of particles at the end of the simulation: %d."%sim.N)
    
    ## Save the checkpoint 
    sim.save("Save States/HL_Tau_Seed_{0}_Checkpoint.bin".format(seed))
    
    if (sim.N - 1 == 5):
        specialfile.write("Seed {0}: All five planets remained.\n".format(seed))
    if (sim.N - 1 == 1):
        specialfile.write("Seed {0}: Only one planet remained.\n".format(seed))
    if (NE == 4):
        specialfile.write("Seed {0}: Four planets have been ejected.\n".format(seed))
    if (NC == 4):
        specialfile.write("Seed {0}: All planets have collided and merged.\n".format(seed))
    
    return sim, sim.N, NE#, longitude, varpi

## For multiple seeds
for i in range(seed_start, seed_end):
    seed = i
    print ("Seed {0} Simulation\n\n".format(seed))
    ## Check if log file exists, if not, create a blank text file
    if (os.path.exists("Log Files/log_seed{0}.txt".format(seed)) == False):
        np.savetxt("Log Files/log_seed{0}.txt".format(seed), emp)

    ## Check if data file exists, if not, create a blank text file    
    if (os.path.exists("Data Files/data_seed{0}.txt".format(seed)) == False):
        np.savetxt("Data Files/data_seed{0}.txt".format(seed), emp)
    
    ## Check if planet file exists, if not, create a blank text file    
    if (os.path.exists("Planet Files/planets_seed{0}.txt".format(seed)) == False):
        np.savetxt("Planet Files/planets_seed{0}.txt".format(seed), emp)
    
    ## Open the log, data and planet files for reading and writing
    logfile = open("Log Files/log_seed{0}.txt".format(seed), "r+")
    datafile = open("Data Files/data_seed{0}.txt".format(seed), "r+")
    planetfile = open("Planet Files/planets_seed{0}.txt".format(seed), "r+")
    
    ## Write subheader for simulation number to log file
    logfile.write("Simulation for Seed {0}\n\n".format(seed))
    
    ## Run one simulation and retrieve number of planets remaining
    sim, NN, NE = run_sim(seed)
    
    ## Write number of planets left to planet file
    planetfile.write(str(NN - 1) + "\t" + str(NE) + "\n")
    
    ## Print statement to make sure the script is running
    print ("Number of planets at the end of simulation is {0} ---------------\n".format(NN - 1))
    
## Close all the files
logfile.close()
datafile.close()
planetfile.close()
specialfile.close()

print "Done."
