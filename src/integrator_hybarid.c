/**
 * @file 	integrator.c
 * @brief 	Leap-frog integration scheme.
 * @author 	Hanno Rein <hanno@hanno-rein.de>
 * @details	This file implements the leap-frog integration scheme.  
 * This scheme is second order accurate, symplectic and well suited for 
 * non-rotating coordinate systems. Note that the scheme is formally only
 * first order accurate when velocity dependent forces are present.
 * 
 * @section 	LICENSE
 * Copyright (c) 2011 Hanno Rein, Shangfei Liu
 *
 * This file is part of rebound.
 *
 * rebound is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * rebound is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with rebound.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include "rebound.h"
#include "integrator_ias15.h"
#include "integrator_whfast.h"

static void reb_integrator_hybarid_check_for_encounter(struct reb_simulation* r);

void reb_integrator_hybarid_additional_forces_mini(struct reb_simulation* mini){
    if (mini->passive_influence){
        struct reb_simulation* r = mini->ri_hybarid.global;
        struct reb_particle* global = r->particles;
        struct reb_particle* mini_particles = mini->particles;
        struct reb_particle* global_prev = r->ri_hybarid.particles_prev;
        const double t_prev = r->t - r->dt;
        const double timefac = (mini->t - t_prev)/r->dt;
        const int N_active = r->N_active;
        const double G = r->G;
        for(int i=N_active;i<r->N;i++){    //planetesimals
            if(r->ri_hybarid.is_in_mini[i]==0){
                const double ix = (1.-timefac)*global_prev[i].x + timefac*global[i].x; //interpolated values
                const double iy = (1.-timefac)*global_prev[i].y + timefac*global[i].y;
                const double iz = (1.-timefac)*global_prev[i].z + timefac*global[i].z;
                const double mp = global[i].m;
                for(int j=0;j<N_active;j++){//massive bodies
                    struct reb_particle* body = &(mini_particles[j]);
                    const double ddx = body->x - ix;
                    const double ddy = body->y - iy;
                    const double ddz = body->z - iz;
                    
                    const double rijinv2 = 1.0/(ddx*ddx + ddy*ddy + ddz*ddz);
                    const double ac = -G*mp*rijinv2*sqrt(rijinv2);
                    
                    body->ax += ac*ddx;     //perturbation on planets due to planetesimals.
                    body->ay += ac*ddy;
                    body->az += ac*ddz;
                }
            }
        }
    }
    /*
    if (mini->passive_influence){
        const double G = mini->G;
        const int N_active = mini->N_active;
        struct reb_simulation* global = mini->ri_hybarid.global;
        struct reb_particle* particles_mini = mini->particles;
        struct reb_particle* particles_global = global->particles;
        struct reb_particle* particles_global_prev = global->ri_hybarid.particles_prev;
        
        //forces from global into mini
        double t_prev = global->t - global->dt;
        const double timefac = (mini->t - t_prev)/global->dt;
        for(int i=N_active;i<global->N;i++){    //planetesimals
            if(global->ri_hybarid.is_in_mini[i]==0){             //find planetesimals which is part of global but not mini
                const double ix = timefac*particles_global[i].x - (1.-timefac)*particles_global_prev[i].x; //interpolated values
                const double iy = timefac*particles_global[i].y - (1.-timefac)*particles_global_prev[i].y;
                const double iz = timefac*particles_global[i].z - (1.-timefac)*particles_global_prev[i].z;
                const double Gm1 = G*particles_global[i].m;
                for(int j=0;j<N_active;j++){//massive bodies
                    struct reb_particle* body = &(particles_mini[j]);
                    const double ddx = body->x - ix;
                    const double ddy = body->y - iy;
                    const double ddz = body->z - iz;
                    
                    const double rijinv2 = 1.0/(ddx*ddx + ddy*ddy + ddz*ddz);
                    const double ac = -Gm1*rijinv2*sqrt(rijinv2);
                    
                    body->ax += ac*ddx;     //perturbation on planets due to planetesimals.
                    body->ay += ac*ddy;
                    body->az += ac*ddz;
                }
            }
        }
    }*/
}

void reb_integrator_hybarid_part1(struct reb_simulation* r){
	const int _N_active = ((r->N_active==-1)?r->N:r->N_active) - r->N_var;
    if (r->ri_hybarid.mini == NULL){
        r->ri_hybarid.mini = reb_create_simulation();
        r->ri_hybarid.mini->usleep = -1; // Disable visualiation
        r->ri_hybarid.mini->integrator = REB_INTEGRATOR_IAS15;
        r->ri_hybarid.mini->additional_forces = reb_integrator_hybarid_additional_forces_mini;
        r->ri_hybarid.mini->ri_hybarid.global = r;
        r->ri_hybarid.mini->passive_influence = r->passive_influence;
    }

    // Remove all particles from mini
    r->ri_hybarid.mini->t = r->t;
    r->ri_hybarid.mini->N = 0;
    r->ri_hybarid.mini->N_active = -1;
    r->ri_hybarid.mini_active = 0;
    r->ri_hybarid.encounter_index_N = 0;
    
    
    if (r->N>r->ri_hybarid.is_in_mini_Nmax){
        r->ri_hybarid.is_in_mini_Nmax = r->N;
        r->ri_hybarid.is_in_mini = realloc(r->ri_hybarid.is_in_mini,r->N*sizeof(int));
    }

    // Add all massive particles
    for (int i=0; i<_N_active; i++){
        reb_add(r->ri_hybarid.mini, r->particles[i]);
        r->ri_hybarid.is_in_mini[i] = 1;
        if (r->ri_hybarid.encounter_index_N>=r->ri_hybarid.encounter_index_Nmax){
            r->ri_hybarid.encounter_index_Nmax += 32;
            r->ri_hybarid.encounter_index = realloc(r->ri_hybarid.encounter_index,r->ri_hybarid.encounter_index_Nmax*sizeof(int));
        }
        r->ri_hybarid.encounter_index[r->ri_hybarid.encounter_index_N] = i;
        r->ri_hybarid.encounter_index_N++;
    }
    r->ri_hybarid.mini->N_active = _N_active;

    reb_integrator_hybarid_check_for_encounter(r);

    //keep this after check_for_encounter - incase particle removed, don't have to re-organize arrays
    if (r->passive_influence){
        if (r->N>r->ri_hybarid.particles_prev_Nmax){
            r->ri_hybarid.particles_prev_Nmax = r->N;
            r->ri_hybarid.particles_prev = realloc(r->ri_hybarid.particles_prev,r->N*sizeof(struct reb_particle));
        }
        memcpy(r->ri_hybarid.particles_prev, r->particles, sizeof(struct reb_particle)*r->N);
    }
    
    reb_integrator_whfast_part1(r);
}
void reb_integrator_hybarid_part2(struct reb_simulation* r){
    reb_integrator_whfast_part2(r);

    struct reb_simulation* mini = r->ri_hybarid.mini;
    if (r->ri_hybarid.mini_active){
        reb_integrate(mini,r->t);
        for (int i=0; i<mini->N; i++){
            r->particles[r->ri_hybarid.encounter_index[i]] = mini->particles[i];
            r->particles[r->ri_hybarid.encounter_index[i]].sim = r;
        }
    }


    // Check for encounters
    //   if new encounters, copy from global to mini
    //
    // Copy positions from mini to global
    //
    // Store the pos to prev_pos
    //
}
	
void reb_integrator_hybarid_synchronize(struct reb_simulation* r){
	// Do nothing.
    reb_integrator_whfast_synchronize(r);
}

void reb_integrator_hybarid_reset(struct reb_simulation* r){
	// Do nothing.
    reb_integrator_whfast_reset(r);
}

static void reb_integrator_hybarid_check_for_encounter(struct reb_simulation* r){
    struct reb_simulation* mini = r->ri_hybarid.mini;
    const int N = r->N;
	const int _N_active = ((r->N_active==-1)?N:r->N_active) - r->N_var;
    struct reb_particle* global = r->particles;
    struct reb_particle p0 = global[0];
    double ejectiondistance2 = 100;
    for (int i=0; i<_N_active; i++){
        struct reb_particle* pi = &(global[i]);
        const double dxi = p0.x - pi->x;
        const double dyi = p0.y - pi->y;
        const double dzi = p0.z - pi->z;
        const double r0i2 = dxi*dxi + dyi*dyi + dzi*dzi;
        const double rhi = r0i2*pow(pi->m/(p0.m*3.),2./3.);
        for (int j=i+1; j<N; j++){
            struct reb_particle pj = global[j];
            double HSR = r->ri_hybarid.switch_ratio;
            
            const double dxj = p0.x - pj.x;
            const double dyj = p0.y - pj.y;
            const double dzj = p0.z - pj.z;
            const double r0j2 = dxj*dxj + dyj*dyj + dzj*dzj;
            const double rhj = r0j2*pow(pj.m/(p0.m*3.),2./3.);
            
            const double dx = pi->x - pj.x;
            const double dy = pi->y - pj.y;
            const double dz = pi->z - pj.z;
            const double rij2 = dx*dx + dy*dy + dz*dz;
            const double ratio = rij2/(rhi+rhj);    //(p-p distance/Hill radii)^2
            r->ri_hybarid.is_in_mini[j] = 0;
            
            double radius2 = pi->r*pi->r;
            double dx1 = (pj.vx - pi->vx)*r->dt; //xf - xi = distance travelled in dt relative to body
            double dy1 = (pj.vy - pi->vy)*r->dt;
            double dz1 = (pj.vz - pi->vz)*r->dt;
            double dx2 = pj.x - pi->x;
            double dy2 = pj.y - pi->y;
            double dz2 = pj.z - pi->z;
            double x = dy1*dz2 - dz1*dy2;
            double y = dz1*dx2 - dx1*dz2;
            double z = dx1*dy2 - dy1*dx2;
            double d2 = (x*x + y*y + z*z)/(dx1*dx1 + dy1*dy1 + dz1*dz1);
            
            //if(pj.id == -100 || rij2 < pi->r*pi->r || d2 < radius2){//collision
            if(pj.id == -100 || rij2 < pi->r*pi->r){
                double invmass = 1.0/(pi->m + pj.m);
                double Ei = reb_tools_energy(r);
                
                pi->vx = (pi->vx*pi->m + pj.vx*pj.m)*invmass;
                pi->vy = (pi->vy*pi->m + pj.vy*pj.m)*invmass;
                pi->vz = (pi->vz*pi->m + pj.vz*pj.m)*invmass;
                pi->m += pj.m;
                mini->particles[i] = *pi;     //need to update mini accordingly
                
                reb_remove(r,j,1);
                
                double Ef = reb_tools_energy(r);
                double dE_collision = Ei - Ef;
                printf("\n\tParticle %d collided with body %d from system at t=%f\n",j,i,r->t);
                
            } else if(ratio < HSR){
                r->ri_hybarid.mini_active = 1;
                if (j>_N_active){
                    reb_add(mini,pj);
                    r->ri_hybarid.is_in_mini[j] = 1;
                    if (r->ri_hybarid.encounter_index_N>=r->ri_hybarid.encounter_index_Nmax){
                        while(r->ri_hybarid.encounter_index_N>=r->ri_hybarid.encounter_index_Nmax) r->ri_hybarid.encounter_index_Nmax += 32;
                        r->ri_hybarid.encounter_index = realloc(r->ri_hybarid.encounter_index,r->ri_hybarid.encounter_index_Nmax*sizeof(int));
                    }
                    r->ri_hybarid.encounter_index[r->ri_hybarid.encounter_index_N] = j;
                    r->ri_hybarid.encounter_index_N++;
                }
                
            } else if (r0j2 > ejectiondistance2){
                double Ei = reb_tools_energy(r);
                reb_remove(r,j,1);
                double Ef = reb_tools_energy(r);
                double dE_collision = Ei - Ef;
                printf("\n\tParticle %d ejected from system at t=%f\n",j,r->t);
            }
        }
    }
}
