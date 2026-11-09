#include "filling.h"
#include<stdlib.h>
#include <math.h>//Pushkar:Make sure it is linked while compilation
#include <assert.h>//Pushkar:Added for SanityCheck
#include <time.h>

/*
 * Fills a cylinder at the location specified by the cylinder object
 */
void fillCylinder(double *phi, cylinder Cylinder,
    domainInfo simDomain, subdomainInfo subdomain)
{
    long phaseStep = subdomain.numCompCells;
    long index;
    long xG, yG;     // Global coordinates
    long x, y, z;        // Local coordinates

    double sum = 0.0;

    for (x = subdomain.xS_r; x < subdomain.xE_r; x++)
    {
        for (y = subdomain.yS_r; y < subdomain.yE_r; y++)
        {
            for (z = subdomain.zS_r; z < subdomain.zE_r; z++)
            {
                index = x*subdomain.xStep + y*subdomain.yStep + z;

                // Computing global coordinates
                xG = subdomain.xS + x - subdomain.xS_r;
                yG = subdomain.yS + y - subdomain.yS_r;
                // zG = subdomain.zS + z - subdomain.zS_r;

                if (Cylinder.phase != simDomain.numPhases-1)
                {
                    // Circle (cylinder) interior/exterior check
                    if ((Cylinder.xC - xG)*(Cylinder.xC - xG)
                    + (Cylinder.yC - yG)*(Cylinder.yC - yG)
                    <= Cylinder.radius*Cylinder.radius)
                    {
                        for (long i = 0; i < simDomain.numPhases-1; i++)
                        {
                            if (i == Cylinder.phase)
                                phi[i*phaseStep + index] = 1.0;
                            else
                                phi[i*phaseStep + index] = 0.0;
                        }
                        phi[(simDomain.numPhases-1)*phaseStep + index] = 0.0;
                    }
                    else
                    {
                        if (phi[Cylinder.phase*phaseStep + index] != 1.0)
                            phi[Cylinder.phase*phaseStep + index] = 0.0;
                    }
                }
                else
                {
                    sum = 0.0;
                    for (long i = 0; i < simDomain.numPhases-1; i++)
                    {
                        sum += phi[i*phaseStep + index];
                    }
                    if (sum <= 1.0 && sum >= 0.0)
                        phi[(simDomain.numPhases-1)*phaseStep + index] = 1.0 - sum;
                    else
                    {
                        for (long i = 0; i < simDomain.numPhases-1; i++)
                        {
                            phi[i*phaseStep + index] = 0.0;
                        }
                        phi[(simDomain.numPhases-1)*phaseStep + index] = 1.0;
                    }
                }
            }
        }
    }
}
void fillCylinderNEXLP(double *phi, cylinder Cylinder,
    domainInfo simDomain, subdomainInfo subdomain)
{
    long phaseStep = subdomain.numCompCells;
    long index;
    long xG, yG;     // Global coordinates
    long x, y, z;        // Local coordinates

    double sum = 0.0;

    for (x = subdomain.xS_r; x < subdomain.xE_r; x++)
    {
        for (y = subdomain.yS_r; y < subdomain.yE_r; y++)
        {
            for (z = subdomain.zS_r; z < subdomain.zE_r; z++)
            {
                index = x*subdomain.xStep + y*subdomain.yStep + z;

                // Computing global coordinates
                xG = subdomain.xS + x - subdomain.xS_r;
                yG = subdomain.yS + y - subdomain.yS_r;
                // zG = subdomain.zS + z - subdomain.zS_r;

                // Circle (cylinder) interior/exterior check
                if ((Cylinder.xC - xG)*(Cylinder.xC - xG)
                + (Cylinder.yC - yG)*(Cylinder.yC - yG)
                <= Cylinder.radius*Cylinder.radius)
                {
                    for (long i = 0; i < simDomain.numPhases; i++)
                    {
                        if (i == Cylinder.phase)
                            phi[i*phaseStep + index] = 1.0;
                        else
                            phi[i*phaseStep + index] = 0.0;
                    }
                }
                else
                {
                    if (phi[Cylinder.phase*phaseStep + index] != 1.0)
                        phi[Cylinder.phase*phaseStep + index] = 0.0;
                }
            }
        }
    }
    for (x = subdomain.xS_r; x < subdomain.xE_r; x++){
        for (y = subdomain.yS_r; y < subdomain.yE_r; y++){
            for (z = subdomain.zS_r; z < subdomain.zE_r; z++){
                index = x*subdomain.xStep + y*subdomain.yStep + z;
                sum = 0.0;
                for (long i = 0; i < simDomain.numPhases-1; i++)
                {
                    sum += phi[i*phaseStep + index];
                }
                if (sum <= 1.0 && sum >= 0.0)
                    phi[(simDomain.numPhases-1)*phaseStep + index] = 1.0 - sum;
                else
                {
                    for (long i = 0; i < simDomain.numPhases-1; i++)
                    {   
                        if(i!=Cylinder.phase)
                            phi[i*phaseStep + index] = 0.0;
                    }
                    phi[Cylinder.phase*phaseStep + index] = 1.0;
                }   
            }
        }
    }   
}

void fillSphere(double *phi, sphere Sphere,
                domainInfo simDomain, subdomainInfo subdomain)
{
    long phaseStep = subdomain.numCompCells;
    long index;
    long xG, yG, zG;    // Global coordinates
    long x, y, z;       // Local coordinates

    double sum = 0.0;

    for (x = subdomain.xS_r; x < subdomain.xE_r; x++)
    {
        for (y = subdomain.yS_r; y < subdomain.yE_r; y++)
        {
            for (z = subdomain.zS_r; z < subdomain.zE_r; z++)
            {
                index = x*subdomain.xStep + y*subdomain.yStep + z;

                // Computing global coordinates
                xG = subdomain.xS + x - subdomain.xS_r;
                yG = subdomain.yS + y - subdomain.yS_r;
                zG = subdomain.zS + z - subdomain.zS_r;

                if (Sphere.phase != simDomain.numPhases-1)
                {
                    // Circle (Sphere)interior/exterior check
                    if ((Sphere.xC - xG)*(Sphere.xC - xG)
                    + (Sphere.yC - yG)*(Sphere.yC - yG)
                    + (Sphere.zC - zG)*(Sphere.zC - zG)
                    <= Sphere.radius*Sphere.radius)
                    {
                        for (long i = 0; i < simDomain.numPhases-1; i++)
                        {
                            if (i == Sphere.phase)
                                phi[i*phaseStep + index] = 1.0;
                            else
                                phi[i*phaseStep + index] = 0.0;
                        }
                        phi[(simDomain.numPhases-1)*phaseStep + index] = 0.0;
                    }
                    else
                    {
                        if (phi[Sphere.phase*phaseStep + index] != 1.0)
                            phi[Sphere.phase*phaseStep + index] = 0.0;
                    }
                }
                else
                {
                    sum = 0.0;
                    for (long i = 0; i < simDomain.numPhases-1; i++)
                    {
                        sum += phi[i*phaseStep + index];
                    }
                    if (sum <= 1.0 && sum >= 0.0)
                        phi[(simDomain.numPhases-1)*phaseStep + index] = 1.0 - sum;
                    else
                    {
                        for (long i = 0; i < simDomain.numPhases-1; i++)
                        {
                            phi[i*phaseStep + index] = 0.0;
                        }
                        phi[(simDomain.numPhases-1)*phaseStep + index] = 1.0;
                    }
                }
            }
        }
    }
}
void fillSphereNEXLP(double *phi, sphere Sphere,
                domainInfo simDomain, subdomainInfo subdomain)
{
    long phaseStep = subdomain.numCompCells;
    long index;
    long xG, yG, zG;    // Global coordinates
    long x, y, z;       // Local coordinates

    double sum = 0.0;

    for (x = subdomain.xS_r; x < subdomain.xE_r; x++)
    {
        for (y = subdomain.yS_r; y < subdomain.yE_r; y++)
        {
            for (z = subdomain.zS_r; z < subdomain.zE_r; z++)
            {
                index = x*subdomain.xStep + y*subdomain.yStep + z;

                // Computing global coordinates
                xG = subdomain.xS + x - subdomain.xS_r;
                yG = subdomain.yS + y - subdomain.yS_r;
                //zG = subdomain.zS + z - subdomain.zS_r;

                    // Sphere interior/exterior check
                if ((Sphere.xC - xG) * (Sphere.xC - xG)
                    + (Sphere.yC - yG) * (Sphere.yC - yG)
                    + (Sphere.zC - zG) * (Sphere.zC - zG)
                    <= Sphere.radius * Sphere.radius)
                {
                    for (long i = 0; i < simDomain.numPhases - 1; i++)
                    {
                        if (i == Sphere.phase)
                            phi[i * phaseStep + index] = 1.0;
                        else
                            phi[i * phaseStep + index] = 0.0;
                    }
                    phi[(simDomain.numPhases - 1) * phaseStep + index] = 0.0;
                }
                else
                {
                    if (phi[Sphere.phase * phaseStep + index] != 1.0)
                        phi[Sphere.phase * phaseStep + index] = 0.0;
                } 
            }
        }
    }
    for (x = subdomain.xS_r; x < subdomain.xE_r; x++) {
        for (y = subdomain.yS_r; y < subdomain.yE_r; y++) {
            for (z = subdomain.zS_r; z < subdomain.zE_r; z++) {
                index = x * subdomain.xStep + y * subdomain.yStep + z;
                sum = 0.0;
                for (long i = 0; i < simDomain.numPhases - 1; i++) {
                    sum += phi[i * phaseStep + index];
                }
                if (sum <= 1.0 && sum >= 0.0) {
                    phi[(simDomain.numPhases - 1) * phaseStep + index] = 1.0 - sum;
                } else {
                    for (long i = 0; i < simDomain.numPhases - 1; i++) {
                        if (i != Sphere.phase)
                            phi[i * phaseStep + index] = 0.0;
                    }
                    phi[Sphere.phase * phaseStep + index] = 1.0;
                }
            }
        }
    }
}

void fillCube(double *phi, cube Cube,
              domainInfo simDomain, subdomainInfo subdomain)
{
    long phaseStep = subdomain.numCompCells;
    long index;
    long xG, yG, zG;     // Global coordinates

    long x, y, z;

    double sum = 0.0;

    for (x = subdomain.xS_r; x < subdomain.xE_r; x++)
    {
        for (y = subdomain.yS_r; y < subdomain.yE_r; y++)
        {
            for (z = subdomain.zS_r; z < subdomain.zE_r; z++)
            {
                index = x*subdomain.xStep + y*subdomain.yStep + z;

                // Computing global coordinates
                xG = subdomain.xS + x - subdomain.xS_r;
                yG = subdomain.yS + y - subdomain.yS_r;
                zG = subdomain.zS + z - subdomain.zS_r;

                if (Cube.phase != simDomain.numPhases-1)
                {
                    // Circle (cylinder) interior/exterior check
                    if (Cube.xS <= xG && Cube.xE >= xG && Cube.yS <= yG && Cube.yE >= yG && Cube.zS <= zG && Cube.zE >= zG)
                    {
                        for (long i = 0; i < simDomain.numPhases-1; i++)
                        {
                            if (i == Cube.phase)
                                phi[i*phaseStep + index] = 1.0;
                            else
                                phi[i*phaseStep + index] = 0.0;
                        }
                        phi[(simDomain.numPhases-1)*phaseStep + index] = 0.0;
                    }
                    else
                    {
                        if (phi[Cube.phase*phaseStep + index] != 1.0)
                            phi[Cube.phase*phaseStep + index] = 0.0;
                    }
                }
                else
                {
                    sum = 0.0;
                    for (long i = 0; i < simDomain.numPhases-1; i++)
                    {
                        sum += phi[i*phaseStep + index];
                    }
                    if (sum <= 1.0 && sum >= 0.0)
                        phi[(simDomain.numPhases-1)*phaseStep + index] = 1.0 - sum;
                    else
                    {
                        for (long i = 0; i < simDomain.numPhases-1; i++)
                        {
                            phi[i*phaseStep + index] = 0.0;
                        }
                        phi[(simDomain.numPhases-1)*phaseStep + index] = 1.0;
                    }
                }
            }
        }
    }
}

void fillEllipse(double *phi, ellipse Ellipse,
                 domainInfo simDomain, subdomainInfo subdomain)
{
    long phaseStep = subdomain.numCompCells;
    long index;
    long xG, yG, zG;     // Global coordinates

    long x, y, z;

    double sum = 0.0;

    for (x = subdomain.xS_r; x < subdomain.xE_r; x++)
    {
        for (y = subdomain.yS_r; y < subdomain.yE_r; y++)
        {
            for (z = subdomain.zS_r; z < subdomain.zE_r; z++)
            {
                index = x*subdomain.xStep + y*subdomain.yStep + z;

                // Computing global coordinates
                xG = subdomain.xS + x - subdomain.xS_r;
                yG = subdomain.yS + y - subdomain.yS_r;
                zG = subdomain.zS + z - subdomain.zS_r;

                if (Ellipse.phase != simDomain.numPhases-1)
                {
                    // Sphere interior/exterior check
                    if ((Ellipse.xC - xG)*(Ellipse.xC - xG)
                        + (Ellipse.yC - yG)*(Ellipse.yC - yG)
                        + (Ellipse.zC - zG)*(Ellipse.zC - zG)
                        <= Ellipse.major_axis*Ellipse.major_axis)
                    {
                        for (long i = 0; i < simDomain.numPhases-1; i++)
                        {
                            if (i == Ellipse.phase)
                                phi[i*phaseStep + index] = 1.0;
                            else
                                phi[i*phaseStep + index] = 0.0;
                        }
                        phi[(simDomain.numPhases-1)*phaseStep + index] = 0.0;
                    }
                    else
                    {
                        if (phi[Ellipse.phase*phaseStep + index] != 1.0)
                            phi[Ellipse.phase*phaseStep + index] = 0.0;
                    }
                }
                else
                {
                    sum = 0.0;
                    for (long i = 0; i < simDomain.numPhases-1; i++)
                    {
                        sum += phi[i*phaseStep + index];
                    }
                    if (sum <= 1.0 && sum >= 0.0)
                        phi[(simDomain.numPhases-1)*phaseStep + index] = 1.0 - sum;
                    else
                    {
                        for (long i = 0; i < simDomain.numPhases-1; i++)
                        {
                            phi[i*phaseStep + index] = 0.0;
                        }
                        phi[(simDomain.numPhases-1)*phaseStep + index] = 1.0;
                    }
                }
            }
        }
    }
}

void fillComposition(double *phi, double *comp,
                     domainInfo simDomain, subdomainInfo subdomain,
                     double ***ceq, double ***cfill)
{
    long index;
    long PHASE_FILLED = 0;
    long step = subdomain.numCompCells;
    long x, y, z;

    for (x = subdomain.xS_r; x < subdomain.xE_r; x++)
    {
        for (y = subdomain.yS_r; y < subdomain.yE_r; y++)
        {
            for (z = subdomain.zS_r; z < subdomain.zE_r; z++)
            {
                index = x*subdomain.xStep + y*subdomain.yStep + z;

                PHASE_FILLED = 0;

                for (long a = 0; a < simDomain.numPhases-1; a++)
                {
                    if (phi[a*step+index] == 1.0)
                    {
                        for (long b = 0; b < simDomain.numComponents-1; b++)
                        {
                            comp[b*step + index] = ceq[a][a][b];
                        }

                        PHASE_FILLED = 1;
                        break;
                    }
                }

                if (!PHASE_FILLED)
                {
                    for (long b = 0; b < simDomain.numComponents-1; b++)
                        comp[b*step + index] = cfill[simDomain.numPhases-1][simDomain.numPhases-1][b];
                }
            }
        }
    }
}

void fillDomain(domainInfo simDomain, subdomainInfo subdomain,
                simParameters simParams, double *phi, double *comp,
                fillParameters *fill)
{
    // Creating pointers for every filling type
    // This enables dynamic creation of filling objects
    sphere *Sphere;
    cylinder *Cylinder;
    cube *Cube;
    ellipse *Ellipse;

    // Maximum number of random filling-generation attempts
    long numTrials = 1e5;

    for (long i = 0; i < fill->countFill; i++)
    {
        if (fill->fillType[i] == FILLCYLINDER)
        {
            Cylinder = (cylinder*)malloc(sizeof(cylinder));

            Cylinder->xC = fill->xC[i];
            Cylinder->yC = fill->yC[i];
            Cylinder->radius = fill->radius[i];
            Cylinder->phase = fill->phase[i];
            Cylinder->zS = fill->zS[i];
            Cylinder->zE = fill->zE[i];

            if (Cylinder->phase < simDomain.numPhases)
            {
                fillCylinder(phi, *Cylinder, simDomain, subdomain);
                Cylinder->phase = simDomain.numPhases-1;
                fillCylinder(phi, *Cylinder, simDomain, subdomain);
            }

            free(Cylinder);
        }

        else if (fill->fillType[i] == FILLSPHERE)
        {
            Sphere = (sphere*)malloc(sizeof(sphere));

            Sphere->xC = fill->xC[i];
            Sphere->yC = fill->yC[i];
            Sphere->zC = fill->zC[i];
            Sphere->radius = fill->radius[i];
            Sphere->phase = fill->phase[i];

            if (Sphere->phase < simDomain.numPhases)
            {
                fillSphere(phi, *Sphere, simDomain, subdomain);
                Sphere->phase = simDomain.numPhases-1;
                fillSphere(phi, *Sphere, simDomain, subdomain);
            }
            free(Sphere);
        }

        else if (fill->fillType[i] == FILLCUBE)
        {
            Cube = (cube*)malloc(sizeof(cube));

            Cube->xS = fill->xS[i];
            Cube->xE = fill->xE[i];
            Cube->yS = fill->yS[i];
            Cube->yE = fill->yE[i];
            Cube->zS = fill->zS[i];
            Cube->zE = fill->zE[i];
            Cube->phase = fill->phase[i];

            if (Cube->phase < simDomain.numPhases)
            {
                fillCube(phi, *Cube, simDomain, subdomain);
                Cube->phase = simDomain.numPhases-1;
                fillCube(phi, *Cube, simDomain, subdomain);
            }
            free(Cube);
        }

        else if (fill->fillType[i] == FILLCYLINDERRANDOM)
        {
            double volParticle = (double)(fill->radius[i]*fill->radius[i])*M_PI;
            double volume      = (double)(simDomain.MESH_X*simDomain.MESH_Y);
            long numParticles   =  ceil(volume*fill->volFrac[i] / volParticle);

            simParams.SEED = clock();

            srand48(simParams.SEED);

            Cylinder = (cylinder*)malloc(sizeof(cylinder) * numParticles);

            long count = 0;

            for (long k = 0; k < numTrials; k++)
            {
                Cylinder[count].radius = (double)fill->radius[i]*(1.0 + (drand48() - 0.5)*fill->radVar[i]);

                long distP, distM;

                // Get x-center at an adequate distance from the domain edge
                // This is done to prevent particles from coming too close as a result of the periodic B.C.
                do
                {
                    Cylinder[count].xC = simDomain.MESH_X*drand48();
                    distP = Cylinder[count].xC + Cylinder[count].radius;
                    distM = Cylinder[count].xC - Cylinder[count].radius;
                } while (!((distP < simDomain.MESH_X-1 - (fill->shieldDist[i]/2)) && (distM > fill->shieldDist[i]/2)));

                // Get y-center
                do
                {
                    Cylinder[count].yC = simDomain.MESH_Y*drand48();
                    distP = Cylinder[count].yC + Cylinder[count].radius;
                    distM = Cylinder[count].yC - Cylinder[count].radius;
                } while (!((distP < simDomain.MESH_Y-1 - (fill->shieldDist[i]/2)) && (distM > fill->shieldDist[i]/2)));

                // Force cylinder to span the entire z-axis in the domain
                Cylinder[count].zS = 0;
                Cylinder[count].zE = simDomain.MESH_Z - 1;

                Cylinder[count].phase = fill->phase[i];

                long j = 0;
                long distance, minDist;

                // Checking overlap with previously filled particles
                while (j < count)
                {
                    distance = (Cylinder[count].xC - Cylinder[j].xC)*(Cylinder[count].xC - Cylinder[j].xC)
                    + (Cylinder[count].yC - Cylinder[j].yC)*(Cylinder[count].yC - Cylinder[j].yC);

                    minDist  = ((Cylinder[count].radius + Cylinder[j].radius)*(1.0 + 0.5*fill->shieldDist[i]))
                    * ((Cylinder[count].radius + Cylinder[j].radius)*(1.0 + 0.5*fill->shieldDist[i]));

                    if (distance < minDist)
                        break;

                    j++;
                }

                if (j == count)
                {
                    if (Cylinder[count].phase < simDomain.numPhases)
                    {
                        fillCylinder(phi, Cylinder[count], simDomain, subdomain);
                        Cylinder[count].phase = simDomain.numPhases-1;
                        fillCylinder(phi, Cylinder[count], simDomain, subdomain);
                    }
                    count++;
                }

                if (count >= numParticles)
                    break;
            }

            free(Cylinder);
        }

        else if (fill->fillType[i] == FILLSPHERERANDOM)
        {
            double volParticle = (double)(fill->radius[i]*fill->radius[i]*fill->radius[i])*4.0*M_PI/3.0;
            double volume      = (double)(simDomain.MESH_X*simDomain.MESH_Y*simDomain.MESH_Z);
            long numParticles   =  ceil(volume*fill->volFrac[i] / volParticle);

            simParams.SEED = clock();

            srand48(simParams.SEED);

            Sphere = (sphere*)malloc(sizeof(sphere) * numParticles);


            long count = 0;

            for (long k = 0; k < numTrials; k++)
            {
                Sphere[count].radius = (double)fill->radius[i]*(1.0 + (drand48() - 0.5)*fill->radVar[i]/100.0);

                long distP, distM;

                // Get x-center at an adequate distance from the domain edge
                // This is done to prevent particles from coming too close as a result of the periodic B.C.
                do
                {
                    Sphere[count].xC = simDomain.MESH_X*drand48();
                    distP = Sphere[count].xC + Sphere[count].radius;
                    distM = Sphere[count].xC - Sphere[count].radius;
                } while (!((distP < simDomain.MESH_X-1 - (fill->shieldDist[i]/2)) && (distM > fill->shieldDist[i]/2)));

                // Get y-center
                do
                {
                    Sphere[count].yC = simDomain.MESH_Y*drand48();
                    distP = Sphere[count].yC + Sphere[count].radius;
                    distM = Sphere[count].yC - Sphere[count].radius;
                } while (!((distP < simDomain.MESH_Y-1 - (fill->shieldDist[i]/2)) && (distM > fill->shieldDist[i]/2)));

                // Get z-center
                do
                {
                    Sphere[count].zC = simDomain.MESH_Z*drand48();
                    distP = Sphere[count].zC + Sphere[count].radius;
                    distM = Sphere[count].zC - Sphere[count].radius;
                } while (!((distP < simDomain.MESH_Z-1 - (fill->shieldDist[i]/2)) && (distM > fill->shieldDist[i]/2)));

                Sphere[count].phase = fill->phase[i];

                long j = 0;
                long distance, minDist;

                // Checking overlap with previously filled particles
                while (j < count)
                {
                    distance = (Sphere[count].xC - Sphere[j].xC)*(Sphere[count].xC - Sphere[j].xC)
                    + (Sphere[count].yC - Sphere[j].yC)*(Sphere[count].yC - Sphere[j].yC)
                    + (Sphere[count].zC - Sphere[j].zC)*(Sphere[count].zC - Sphere[j].zC);

                    minDist  = ((fill->shieldDist[i] + 1.0)*(Sphere[count].radius + Sphere[j].radius))
                    * ((fill->shieldDist[i] + 1.0)*(Sphere[count].radius + Sphere[j].radius));

                    if (distance < minDist)
                        break;

                    j++;
                }

                if (j == count)
                {
                    if (Sphere[count].phase < simDomain.numPhases)
                    {
                        fillSphere(phi, Sphere[count], simDomain, subdomain);
                        Sphere[count].phase = simDomain.numPhases-1;
                        fillSphere(phi, Sphere[count], simDomain, subdomain);
                    }
                    count++;
                }

                if (count >= numParticles)
                    break;
            }

            free(Sphere);
        }
        ////////////////////Pushker:Added functions
        else if (fill->fillType[i] == FILLCUBEPATTERN)
        {
            long xlo=0, ylo=0, zlo=0,sgn,gap,sx,sy,sz,gfrac;
            long xhi=1, yhi=1, zhi=1;
            long l,m,n;
            double sfrac;

            simParams.SEED = clock();

            srand48(simParams.SEED);

            // Deciding Number of Particles based on Shield Distance
            ldiv_t resx, resy, resz;

            resx = ldiv(simDomain.MESH_X, fill->xS[i] + fill->shieldDist[i]);
            resy = ldiv(simDomain.MESH_Y, fill->yS[i] + fill->shieldDist[i]);
            resz = ldiv(simDomain.MESH_Z, fill->zS[i] + fill->shieldDist[i]);

            // Ensure at least one particle along each dimension
            if (resx.quot == 0) resx.quot = 1;
            if (resy.quot == 0) resy.quot = 1;
            if (resz.quot == 0) resz.quot = 1;

            long numParticles = resx.quot * resy.quot * resz.quot;

            printf("Filling %ld particles.\n", numParticles);

            Cube = (cube*)malloc(sizeof(cube));

            long count = 0;
            double r;
            gap = fill->shieldDist[i];
            gfrac = fill->shiftFrac[i];
            sfrac = fill->volFrac[i];
            sx = fill->xS[i];
            sy = fill->yS[i];
            sz = fill->zS[i];
            //Pushkar: In a pattern , filling cubes in the domain
            for (l=0; l<resx.quot; l++ )
            {
                if ( simDomain.MESH_X > 1 )
                {
                    r = drand48();
                    sgn = 2*lround(r) - 1;
                    xlo = l*(gap+sx) + gap + sgn*gfrac*gap - (2*r-1)*sfrac*sx;
                    xhi = (l+1) * (gap+sx) + sgn*gfrac*gap + (2*r-1)*sfrac*sx;
                }
                for ( m=0; m<resy.quot; m++ )
                {
                    if ( simDomain.MESH_Y > 1 )
                    {
                        r = drand48();
                        sgn = 2*lround(r) - 1;
                        ylo = m*(gap+sy) + gap + sgn*gfrac*gap - (2*r-1)*sfrac*sy;
                        yhi = (m+1) * (gap+sy) + sgn*gfrac*gap + (2*r-1)*sfrac*sy;
                    }
                    for (n=0; n<resz.quot; n++ )
                    {
                        if ( simDomain.MESH_Z > 1 )
                        {
                            r = drand48();
                            sgn = 2*lround(r) - 1;
                            zlo = n*(gap+sz) + gap + sgn*gfrac*gap - (2*r-1)*sfrac*sz;
                            zhi = (n+1) * (gap+sz) + sgn*gfrac*gap + (2*r-1)*sfrac*sz;
                        }
                        Cube->xS = xlo;
                        Cube->xE = xhi;
                        Cube->yS = ylo;
                        Cube->yE = yhi;
                        Cube->zS = zlo;
                        Cube->zE = zhi;
                        long randPhase = (int)(drand48() * simDomain.numPhases);
                        Cube->phase = randPhase;//Pushkar:Also we are filling cube pattern with random phases 
                        if (Cube->phase < simDomain.numPhases){
                            fillCube(phi, *Cube, simDomain, subdomain);
                            Cube->phase = simDomain.numPhases-1;
                            fillCube(phi, *Cube, simDomain, subdomain);
                        }
                    }
                }
            }
            free(Cube);
        }
        /////////////////////////////////////////////////////////////////
        /*else if (fill->fillType[i] == FILLVORONOI2D)
        {
            printf("[Rank %ld] Starting FILLVORONOI2D\n", subdomain.rank);
        
            long phaseStep = subdomain.numCompCells;
            long x, y, gid;
            int k, location, s;
            double *n = NULL, *m = NULL, *l = NULL, minimum, size_min = fill->volFrac[i];
            long *phase = NULL;
        
            long globalNx = simDomain.MESH_X;
            long globalNy = simDomain.MESH_Y;
            long global_NUMPOINTS = fill->shieldDist[i];
        
            long base = (global_NUMPOINTS / subdomain.size) * subdomain.rank +
                        (subdomain.rank < global_NUMPOINTS % subdomain.size ? subdomain.rank : global_NUMPOINTS % subdomain.size);
            long local_NUMPOINTS = global_NUMPOINTS / subdomain.size + (subdomain.rank < global_NUMPOINTS % subdomain.size ? 1 : 0);
            long limit = base + local_NUMPOINTS;
        
            printf("[Rank %ld] global_NUMPOINTS = %ld, base = %ld, local = %ld, limit = %ld\n",
                   subdomain.rank, global_NUMPOINTS, base, local_NUMPOINTS, limit);
        
            // Allocations
            n = (double *)calloc(global_NUMPOINTS, sizeof(double));
            m = (double *)calloc(global_NUMPOINTS, sizeof(double));
            l = (double *)calloc(global_NUMPOINTS, sizeof(double));
            phase = (long *)calloc(global_NUMPOINTS, sizeof(long));
        
            if (!n || !m || !l || !phase) {
                printf("[Rank %ld] Memory allocation failed.\n", subdomain.rank);
                MPI_Abort(MPI_COMM_WORLD, -1);
            }
        
            printf("[Rank %ld] Memory allocated.\n", subdomain.rank);
        
            // Initialize this rank's share of Voronoi points
            for (long g_idx = base; g_idx < limit; g_idx++) {
                n[g_idx] = drand48() * globalNx;
                m[g_idx] = drand48() * globalNy;
                phase[g_idx] = lrand48() % (simDomain.numPhases - 1);
            }
        
            printf("[Rank %ld] Seed points initialized.\n", subdomain.rank);
        
            for (x = subdomain.xS; x < subdomain.xE; x++) {
                for (y = subdomain.yS; y < subdomain.yE; y++) {
                    gid = (x - subdomain.xS + subdomain.xS_r - subdomain.xS) * subdomain.shiftPointer +
                          (y - subdomain.yS + subdomain.yS_r - subdomain.yS);
        
                    for (k = 0; k < global_NUMPOINTS; k++)
                        l[k] = (n[k] - x) * (n[k] - x) + (m[k] - y) * (m[k] - y);
        
                    minimum = l[0];
                    location = phase[0];
                    for (k = 1; k < global_NUMPOINTS; k++) {
                        if (l[k] < minimum) {
                            minimum = l[k];
                            location = phase[k];
                        }
                    }
        
                    for (s = 0; s < simDomain.numPhases - 1; s++)
                        phi[s * phaseStep + gid] = (location == s) ? 1.0 : 0.0;
                }
            }
        
            printf("[Rank %ld] Voronoi fill complete.\n", subdomain.rank);
        
            Cube = (cube*)malloc(sizeof(cube));
            if (!Cube) {
                printf("[Rank %ld] Cube malloc failed.\n", subdomain.rank);
                MPI_Abort(MPI_COMM_WORLD, -1);
            }
        
            Cube->xS = fill->xS[i];
            Cube->xE = fill->xE[i];
            Cube->yS = fill->yS[i];
            Cube->yE = fill->yE[i];
            Cube->phase = simDomain.numPhases - 1;
        
            fillCube(phi, *Cube, simDomain, subdomain);
            
            cube Cube_local;
            Cube_local.xS = fill->xS[i];
            Cube_local.xE = fill->xE[i];
            Cube_local.yS = fill->yS[i];
            Cube_local.yE = fill->yE[i];
            Cube_local.zS = subdomain.zS;
            Cube_local.zE = subdomain.zE;
            Cube_local.phase = simDomain.numPhases - 1;

            fillCube(phi, Cube_local, simDomain, subdomain);

            printf("[Rank %ld] Local cube fill done.\n", subdomain.rank);
        
            printf("[Rank %ld] Freeing arrays...\n", subdomain.rank);
            free(n);
            free(m);
            free(l);
            free(phase);
            printf("[Rank %ld] All arrays freed.\n", subdomain.rank);
        }
        */
        
        
        
        else if (fill->fillType[i] == FILLVORONOI2D)
        {
            long phaseStep = subdomain.numCompCells;
            long x, y, gid, gid1;
            long limit_x, limit_y;
            int k, location, s;
            int *FLAG;
            double *n, *m,*l, minimum,size_min = fill->volFrac[i];
            long *phase,NUMPOINTS_VORONOI = fill->shieldDist[i];
            long xG, yG;     // Global coordinates

            Cube = (cube*)malloc(sizeof(cube));
            Cube->xS = fill->xS[i];
            Cube->xE = fill->xE[i];
            Cube->yS = fill->yS[i];
            Cube->yE = fill->yE[i];
            Cube->phase = simDomain.numPhases-1;

            FLAG  = (int*)calloc(subdomain.numCompCells,sizeof(int));
            n     = (double *)calloc(NUMPOINTS_VORONOI,sizeof(double));
            m     = (double *)calloc(NUMPOINTS_VORONOI,sizeof(double));
            l     = (double *)calloc(NUMPOINTS_VORONOI,sizeof(double));
            phase = (long *)calloc(NUMPOINTS_VORONOI,sizeof(long));
            long rand_x, rand_y;
            int PHASE_FILLED=0;
            limit_x =  subdomain.shiftPointer;
            limit_y =  subdomain.shiftPointer;
            //printf("xE_c-xS_c=%ld,yE_c-yS_c=%ld\n",(subdomain.xE_c-subdomain.xS_c),(subdomain.yE_c-subdomain.yS_c));
            //printf("limit_x=%ld,limit_y=%ld,size=%ld\n",limit_x,limit_y,subdomain.size);
            unsigned int dummy = 0,dummy1=0,dummy3=0,dummy4 = 0;
            for(k=0;k<NUMPOINTS_VORONOI;k++){
                while (PHASE_FILLED!=1) {
                    //Randomly generate x and y coordinated in each of subdomain with local index limits
                    rand_x = (long)(subdomain.xS_r+drand48()*limit_x);
                    rand_y = (long)(subdomain.yS_r+drand48()*limit_y);
                    //gid = (subdomain.xS + rand_x - subdomain.xS_r)*subdomain.shiftPointer + (subdomain.yS + rand_y - subdomain.yS_r);//Pushkar:Global ID in each subdomain
                    gid = rand_x*subdomain.shiftPointer + rand_y;//Pushkar:Local ID in each subdomain
                    //printf("rand_x=%ld,rand_y=%ld\n",rand_x,rand_y);
                    //printf("gid=%ld,\n",gid);
                    //printf("gid1=%ld,\n",lid);
                    if (FLAG[gid]!=1) {
                        //Global voronoi centres
                        //n[k] = (double)(subdomain.xS + rand_x - subdomain.xS_r);
                        //m[k] = (double)(subdomain.yS + rand_y - subdomain.yS_r);
                        n[k] = (double)rand_x;
                        m[k] = (double)rand_y;
                        //printf("n=%f,m=%f\n",n[k],m[k]);
                        //printf("xS=%ld,yS=%ld\n",subdomain.xS,subdomain.yS);
                        for(x = subdomain.xS_r;x <= subdomain.xE_r;x++) {
                            for (y = subdomain.yS_r; y <= subdomain.yE_r; y++) {
                                    // Computing global coordinates
                                    xG = subdomain.xS + x - subdomain.xS_r;
                                    yG = subdomain.yS + y - subdomain.yS_r;
                                    if(dummy3<1){//As expected we get 2 different coordinates in 2 subdomains with NPROCS=2
                                        printf("xG=%ld,yG=%ld\n",xG,yG);
                                        dummy3++;     
                                    }
                                    if ((m[k]-(double)yG)*(m[k]-(double)yG) + (n[k]-(double)xG)*(n[k]-(double)xG) <= size_min*size_min) {
                                        //gid1 =(subdomain.xS + rand_x - subdomain.xS_r)*subdomain.shiftPointer + (subdomain.yS + rand_y - subdomain.yS_r);
                                        gid1 = rand_x*subdomain.shiftPointer + rand_y;//Pushkar:Local ID in each subdomain
                                        FLAG[gid1] = 1;
                                    }
                                }
                            }
                        PHASE_FILLED=1;
                    }
                }
                PHASE_FILLED=0;
            }
            //printf("ShiftPointer=%ld\n",subdomain.shiftPointer);
            //printf("nCompCells=%ld\n",subdomain.numCompCells);
            for(unsigned int z=0 ; z< subdomain.numCompCells;z++){
                if(FLAG[z]==1)
                    dummy4++;
            }
            printf("Flag_count=%u\n",dummy4);
            printf("rank=%ld,numCompCells=%ld\n",subdomain.rank,subdomain.numCompCells);
            for(k=0;k<NUMPOINTS_VORONOI;k++) {
                phase[k] = lrand48()%(simDomain.numPhases-1);//06072025:Instead of all phases I need to select phase from a list of phase indices
                //printf("phase[k]=%ld\n",phase[k]);
            }
            for (x = subdomain.xS_r; x <= subdomain.xE_r; x++)
            {
                for (y = subdomain.yS_r; y <= subdomain.yE_r; y++)
                {
                    //gid = (subdomain.xS + rand_x - subdomain.xS_r)*subdomain.shiftPointer + (subdomain.yS + rand_y - subdomain.yS_r);//Pushkar:Local ID in each subdomain
                    // Computing global coordinates
                    xG = subdomain.xS + x - subdomain.xS_r;
                    yG = subdomain.yS + y - subdomain.yS_r;
                    gid = xG*subdomain.shiftPointer + yG;//Pushkar:Local ID in each subdomain
                    // Computing global coordinates
                    //xG = subdomain.xS + x - subdomain.xS_r;
                    //yG = subdomain.yS + y - subdomain.yS_r;
                    for(k=0; k<NUMPOINTS_VORONOI; k++) {
                        l[k] = (n[k]-(double)xG)*(n[k]-(double)xG) + (m[k]-(double)yG)*(m[k]-(double)yG);
                    }
                    minimum  = l[0];
                    location = 0;
                    for(k=0;k<NUMPOINTS_VORONOI;k++) {
                        if(l[k]<minimum) {
                            minimum  = l[k];
                            location = phase[k];
                        }
                    }

                    for (s = 0; s < simDomain.numPhases-1; s++)
                    {
                        if (location == s){
                            phi[s*phaseStep + gid] = 1.0;
                        }
                        else
                            phi[s*phaseStep + gid] = 0.0;
                    }
                }
            }
            fillCube(phi, *Cube, simDomain, subdomain);
            //Ensuring Boundary Layer is not initialized with matrix phase un
            free(Cube);
            free(FLAG);
            free(n);
            free(m);
            free(l);
            free(phase);
        }
        
        /////////////////////////////////////////////////////////////////////////////
        else if (fill->fillType[i] == FILLCYLINDERRANDOMNEXLP)
        {
            double volParticle = (double)(fill->radius[i]*fill->radius[i])*M_PI;
            double volume      = (double)(simDomain.MESH_X*simDomain.MESH_Y);
            long numParticles   =  ceil((double)volume*fill->volFrac[i] / volParticle);
            printf("volParticle=%f,volume=%f,volfrac=%f\n",volParticle,volume,fill->volFrac[i]);
            printf("numParticles=%ld\n",numParticles);
            simParams.SEED = clock();

            srand48(simParams.SEED);

            Cylinder = (cylinder*)malloc(sizeof(cylinder) * numParticles);
            long count = 0;

            for (long k = 0; k < numTrials; k++)
            {
                Cylinder[count].radius = (double)fill->radius[i]*(1.0 + (drand48() - 0.5)*fill->radVar[i]);

                long distP, distM;

                // Get x-center at an adequate distance from the domain edge
                // This is done to prevent particles from coming too close as a result of the periodic B.C.
                do
                {
                    Cylinder[count].xC = simDomain.MESH_X*drand48();
                    distP = Cylinder[count].xC + Cylinder[count].radius;
                    distM = Cylinder[count].xC - Cylinder[count].radius;
                } while (!((distP < simDomain.MESH_X-1 - (fill->shieldDist[i]/2)) && (distM > fill->shieldDist[i]/2)));

                // Get y-center
                do
                {
                    Cylinder[count].yC = simDomain.MESH_Y*drand48();
                    distP = Cylinder[count].yC + Cylinder[count].radius;
                    distM = Cylinder[count].yC - Cylinder[count].radius;
                } while (!((distP < simDomain.MESH_Y-1 - (fill->shieldDist[i]/2)) && (distM > fill->shieldDist[i]/2)));

                // Force cylinder to span the entire z-axis in the domain
                Cylinder[count].zS = 0;
                Cylinder[count].zE = simDomain.MESH_Z - 1;

                Cylinder[count].phase = fill->phase[i];

                long j = 0;
                long distance, minDist;

                // Checking overlap with previously filled particles
                while (j < count)
                {
                    distance = (Cylinder[count].xC - Cylinder[j].xC)*(Cylinder[count].xC - Cylinder[j].xC)
                    + (Cylinder[count].yC - Cylinder[j].yC)*(Cylinder[count].yC - Cylinder[j].yC);

                    minDist  = ((Cylinder[count].radius + Cylinder[j].radius)*(1.0 + 0.5*fill->shieldDist[i]))
                    * ((Cylinder[count].radius + Cylinder[j].radius)*(1.0 + 0.5*fill->shieldDist[i]));

                    if (distance < minDist)
                        break;

                    j++;
                }

                if (j == count)
                {
                    if (Cylinder[count].phase < simDomain.numPhases)
                    {
                        fillCylinderNEXLP(phi, Cylinder[count], simDomain, subdomain);
                        Cylinder[count].phase = simDomain.numPhases-1;
                        fillCylinderNEXLP(phi, Cylinder[count], simDomain, subdomain);
                    }
                    count++;
                }

                if (count >= numParticles)
                    break;
            }

            free(Cylinder);
        }
        /////////////////////////////////////////////////////////////////////////////////////////////
        else if (fill->fillType[i] == FILLCYLINDERNEXLP)
        {
            Cylinder = (cylinder*)malloc(sizeof(cylinder));

            Cylinder->xC = fill->xC[i];
            Cylinder->yC = fill->yC[i];
            Cylinder->radius = fill->radius[i];
            Cylinder->phase = fill->phase[i];
            Cylinder->zS = fill->zS[i];
            Cylinder->zE = fill->zE[i];

            if (Cylinder->phase < simDomain.numPhases)
            {
                fillCylinderNEXLP(phi, *Cylinder, simDomain, subdomain);
                Cylinder->phase = simDomain.numPhases-1;
                fillCylinderNEXLP(phi, *Cylinder, simDomain, subdomain);
            }

            free(Cylinder);
        }
        //////////////////////////////////////////////////////////////////////////
        else if (fill->fillType[i] == FILLSPHERENEXLP)
        {
            Sphere = (sphere*)malloc(sizeof(sphere));

            Sphere->xC = fill->xC[i];
            Sphere->yC = fill->yC[i];
            Sphere->zC = fill->zC[i];
            Sphere->radius = fill->radius[i];
            Sphere->phase = fill->phase[i];

            if (Sphere->phase < simDomain.numPhases)
            {
                fillSphereNEXLP(phi, *Sphere, simDomain, subdomain);
                Sphere->phase = simDomain.numPhases-1;
                fillSphereNEXLP(phi, *Sphere, simDomain, subdomain);
            }
            free(Sphere);
        }
        ////////////////////////////////////////////////////////////////////////////
        else if (fill->fillType[i] == FILLCUBERANDOM)
        {
            long gap = fill->shieldDist[i];
            long sx = fill->xS[i];
            long sy = fill->yS[i];
            long sz = fill->zS[i];
            double vf = fill->volFrac[i] / 100.0;
            long phaseStep = subdomain.numCompCells;

            double vol_particle = (double)(sx * sy * sz);
            double total_volume = (double)(simDomain.MESH_X * simDomain.MESH_Y * simDomain.MESH_Z);
            long total_particles = (long)ceil((vf * total_volume) / vol_particle);

            long num_particles = total_particles / subdomain.size;
            if (subdomain.rank < total_particles % subdomain.size)
                num_particles += 1;

            printf("[Rank %d] Intended to fill %ld cubes (global total = %ld)\n",
                subdomain.rank, num_particles, total_particles);

            simParams.SEED = clock() + subdomain.rank * 7919;
            srand48(simParams.SEED);

            cube *Cube = (cube *)malloc(sizeof(cube) * num_particles);
            long count = 0;
            long maxTrials = 100 * num_particles;

            for (long trial = 0; trial < maxTrials && count < num_particles; ++trial)
            {
                //Globally selecting random Cube centres
                long xC = subdomain.xS + sx / 2 + (long)(drand48() * (subdomain.xE - subdomain.xS - sx));
                long yC = subdomain.yS + sy / 2 + (long)(drand48() * (subdomain.yE - subdomain.yS - sy));
                long zC = subdomain.zS + sz / 2 + (long)(drand48() * (subdomain.zE - subdomain.zS - sz));
                //printf("Trial:%ld,xC:%ld,yC:%ld,zC:%ld\n",trial,xC,yC,zC);

                //Globally marking edge positions of cube
                long xlo = xC - sx / 2;
                long xhi = xlo + sx;
                long ylo = yC - sy / 2;
                long yhi = ylo + sy;
                long zlo = zC - sz / 2;
                long zhi = zlo + sz;
               
                // Assign cube
                Cube[count].xS = xlo;
                Cube[count].xE = xhi;
                Cube[count].yS = ylo;
                Cube[count].yE = yhi;
                Cube[count].zS = zlo;
                Cube[count].zE = zhi;
                Cube[count].phase = (int)(drand48() * (simDomain.numPhases - 1));

                // Overlap check
                int overlap = 0;
                for (long j = 0; j < count; ++j)
                {
                    long x_overlap = !(xhi + gap <= Cube[j].xS || xlo - gap >= Cube[j].xE);
                    long y_overlap = !(yhi + gap <= Cube[j].yS || ylo - gap >= Cube[j].yE);
                    long z_overlap = !(zhi + gap <= Cube[j].zS || zlo - gap >= Cube[j].zE);
                    //printf("xoverlap:%ld,yoverlap:%ld,zoverlap:%ld\n",x_overlap,y_overlap,z_overlap);
                    if (x_overlap && y_overlap && z_overlap)
                    {
                        overlap = 1;
                        break;
                    }
                }
                if (overlap)
                    continue;

                fillCube(phi, Cube[count], simDomain, subdomain);

                Cube[count].phase = simDomain.numPhases - 1;
                fillCube(phi, Cube[count], simDomain, subdomain);

                count++;
            }
            printf("[Rank %d] Successfully placed %ld / %ld cubes\n", subdomain.rank, count, num_particles);
            free(Cube);
        }
        else
        {
            printf("Invalid filling parameters. Please check your filling file\n");
        }
    }

    fillComposition(phi, comp, simDomain, subdomain, simParams.ceq_host, simParams.cfill_host);

    // The following were allocated in fill_domain.c
    free(fill->fillType);
    free(fill->xC);
    free(fill->yC);
    free(fill->zC);
    free(fill->xS);
    free(fill->xE);
    free(fill->yS);
    free(fill->yE);
    free(fill->zS);
    free(fill->zE);
    free(fill->radius);
    free(fill->phase);
    free(fill->major_axis);
    free(fill->eccentricity);
    free(fill->rot_angle);
    free(fill->seed);
    free(fill->volFrac);
    free(fill->shieldDist);
    free(fill->shiftFrac);//Pushkar:Added 
    free(fill->radVar);
}
