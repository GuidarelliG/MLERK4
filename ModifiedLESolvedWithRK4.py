import math , csv
from matplotlib import pyplot
"""
4th Order Runge-Kutta for modified lane emdan equations:

This code uses shooting method to find the approriate parameters for
HSE profile
"""

#CONSTANTS
G = 6.67*(10**-8)   #Grav const in cgs
n = 3.0             #Polytropic index to be stable against convection n> 1/(gamma - 1)
                    #So for monatonic gas (gamma = 5/3) n > 3/2


#INPUTS: 
h = 4.2*(10**10)             #In cm, this should be at least 20 times smallest grid-cell
MESAprofile = "mesaprofiles.csv"   #Name of the file if it is in the same directory or the path if not.
Resolution = 1000           #How many points do you want to resolve h with
percenterr = 0.1            #The percent error allowed for the derivitive match, so 0.1 is 0.1% error

#________________________________________________________________
# RK4 functions start here:
#
# note that if you are not using astrobear spline softening function
# for the point particle, you have to change gc and dgc_dr.
#_________________________________________________________________


def gc(m_p, x, h):
    global G
    y = x/h
    if(x >= h):
        return G*m_p/(x**2)
    elif(h/2 <= x and x < h ):
        a = 64.0/3.0
        b = -48.0
        c = 192.0/5.0
        d = -32.0/3.0
        e = -2.0/30.0
        ans = G*m_p*(y*(a + y*(b + y*(c + d*y)))+ (e/(y**2)))/(h**2)
        return ans
    elif(h/2 > x):
        a = 32.0/3.0
        b = -192.0/5.0
        c = 32.0
        return G*m_p*(y*(a + (y**2)*(b+c*y)))/(h**2)

def dgc_dr(m_p, x, h):
    global G
    y = x/h
    if(x >= h):
        return -2.0*G*m_p/(x**3)
    elif(h/2 <= x and x < h ):
        a = 64.0/3.0
        b = -48.0
        c = 192.0/5.0
        d = -32.0/3.0
        e = -2.0/30.0
        const = G*m_p/((h**3)) #below is product rule expansion
        d1 = (a + y*(b + y*(c + d*y)))
        d2 = (b + y*(c + d*y))
        d3 = (c + d*y)
        d4 = d
        d5 = -2.0*e/(y**3)
        return ((d1 + y*(d2+ y*(d3 + y*d4)))+ d5)*const
    elif(h/2 > x):
        a = 32.0/3.0
        b = -192.0/5.0
        c = 32.0
        const = G*m_p/((h**3))
        d1 = (a + (y**2)*(b+c*y))
        d2 = (b+c*y)
        return (d1 + y*(2*y*d2 + c*(y**2)))*const
    
"""
The Lane emdan equations result in a second order DE and RK4
is used to numerically solve

for RK4 Code:

xi ->           t
theta ->        y
dtheta/dxi ->   z
dtheta/dxi 'function' ->   f
df/dxi ->       g
"""
def f(t,y,z,n,alpha,rho_c, m_p,h):
    return z

def g(t,y,z,n,alpha,rho_c, m_p,h):
    global G
    return (-2.0*z/t) - y**n - (1.0/(4.0*math.pi*G*rho_c))*(dgc_dr(m_p,alpha*t,h) + (2.0*gc(m_p,alpha*t,h)/(alpha*t)))


def RKstep(t,y,z,n,alpha,rho_c, m_p,h , dt):
    k0=dt*f(t,y,z,n,alpha,rho_c, m_p,h)
    l0=dt*g(t,y,z,n,alpha,rho_c, m_p,h)
    k1=dt*f(t+(1/2)*dt,y+(1/2)*k0,z+(1/2)*l0,n,alpha,rho_c, m_p,h)
    l1=dt*g(t+(1/2)*dt,y+(1/2)*k0,z+(1/2)*l0,n,alpha,rho_c, m_p,h)
    k2=dt*f(t+(1/2)*dt,y+(1/2)*k1,z+(1/2)*l1,n,alpha,rho_c, m_p,h)
    l2=dt*g(t+(1/2)*dt,y+(1/2)*k1,z+(1/2)*l1,n,alpha,rho_c, m_p,h)
    k3=dt*f(t+dt,y+k2,z+l2,n,alpha,rho_c, m_p,h)
    l3=dt*g(t+dt,y+k2,z+l2,n,alpha,rho_c, m_p,h)
    
    t_next = t + dt
    y_next = y + (1/6)*(k0+2*k1+2*k2+k3)
    z_next = z + (1/6)*(l0+2*l1+2*l2+l3)
    return t_next, y_next, z_next

#_______________________________________________________
#
#These functions are used to run RK4, get initial conditions,
#Iterate over solutions and save the data. 
#_______________________________________________________

def getBC(temp, h):
    """
    The boundry conditions:

    smoothing radius ->         h
    mass inside radius of h ->  m_h
    density at h ->             rho_h
    density der at h ->         drhodr_h
    """
    print("Getting Boundry Conditions")
    rad = []
    den = []
    per = []
    masses = []
    i = 0
    length = 0
    h_index = 0
    tot_mass = 0

    #This loop gets actual h, m_h and rho_h
    ifile  = open(temp, "rt")
    read = csv.reader(ifile, delimiter=' ', quotechar='|')
    for row in read:
        if(i == 0):
            length = int(row[0])-1
        if(i>0):
            rad = rad + [float(row[0])]
            den = den + [float(row[1])]
            per = per + [float(row[2])]
            if(rad[-1]<h):
                h_index = i-1
                if(i == 1):
                    tot_mass = tot_mass + (4.0/3.0)*math.pi*(rad[-1]**3)*den[-1]
                    masses = masses + [tot_mass]
                else: #I use trapizodal intergration on a sphere
                    slope = (den[-1]-den[-2])/(rad[-1]-rad[-2])
                    inter = den[-1] - (slope*rad[-1])
                    avg_den = (3.0/4.0)*slope*((rad[-1]**4)-(rad[-2]**4))/((rad[-1]**3)-(rad[-2]**3)) + inter
                    tot_mass = tot_mass + (4.0/3.0)*math.pi*((rad[-1]**3)-(rad[-2]**3))*avg_den
                    masses = masses + [tot_mass]
            else:
                slope = (den[-1]-den[-2])/(rad[-1]-rad[-2])
                inter = den[-1] - (slope*rad[-1])
                avg_den = (3.0/4.0)*slope*((rad[-1]**4)-(rad[-2]**4))/((rad[-1]**3)-(rad[-2]**3)) + inter
                masses = masses + [masses[-1] + (4.0/3.0)*math.pi*((rad[-1]**3)-(rad[-2]**3))*avg_den]
        i = i+1

    #this calculates drhodr_h

    drhodr_h = (den[h_index+2]-den[h_index-1])/(rad[h_index+2]-rad[h_index-1])
    h = rad[h_index] # this has to be the actual h as then I can merge the two exactly
    rho_h = den[h_index]

    return h, h_index, rho_h, drhodr_h, tot_mass, rad, den, per, masses



def findsol(n,h,alpha,rho_c,m_p,dt):
    global G
    ts = [dt]
    ys = [0.999]
    zs = [0.0]
    i = 0
    K = 4*math.pi*G*(alpha**2)/((1+n)*(rho_c**((1/n)-1)))
    rhos = [rho_c]
    rs = [ts[0]*alpha]
    pressures = [K*(rho_c**((1/n)+1))]
    mass = m_p*1.0
    intmasses = [mass]
    while(ys[-1] > 0 and ts[-1]*alpha< h):
        t_next , y_next, z_next = RKstep(ts[-1],ys[-1],zs[-1],n,alpha,rho_c, m_p,h , dt)
        ts = ts + [t_next]
        rs = rs + [t_next*alpha]
        ys = ys + [y_next]
        rhos = rhos + [rho_c*(y_next**n)]
        pressures = pressures + [K*(rhos[-1]**((1/n)+1))]
        zs = zs + [z_next]
        i = i+1
        if(i == 1):
            mass = mass + (4.0/3.0)*math.pi*(rs[-1]**3)*rhos[-1]
        else: #I use trapizodal intergration on a sphere
            slope = (rhos[-1]-rhos[-2])/(rs[-1]-rs[-2])
            inter = rhos[-1] - (slope*rs[-1])
            avg_den = (3.0/4.0)*slope*((rs[-1]**4)-(rs[-2]**4))/((rs[-1]**3)-(rs[-2]**3)) + inter
            mass = mass + (4.0/3.0)*math.pi*((rs[-1]**3)-(rs[-2]**3))*avg_den
        intmasses = intmasses + [mass]
    return rs, rhos, mass, pressures, intmasses




def shoot(n,h,alpha,rho_c,m_p,Resolution,rho_h,drhodr_h,tot_mass):
    a = 1.0 #convergence rate for rho_c
    b = 0.01 #convergence rate for alpha
    c = 0.25 #convergence rate for mass
    e1 = 1
    e2 = 1
    e3 = 1
    #while the error of any parameter is too large...
    j = 0
    while(e1**2+e2**2+e3**2>(percenterr/100)**2 and j<1000):
        dt = (h/alpha)/Resolution
        rs, rhos, mass, pressures,intmasses = findsol(n,h,alpha,rho_c,m_p,dt)
        #pyplot.loglog(rs,rhos)
        tempder = (rhos[-1]-rhos[-2])/(rs[-1]-rs[-2])
        e1 = (mass-tot_mass)/tot_mass 
        e2 = ((tempder - drhodr_h)/abs(drhodr_h))
        e3 = ((rhos[-1]- rho_h)/rho_h)
        rho_c = rho_c  - a*rho_h*e3
        alpha = alpha - b*alpha*((tempder - drhodr_h)/abs(drhodr_h))
        m_p = m_p - c*tot_mass*e1
        j = j + 1
        #print(alpha)
    return alpha, rho_c, m_p, rhos, rs, pressures, intmasses


def savedata(name, x,y,z):
    print("Saving Data")
    with open(name, "w", newline='') as f:
        writer = csv.writer(f, delimiter=" ")
        length = len(x)
        f.write(str(length)+'\n')
        for i in range(length):
            #a = "{:.5E}".format(x[i])
            #b = "{:.5E}".format(y[i])
            #c = "{:.5E}".format(z[i])
            s = [x[i],y[i],z[i]]
            writer.writerow(s)

def adjpressure(alpha, h, m_p,rs, rhos,intmasses, P0):
    global G
    Ps = [P0]
    for i in range(len(rs)-1):
        dr = rs[-(i+1)] - rs[-(i+2)]
        dp1 = (-1.0*G*(intmasses[-(i+1)]-m_p))*rhos[-(i+1)]*dr/(rs[-(i+1)]**2)
        dp2 = (-1.0*rhos[-(i+1)]*dr*gc(m_p, rs[-(i+1)], h))
        Ps = Ps + [Ps[-1]-dp1-dp2]
    Ps.reverse()
    return Ps
    
#__________________________________________________________
#
# This is the start of the code
#__________________________________________________________

"""
THESE PARAMETERS WORK
alpha = 3.8*(10**10)
rho_c = 14.0
m_p = 0.48*(10**33)
h = 5.0*(10**9)
n = 3.0
dt = (h/alpha)*0.001
"""

#Get the initial conditions at r = h and also read in the mesa profile
h, h_index, rho_h, drhodr_h, tot_mass, rad, den, per, masses = getBC(MESAprofile,h)

#initial guesses
rho_c = 15.0*rho_h
alpha = ((per[h_index]/(rho_h**((1/n)+1)))*((1+n)*(rho_c**((1/n)-1)))/(4*math.pi*G))**0.5
m_p  = tot_mass - (4.0/3.0)*math.pi*rho_c*(h**3)

print("Finding Optimal Solution")
alpha, rho_c, m_p, rhos, rs, pressures, intmasses = shoot(n,h,alpha,rho_c,m_p,Resolution,rho_h,drhodr_h,tot_mass)

#Added Plus One because dont want two of the same r values at h (screws up interpolating functions)
rs = rs+rad[h_index+1:]
rhos = rhos+den[h_index+1:]
intmasses = intmasses + masses[h_index+1:]

print("Adjusting Pressure")
pressures = adjpressure(alpha, h, m_p,rs, rhos,intmasses, per[-1])

savedata("amb.data",rs,rhos,pressures)
#pyplot.loglog(rad,per)
#pyplot.loglog(rs,pressures)
pyplot.loglog(rad,den)
pyplot.loglog(rs,rhos)

print("Done")
print("smoothing radius = ", str(h/(10.0**10)), "10^10 cm")
print("particle mass = ", str(m_p/(1.99e33)), "Solar masses")

pyplot.show()

    
    
