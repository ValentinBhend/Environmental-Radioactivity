# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 10:54:21 202

@author: vbhen
"""

import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

T_half_Rn222 = 3.823 * 24 * 3600
lambda_Rn222 = np.log(2) / T_half_Rn222

T_half_Po218 = 3.04 * 60
lambda_Po218 = np.log(2) / T_half_Po218

T_half_Pb214 = 26.9 * 60
lambda_Pb214 = np.log(2) / T_half_Pb214

T_half_Bi214 = 19.9 * 60
lambda_Bi214 = np.log(2) / T_half_Bi214

T_half_Po214 = 164.3 * 1e-6
lambda_Po214 = np.log(2) / T_half_Po214

T_half_Po214 = 164.3 * 1e-6
lambda_Po214 = np.log(2) / T_half_Po214

T_half_Rn220 = 55.6
lambda_Rn220 = np.log(2) / T_half_Rn220

T_half_Pb212 = 10.64 * 3600
lambda_Pb212 = np.log(2) / T_half_Pb212

T_half_Bi212 = 1.01 * 3600
lambda_Bi212 = np.log(2) / T_half_Bi212

T_half_Rn219 = 3.96
lambda_Rn219 = np.log(2) / T_half_Rn219

T_half_Pb211 = 36.1 * 60
lambda_Pb211 = np.log(2) / T_half_Pb211

T_half_Bi211 = 2.14 * 60
lambda_Bi211 = np.log(2) / T_half_Bi211


def from_txt(filename):
    counts = np.zeros(0)
    first = True
    with open(filename, newline = '') as data:                                                                                          
        data_reader = csv.reader(data, delimiter='\t')
        for data in data_reader:
            if(first == True):
                first = False
            else:
                counts = np.append(counts, int(data[1]))
    return counts

def binning(counts, binsize):
    bincount = int(len(counts)/binsize)
    counts = counts[0:bincount*binsize]
    output = np.zeros(bincount)
    for i in range(bincount):
        s = 0
        for j in range(binsize):
            s += counts[i*binsize+j]
        output[i] = s / binsize
    return output

def chart(binned, binsize, title):
    t = np.zeros(len(binned))
    for i in range(len(binned)):
        t[i] = i*binsize
    plt.plot(t,binned)
    plt.title(title)
    plt.show()
    
def deviation(data):
    avg = np.average(data)
    s = 0
    for d in data:
        s += (d - avg)**2
        s = s / (len(data) - 1)
    return np.sqrt(s)

def chain_3(t, l1, l2, l3, N0):
    #a1 = N0*l1 * np.exp(-l1*t)
    a2 = N0*l2*l1 * (np.exp(-l1*t) - np.exp(-l2*t))/(l2-l1)
    a3 = N0*l1*l2*l3 * ( np.exp(-l1*t)/((l2-l1)*(l3-l1)) + np.exp(-l2*t)/((l1-l2)*(l3-l2)) + np.exp(-l3*t)/((l1-l3)*(l2-l3)))
    return a2+a3

def chain_2(t, N0, l1, l2):
    a1 = N0*l1 * np.exp(-l1*t)
    a2 = N0*l2*l1 * (np.exp(-l1*t) - np.exp(-l2*t))/(l2-l1)
    return a1+a2

def activity_Pb211(t, N211):
    return N211 * lambda_Pb211 * np.exp(-lambda_Pb211*t)

def decay_fit(t, N222, l1):
    return N222 * l1 * np.exp(-l1*t)

def d_bg(N,t):
    return np.sqrt(N)/t

def d_A(N,t):
    bg = background
    dbg = d_background
    return np.sqrt( (k*np.sqrt(N)/t)**2 + (k*dbg)**2 ) # + ((N/t-bg)*dk)**2


#################   Background measurement   ##################
background_counts = from_txt('background.txt')
background = np.average(background_counts)
t_background = np.linspace(0,len(background_counts), num=len(background_counts))

t_background_10sec = np.linspace(0,len(background_counts), num=int(len(background_counts)/10))
background_bin10sec = binning(background_counts, 10)
n_b10s = len(background_bin10sec)

dbackground_bin10sec = np.zeros(n_b10s)
for i in range(n_b10s):
    #dbackground_bin10sec[i] = deviation(background_counts[i*10:(i+1)*10])
    dbackground_bin10sec[i] = d_bg(sum(background_counts[i*10:(i+1)*10]), 10)

t_background_1min = np.linspace(0,len(background_counts), num=int(len(background_counts)/60))
background_bin1min = binning(background_counts, 60)
n_b1m = len(background_bin1min)
dbackground_bin1m = np.zeros(n_b1m)
for i in range(n_b1m):
    #dbackground_bin1m[i] = deviation(background_counts[i*60:(i+1)*60])
    dbackground_bin1m[i] = d_bg(sum(background_counts[i*60:(i+1)*60]), 60)

t_background_10min = np.linspace(0,len(background_counts), num=int(len(background_counts)/600))
background_bin10min = binning(background_counts, 600)
n_b10m = len(background_bin10min)
dbackground_bin10min = np.zeros(n_b10m)
for i in range(n_b10m):
    #dbackground_bin10min[i] = deviation(background_counts[i*10*60:(i+1)*10*60])
    dbackground_bin10min[i] = d_bg(sum(background_counts[i*10*60:(i+1)*10*60]), 10*60)

#d_background = deviation(background_counts)
d_background = d_bg(sum(background_counts), len(background_counts))
print("measured background: " + str(background) + " +/- " + str(d_background))
print("background measurement time (h): " + str(len(background_counts)/3600))

plt.scatter(t_background_10sec/3600, background_bin10sec, s=0.2, color = 'k', label = "10 s")
plt.errorbar(t_background_1min/3600, background_bin1min, yerr=dbackground_bin1m,  fmt="none", color='b', alpha=0.3, capsize=2)
plt.scatter(t_background_1min/3600, background_bin1min, s=10, color = 'b', label = "1 min")
plt.errorbar(t_background_10min/3600, background_bin10min, yerr=dbackground_bin10min,  fmt="none", color='r', alpha=0.5, capsize=3, elinewidth=2)
plt.scatter(t_background_10min/3600, background_bin10min, s=40, color = 'r', label = "10 min")
plt.ylim(0,1.5)
plt.title("background measurement with different binning times")
plt.xlabel("time (h)")
plt.ylabel("activity (cps)")
plt.legend()
plt.savefig('background.pdf', format='pdf')
plt.show()
###############################################################


#################   Efficiency calibration   ##################
kcl_counts = from_txt('kcl.txt') - background
kcl = np.average(kcl_counts)
#dkcl_avg = deviation(kcl_counts)
dkcl_avg = d_bg(sum(kcl_counts), len(kcl_counts))

k40_portion = 0.000117 # manual
T_half_40k = 1.248 * 1e9 * 3600 * 24 * 365.25 # https://www.cns-snc.ca/media/uploads/teachers/K40_4pg_10_06.pdf
mol_mass_kcl = 74.55 * 1e-3 # https://pubchem.ncbi.nlm.nih.gov/compound/4873
lambda_40k = np.log(2) / T_half_40k
m_kcl = 1.9 * 1e-3
dm_kcl = 0.1 * 1e-3
mol = 6.02214076 * 1e23
N_kcl = mol * m_kcl / mol_mass_kcl
dN_kcl = mol * dm_kcl / mol_mass_kcl
N_40k = k40_portion * N_kcl
dN_40k = k40_portion * dN_kcl
A_kcl = N_40k * lambda_40k
dA_kcl = dN_40k * lambda_40k

k = A_kcl / kcl
dk = np.sqrt( (dA_kcl/(kcl-background))**2 + (A_kcl*dkcl_avg/(kcl-background)**2)**2 + (A_kcl*d_background/(kcl-background)**2)**2 )
print("efficiency: " + str(k) + " +/- " + str(dk))
print("efficiency measurement time (h): " + str(len(kcl_counts)/3600))

kcl_bin5min = binning(kcl_counts, 5*60)
t_kcl = np.linspace(0,len(kcl_counts), num = int(len(kcl_counts)/(5*60)))

n_kcl = len(kcl_bin5min)
dkcl = np.zeros(n_kcl)
for i in range(n_kcl):
    dkcl[i] = d_bg(sum(kcl_counts[i*5*60:(i+1)*5*60]), 5*60)

plt.errorbar(t_kcl/60, kcl_bin5min, yerr=dkcl, fmt="none", color='b', alpha=0.5, capsize=6, elinewidth=2)
plt.scatter(t_kcl/60, kcl_bin5min, s=30)
plt.title("measured activity of KCl (bintime: 5min)")
plt.xlabel("time (min)")
plt.ylabel("activity (cps)")
#plt.ylim(0,4.5)
plt.grid()
plt.savefig('kcl.pdf', format='pdf')
plt.show()
###############################################################


######################   Air measurement   #####################
air_counts_raw = from_txt('air.txt')
air_counts = (air_counts_raw - background) * k
air_bin5min = binning(air_counts, 60*5)
air_bin5min = air_bin5min[0:12*25]
n_air_5min = len(air_bin5min)
dair_bin5min = np.zeros(n_air_5min)


for i in range(n_air_5min):
    #dair_bin5min[i] = deviation(air_counts_raw[i*60*5:(i+1)*60*5])
    #dair_bin5min[i] = np.sqrt( (dair_bin5min[i]*k)**2 + (d_background*k)**2 )
    dair_bin5min[i] = d_A(sum(air_counts_raw[i*60*5:(i+1)*60*5]), 60*5)


print("air first 5min")
a0 = air_bin5min[0]
da0 = dair_bin5min[0]

vol = 25.39
dvol = 0.002

dose = (a0/vol) * 3.77*1e-9 * 2250 * 1000000
ddose = 3.77*1e-9 * 2250 * np.sqrt( (da0/vol)**2 + (a0*dvol / (vol**2))**2 ) *1000000
print("activity first 5min: " + str(a0) + " +/- " + str(da0))
print("dose first 5min: " + str(dose) + " +/- " + str(ddose))


def air_fit(t, N1, l1, N2, l2):
    a1 = decay_fit(t, N1, l1)
    a2 = decay_fit(t, N2, l2)
    return a1+a2

n = len(air_bin5min)
t_air_5min_25h = np.linspace(0, n*60*5, num=n)

popt_a, pcov_a = curve_fit(air_fit, t_air_5min_25h, air_bin5min, p0=[1e5, lambda_Rn222, 1e5, lambda_Pb212], maxfev=1000000)

N1 = popt_a[0]
l1 = popt_a[1]
N2 = popt_a[2]
l2 = popt_a[3]
T1 = np.log(2)/l1
T2 = np.log(2)/l2

dN1 = np.sqrt(np.diag(pcov_a))[0]
dl1 = np.sqrt(np.diag(pcov_a))[1]
dN2 = np.sqrt(np.diag(pcov_a))[2]
dl2 = np.sqrt(np.diag(pcov_a))[3]
dT1 = np.log(2) * dl1 / l1**2
dT2 = np.log(2) * dl2 / l2**2

print("-----------   air fit 1:   -----------" )
print("N1 = " + str(N1) + " +/- " + str(dN1))
print("lamb1 = " + str(l1) + " +/- " + str(dl1))
print("T 1 (d) = " + str(T1 / (3600*24)) + " +/- " + str(dT1/(3600*24)))
print("N2 = " + str(N2) + " +/- " + str(dN2))
print("lamb2 = " + str(l2) + " +/- " + str(dl2))
print("T 2 (min) = " + str(T2 / 60) + " +/- " + str(dT2/60))


plt.errorbar(t_air_5min_25h/3600, air_bin5min, yerr=dair_bin5min,  fmt="none", color='b', alpha=0.3, capsize=2)
plt.semilogy(t_air_5min_25h/3600, air_fit(t_air_5min_25h, N1, l1, N2, l2), label='fit', color='r')
plt.scatter(t_air_5min_25h/3600, air_bin5min, label='data', s=6)
plt.legend()
plt.title("Environmental air measurement")
plt.xlabel("time (h)")
plt.ylabel("activity (cps)")
plt.ylim(0.1)
plt.grid()
plt.savefig('air11.pdf', format='pdf')
plt.show()

## test
popt_at, pcov_at = curve_fit(chain_2, t_air_5min_25h, air_bin5min, p0=[1e5, lambda_Pb212, lambda_Rn222], maxfev=1000000)

N_1 = popt_at[0]
l_1 = popt_at[1]
l_2 = popt_at[2]
T_1 = np.log(2)/l_1
T_2 = np.log(2)/l_2

dN_1 = np.sqrt(np.diag(pcov_at))[0]
dl_1 = np.sqrt(np.diag(pcov_at))[1]
dl_2 = np.sqrt(np.diag(pcov_at))[2]
dT_1 = np.log(2) * dl1 / l_1**2
dT_2 = np.log(2) * dl2 / l_2**2

print("-----------   air fit test:   -----------" )
print("N1 = " + str(N_1) + " +/- " + str(dN_1))
print("lamb1 = " + str(l_1) + " +/- " + str(dl_1))
print("T 1 (min) = " + str(T_1 / 60) + " +/- " + str(dT_1/60))
print("lamb2 = " + str(l_2) + " +/- " + str(dl_2))
print("T 2 (h) = " + str(T_2 / 3600) + " +/- " + str(dT_2/3600))

plt.errorbar(t_air_5min_25h/3600, air_bin5min, yerr=dair_bin5min,  fmt="none", color='b', alpha=0.3, capsize=2)
plt.semilogy(t_air_5min_25h/3600, chain_2(t_air_5min_25h, N_1, l_1, l_2), label='fit', color='r')
plt.scatter(t_air_5min_25h/3600, air_bin5min, label='data', s=6)
plt.legend()
plt.title("Environmental air measurement test")
plt.xlabel("time (h)")
plt.ylabel("activity (cps)")
plt.ylim(0.1)
plt.grid()
plt.savefig('air12.pdf', format='pdf')
plt.show()




## -----------------------------
air_bin5min_t1 = air_bin5min[0:12*2]
air_bin5min_t2 = air_bin5min[12*6:12*25]

n1 = len(air_bin5min_t1)
t_air_5min_t1 = np.linspace(0,n1*5*60, num=n1)
n2 = len(air_bin5min_t2)
t_air_5min_t2 = np.linspace(6*3600,6*3600+n2*5*60, num=n2)

n3 = int(len(t_air_5min_25h)*0.28)
t_air_5min_5h = t_air_5min_25h[0:n3]

popt_a1, pcov_a1 = curve_fit(decay_fit, t_air_5min_t1, air_bin5min_t1, p0=[1e5, lambda_Pb211], maxfev=10000)
popt_a2, pcov_a2 = curve_fit(decay_fit, t_air_5min_t2, air_bin5min_t2, p0=[1e5, lambda_Rn222], maxfev=10000)

N11 = popt_a1[0]
l11 = popt_a1[1]
N21 = popt_a2[0]
l21 = popt_a2[1]
T11 = np.log(2)/l11
T21 = np.log(2)/l21

dN11 = np.sqrt(np.diag(pcov_a1))[0]
dl11 = np.sqrt(np.diag(pcov_a1))[1]
dN21 = np.sqrt(np.diag(pcov_a2))[0]
dl21 = np.sqrt(np.diag(pcov_a2))[1]
dT11 = np.log(2) * dl11 / l11**2
dT21 = np.log(2) * dl21 / l21**2

print("-----------   air fit 2:   -----------" )
print("N1 = " + str(N11) + " +/- " + str(dN11))
print("lamb1 = " + str(l11) + " +/- " + str(dl11))
print("T 1 (min) = " + str(T11 / 60) + " +/- " + str(dT11/60))
print("N2 = " + str(N21) + " +/- " + str(dN21))
print("lamb2 = " + str(l21) + " +/- " + str(dl21))
print("T 2 (h) = " + str(T21 / 3600) + " +/- " + str(dT21/3600))

plt.errorbar(t_air_5min_25h/3600, air_bin5min, yerr=dair_bin5min,  fmt="none", color='b', alpha=0.3, capsize=2)
plt.semilogy(t_air_5min_25h/3600, air_fit(t_air_5min_25h, N11, l11, N21, l21), label='fits combined', color='g')
plt.semilogy(t_air_5min_5h/3600, decay_fit(t_air_5min_5h, N11, l11), label='fit fast decay', color='k')
plt.semilogy(t_air_5min_25h/3600, decay_fit(t_air_5min_25h, N21, l21), label='fit slow decay', color='r')
plt.scatter(t_air_5min_25h/3600, air_bin5min, label='data', s=6)
plt.legend()
plt.ylim(0.1)
plt.title("Environmental air measurement")
plt.xlabel("time (h)")
plt.ylabel("activity (cps)")
plt.grid()
plt.savefig('air13.pdf', format='pdf')
plt.show()


#############################################################################


################   Stone measurements   #####################################
stone3_5min_counts = (from_txt('stone3_5min.txt') - background)*k
stone3_5min_bin10min = binning(stone3_5min_counts, 10*60)

n_stone3_10min = len(stone3_5min_bin10min)
dstone3_5min_bin10min = np.zeros(n_stone3_10min)
for i in range(n_stone3_10min):
    #dstone3_5min_bin10min[i] = deviation(stone3_5min_counts[i*60*10:(i+1)*60*10])
    #dstone3_5min_bin10min[i] = np.sqrt( (dstone3_5min_bin10min[i]*k)**2 + (d_background*k)**2 )
    dstone3_5min_bin10min[i] = d_A(sum(stone3_5min_counts[i*60*10:(i+1)*60*10]), 60*10)
#stone3_5min_bin5min = stone3_5min_bin5min[0:120]
n = len(stone3_5min_bin10min)
t_stone3_5min_bin10min = np.linspace(0, n*10*60, num=n)

def stone_fit_5min(t, N222, l1, l2, l3):
    return chain_3(t, l1, l2, l3, N222)

popt_s3_5min, pcov_s3_5min = curve_fit(stone_fit_5min, t_stone3_5min_bin10min, stone3_5min_bin10min, p0=[1e7, lambda_Rn222, lambda_Pb214, lambda_Bi214], maxfev=10000)

N0 = popt_s3_5min[0]
ls1 = popt_s3_5min[1]
ls2 = popt_s3_5min[2]
ls3 = popt_s3_5min[3]
Ts1 = np.log(2)/ls1
Ts2 = np.log(2)/ls2
Ts3 = np.log(2)/ls3

dN0 = np.sqrt(np.diag(pcov_s3_5min))[0]
dls1 = np.sqrt(np.diag(pcov_s3_5min))[1]
dls2 = np.sqrt(np.diag(pcov_s3_5min))[2]
dls3 = np.sqrt(np.diag(pcov_s3_5min))[3]
dTs1 = np.log(2) * dls1 / ls1**2
dTs2 = np.log(2) * dls2 / ls2**2
dTs3 = np.log(2) * dls3 / ls3**2


print("-----------   stone 5min:   -----------")
print("N0 = " + str(N0) + " +/- " + str(dN0))
print("lamb1 = " + str(ls1) + " +/- " + str(dls1))
print("T 1 (d) = " + str(Ts1 / (24*3600)) + " +/- " + str(dTs1/(24*3600)))
print("lamb2 = " + str(ls2) + " +/- " + str(dls2))
print("T 2 (min) = " + str(Ts2 / 60) + " +/- " + str(dTs2/60))
print("lamb3 = " + str(ls3) + " +/- " + str(dls3))
print("T 3 (min) = " + str(Ts3 / 60) + " +/- " + str(dTs3/60))

plt.errorbar(t_stone3_5min_bin10min/3600, stone3_5min_bin10min, yerr=dstone3_5min_bin10min,  fmt="none", color='b', alpha=0.3, capsize=2)
plt.plot(t_stone3_5min_bin10min/3600, stone_fit_5min(t_stone3_5min_bin10min, popt_s3_5min[0], popt_s3_5min[1], popt_s3_5min[2], popt_s3_5min[3]), label='fit', color='r')
plt.scatter(t_stone3_5min_bin10min/3600, stone3_5min_bin10min, label='data', s=1, color='b')
plt.title("Ingrowth measurement (sample 3)")
plt.xlabel("time (h)")
plt.ylabel("activity (cps)")
plt.grid()
plt.legend()
plt.savefig('ing3.pdf', format='pdf')
plt.show()

stone3_2h_counts = (from_txt('stone3_2h.txt') - background)*k
stone3_2h_bin1h = binning(stone3_2h_counts, 3600)

n_stone3_2h = len(stone3_2h_bin1h)
dstone3_2h_bin1h = np.zeros(n_stone3_2h)
for i in range(n_stone3_2h):
    #dstone3_2h_bin1h[i] = deviation(stone3_2h_counts[i*3600:(i+1)*3600])
    #dstone3_2h_bin1h[i] = np.sqrt( (dstone3_2h_bin1h[i]*k)**2 + (d_background*k)**2 )
    dstone3_2h_bin1h[i] = d_A(sum(stone3_2h_counts[i*3600:(i+1)*3600]), 3600)

n = len(stone3_2h_bin1h)
t_stone3_2h_bin1h = np.linspace(0, n*3600, num=n)

popt_s3_2h, pcov_s3_2h = curve_fit(decay_fit, t_stone3_2h_bin1h, stone3_2h_bin1h, p0=[1e8, lambda_Rn222], maxfev=10000)

Ns10 = popt_s3_2h[0]
ls10 = popt_s3_2h[1]
Ts10 = np.log(2)/ls10

dNs10 = np.sqrt(np.diag(pcov_s3_2h))[0]
dls10 = np.sqrt(np.diag(pcov_s3_2h))[1]
dTs10 = np.log(2) * dls10 / ls10**2

print("-----------   stone 2h:   -----------")
print("N0 = " + str(Ns10/2) + " +/- " + str(dNs10/2))
print("lamb1 = " + str(ls10) + " +/- " + str(dls10))
print("T 1 (h) = " + str(Ts10 / 3600) + " +/- " + str(dTs10/3600))

plt.errorbar(t_stone3_2h_bin1h/3600, stone3_2h_bin1h, yerr=dstone3_2h_bin1h,  fmt="none", color='b', alpha=0.3, capsize=2)
plt.plot(t_stone3_2h_bin1h/3600, decay_fit(t_stone3_2h_bin1h, popt_s3_2h[0], popt_s3_2h[1]), label='fit', color='r')
plt.scatter(t_stone3_2h_bin1h/3600, stone3_2h_bin1h, label='data', s=2, color='b')
plt.title("Decay measurement (sample 3)")
plt.xlabel("time (h)")
plt.ylabel("activity (cps)")
plt.grid()
plt.legend()
plt.savefig('dec3lin.pdf', format='pdf')
plt.show()

plt.errorbar(t_stone3_2h_bin1h/3600, stone3_2h_bin1h, yerr=dstone3_2h_bin1h,  fmt="none", color='b', alpha=0.3, capsize=2)
plt.semilogy(t_stone3_2h_bin1h/3600, decay_fit(t_stone3_2h_bin1h, popt_s3_2h[0], popt_s3_2h[1]), label='fit', color='r')
plt.scatter(t_stone3_2h_bin1h/3600, stone3_2h_bin1h, label='data', s=2, color='b')
plt.title("Decay measurement (sample 3)")
plt.xlabel("time (h)")
plt.ylabel("activity (cps)")
plt.grid()
plt.legend()
plt.savefig('dec3log.pdf', format='pdf')
plt.show()

