import cantera as ct
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from scipy.optimize import curve_fit

def density_igm(pressure, B, Temperature):
    density_1 = pressure/(B*Temperature)
    return density_1

def density_cantera(gas_eq, pressure, Temperature):
    gas = ct.Solution('JANAF.yaml')
    density = [0]*(len(pressure))
    for i in range(len(pressure)):
        gas.TPX = Temperature[i], pressure[i], {gas_eq:1}
        density[i] = gas.density_mass
    return density


def quad_func(x, a, b, c):
    return a*x**2 + b*x + c

def linear_func(x, a, b):
    return a*x + b

def euler_solver(function, x0, xmax, timestep):
    div = int((xmax - x0)/timestep)
    dy = [0]*(div-1)
    for i in range(div-1):
        dx = timestep
        dy[i] = function[i+1] - function[i]
    return [val/dx for val in dy]

"""
Parameters
"""

D_vent = 0.0006 #0.6mm
A = ((D_vent/2)**2)*math.pi #cross-section Area
R = 8.3145;  #J/mol*K
u = 0.04401; #kg/mol
B = R / u;
timestep = 1 #for derivative

"""
Data reading from CSV file
"""

data = pd.read_csv("vent_valve/ColdFlow201022.csv")
time = data['Time since tanking start [s]']
p_tank = data[' Pressure in tank [bar]']
mass_ox_tank = data[' Mass of oxidizer in tank [kg]']
temperature_ox = data[' Temperature of oxidizer [*C]']

"""
Test Charts
"""

fig, axs = plt.subplots(1, 3, figsize=(19, 5))

# Pressure
axs[0].plot(time, p_tank)
axs[0].set_xlabel('Time[s]')
axs[0].set_ylabel('Pressure[bar]')
axs[0].set_title('Pressure inside a tank')

# Tank mass
axs[1].plot(time, mass_ox_tank)
axs[1].set_xlabel('Time[s]')
axs[1].set_ylabel('Mass[kg]')
axs[1].set_title('Mass of a tank')

# Temperature
axs[2].plot(time, temperature_ox)
axs[2].set_xlabel('Time[s]')
axs[2].set_ylabel('Temperature[*C]')
axs[2].set_title('Temperature inside a tank')

plt.tight_layout()
plt.show()

"""
Calculations
"""

# Time limits when the valve was closed
t1_electrovalve_closed = 152.25
t2_electrovalve_opened = 280.55

# Indexes of time when valve was closed
time_index = time.index[time.between(t1_electrovalve_closed, t2_electrovalve_opened)].tolist()

# Density ideal gas
Pressure = p_tank[time_index]*100000
Temperature = temperature_ox[time_index]+273.15
density_ideal_gas = density_igm(Pressure, B, Temperature)
density_cantera_data = density_cantera('N2O', Pressure.values, Temperature.values)
time_plt = time[time_index]

# Curve fits for density calculations
parameters_igm, _ = curve_fit(quad_func, time_plt, density_ideal_gas)
x_fit_igm = np.linspace(min(time_plt), max(time_plt), len(time_plt))
y_fit_igm = quad_func(x_fit_igm, parameters_igm[0], parameters_igm[1], parameters_igm[2])

parameters_cantera, _ = curve_fit(quad_func, time_plt, density_cantera_data)
x_fit_cantera = np.linspace(min(time_plt), max(time_plt), len(time_plt))
y_fit_cantera = quad_func(x_fit_cantera, parameters_cantera[0], parameters_cantera[1], parameters_cantera[2])

# Density chart
plt.plot(time_plt, density_ideal_gas, label='Ideal Gas Data')
plt.plot(time_plt, density_cantera_data, label='Cantera Data')
plt.plot(x_fit_igm, y_fit_igm, 'r', label='Ideal gas Curve fitted')
plt.plot(x_fit_cantera, y_fit_cantera, 'b', label='Cantera Curve fitted')
plt.xlabel('Time[s]')
plt.ylabel('Density[kg/m^3]')
plt.title('Density Ideal Gas vs. Redlich-Kwong')
plt.legend()
plt.show()

# Mass released from a tank
Mass_released = max(mass_ox_tank[time_index]) - mass_ox_tank[time_index]

# Curve fit
parameters_1, _ = curve_fit(quad_func, time_plt, Mass_released)
y_fit_1 = quad_func(x_fit_igm, parameters_1[0], parameters_1[1], parameters_1[2])

# Mass released Chart
plt.plot(time_plt, Mass_released, label='Data')
plt.plot(x_fit_igm, y_fit_1, 'r', label='Curve fitted')
plt.xlabel('Time[s]')
plt.ylabel('Mass[kg]')
plt.title('Mass released from a tank')
plt.legend()
plt.show()

#Mass flow rate calculations using numerical derivative
x2_fit = np.linspace(min(x_fit_igm), max(x_fit_igm), (int((max(x_fit_igm)-min(x_fit_igm))/timestep)))
Mass_flow_released = euler_solver(quad_func(x2_fit, parameters_1[0], parameters_1[1], parameters_1[2]), min(x_fit_igm), max(x_fit_igm), timestep)

#linear aprox of mass flow rate
parameters_2, _ = curve_fit(linear_func, x2_fit[:-1], Mass_flow_released) #x2_fit needs to be cut by one because the values of Mass_flow_released are avarages between two x2_fit
time_fit = np.linspace(min(time_plt),max(time_plt),len(time_plt))

#We need to have a correct table size from linear aprox for Cd calculations (euler solver returns smaller size table)
Mass_flow_released_correct_size = linear_func(time_fit, parameters_2[0], parameters_2[1])


# Mass Flow chart
plt.plot(x2_fit[:-1], Mass_flow_released, label='Data')
plt.plot(time_fit, Mass_flow_released_correct_size, 'r', label='Curve fitted')
plt.xlabel('Time[s]')
plt.ylabel('Mass flow[kg/s]')
plt.title('Mass Flow from a tank')
plt.legend()
plt.show()

"""
Loss Coefficient
"""


Cd_igm = Mass_flow_released_correct_size/(A*((density_ideal_gas.values*Pressure.values*2)**0.5))
Cd_cantera = Mass_flow_released_correct_size/(A*((density_cantera_data*Pressure.values*2)**0.5))

#curve fit
parameters_3, _ = curve_fit(quad_func, time_plt, Cd_igm)
Cd_fit_igm = quad_func(time_fit, parameters_3[0], parameters_3[1], parameters_3[2])

parameters_4, _ = curve_fit(quad_func, time_plt, Cd_cantera)
Cd_fit_cantera = quad_func(time_fit, parameters_4[0], parameters_4[1], parameters_4[2])

plt.plot(time_fit, Cd_igm, label='Ideal gas Cd')
plt.plot(time_fit, Cd_cantera, label='Cantera Cd')
plt.plot(time_fit, Cd_fit_igm, 'r', label='Ideal gas Curve fitted')
plt.plot(time_fit, Cd_fit_cantera, 'b', label='Cantera Curve fitted')
plt.xlabel('Time[s]')
plt.ylabel('Cd')
plt.title('Loss coefficient')
plt.legend()
plt.show()

