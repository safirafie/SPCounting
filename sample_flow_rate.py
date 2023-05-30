import math

def flow_rate(delta_p_psi, R=40e-6/2, eta=1.2e-3, L=0.30):
    '''
    The Hagen-Poiseuille equation relates the flow rate of a fluid through a capillary tube to the pressure 
    difference across the tube, the tube diameter, and the viscosity of the fluid. The equation is given by: 
    Q = (pi * R^4 * deltaP) / (8 * eta * L) 
    where 
    - Q is the flow rate, 
    - R is the radius of the capillary 
    - deltaP is the pressure difference 
    - eta is the viscosity of the fluid. viscosity of ethanol in Pa*s and viscosity of water approximately 0.001 Pa·s at 20°C
    - L is the length of the capillary

    Parameters:
    delta_p_psi: pressure drop in PSI can be a single value or an array of values
    R: radius of capillary in meters (default: 40e-6/2)
    eta: viscosity of fluid in Pa*s (default: 1.2e-3)
    L: length of capillary in meters (default: 0.30)
    '''
    delta_p = delta_p_psi * 6894.76  # pressure drop in Pascals

    flow_rate = (math.pi * R**4 * delta_p) / (8 * eta * L) # flow rate in m^3/s
    flow_rate_nl = flow_rate * 1e3* 1e9 *60  # flow rate in nl/min, where 1e3 converts m^3 to l, 1e9 converts l to nl and 60 converts s to min
    #print("The volume flow rate in m^3/s: ", flow_rate)
    #print("The volume flow rate in nl/min: ", flow_rate_nl)
    return flow_rate_nl
