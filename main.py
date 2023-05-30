import particle_linear_density as pld
import particle_velocity as pv
import transmission_efficiency as te
import numpy as np
import matplotlib.pyplot as plt
import sample_flow_rate as sfr
import tif_to_h5 as th5
import read_h5 as rh5
import read_particle_data as rpd

'''
This is the main program to run the functions in the other files. The main program is divided into different sections:
- Convert TIFF to HDF5 file
- Read the HDF5 file
- Count the particles in HDF5 file
- Plot the particles
- Read counts
- Calculate the speed and average speed
- Calculate the transmission efficiency
- Calculate the sample flow rate
'''




''' Counts:
20230314:
- UV 6:  10 and 11
- UV 8:  2 and 3
- UV 10: 18 and 19
- UV 12: 22 and 23
----------------
- xray 6:  30 and 31
- xray 8:  28 and 29
- xray 10: 32 and 33
- xray 12: 34 and 35
----------------
- voltage 6:  12 and 13
- voltage 8:  4 and 5
- voltage 10: 20 and 21
- voltage 12: 24 and 25
----------------
- orifice 6:  39 and 40
- orifice 8:  37 and 38
- orifice 10: 43 and 44
- orifice 12: 41 and 42
----------------
- gas 6:  63 and 64
- gas 8:  65 and 66
- gas 10: 67 and 68
- gas 12: 69 and 70
----------------
- multi 6:  78 and 79
- multi 8:  76 and 77
- multi 10: 74 and 75
- multi 12: 72 and 73
'''


######################################################################################## Convert TIFF to HDF5 file and read the HDF5 file
# Convert TIFF to HDF5 file
run_date_convert_tif_to_h5 = "20230417"
run_convert_tif_to_h5 = "_42"
convert_tif_to_h5 = 0

if convert_tif_to_h5 == 1:
    th5.convert_tif_to_h5(run_date_convert_tif_to_h5, run_convert_tif_to_h5)

######################################################################################### Read the HDF5 file
read_hdf5_file = 0
run_date_read_hdf5_file = "20230417"
run_read_hdf5_file = "_22"

if read_hdf5_file == 1:
    image_stack = rh5.read_images_from_hdf5(run_date_read_hdf5_file, run_read_hdf5_file)

######################################################################################## Count the particles
# Count the particles in HDF5 file
# Use SPCouting_h5.py to generate the particle data txt file


######################################################################################## Plot the particles
# Plot the particles
plot_particles = 0
run_date_plot_particles = "20230314"
run_plot_particles = "_5"

# Plot the particles
if plot_particles == 1:
    pld.plot_particles(run_date_plot_particles, run_plot_particles)


###################################################################################### Read counts
read_counts = 0
run_date_read_counts = "20230314"
run_read_counts = "_68"
if read_counts == 1:
    rpd.plot_particles(run_date_read_counts, run_read_counts)
    rpd.read_counts(run_date_read_counts, run_read_counts)

histogram_fitting_y_coords = 0
if histogram_fitting_y_coords == 1:
    rpd.histogram_fitting_y_coords(run_date_read_counts, run_read_counts)

######################################################################################### Calculate speed and average speed
calculate_speed = 0
run_date_speed = "20230417"
run_speed = "_55"
if calculate_speed == 1:
    pv.calculate_speed(run_date_speed, run_speed)

calculate_average_speed = 0
run_date_average_speed = "20230417"
run_list_1_l_per_min = ["_52", "_51", "_50", "_49","_48", "_47", "_44", "_41", "_40", "_39", "_38", "_37", "_36", "_35", "_34", "_33", "_32", "_31", "_23", "_22", "_21", "_20", "_19"] # 1 l/min
run_list_05_l_per_min = ["_30", "_29", "_28", "_27", "_26", "_25"] # 0.5 l/min
#        run_list = ["_52", "_51", "_50", "_49","_48", "_47", "_44", "_41", "_40", "_39", "_38", "_37", "_36", "_35", "_34", "_33", "_32", "_31", "_23", "_22", "_21", "_20", "_19"]
#        #run_list = ["_30", "_29", "_28", "_27", "_26", "_25"]
run_list_average_speed = run_list_05_l_per_min
bin_number = 100
lower_speed_limit = 22
upper_speed_limit = 40
configuration = "uv"

# Calculate the velocity
if calculate_average_speed == 1:
    particle_velocity = pv.calculate_average_speed(run_date_average_speed, run_list_average_speed, bin_number, lower_speed_limit, upper_speed_limit)
    particle_velocity = particle_velocity*1e6*60    # average velocity in um/min 
elif configuration == "gas":
    particle_velocity = 28.94e6*60    # um/min the average velocity from the runs with 0.5 l/min N2 flow rate
else:
    particle_velocity = 29.1e6*60    # um/min the average velocity from the runs with 1 l/min N2 flow rate
print("The average velocity is: ", particle_velocity/(1e6*60), " m/s")

################################################################################################### The sample flow rate
calculate_sample_flow_rate = 0

if calculate_sample_flow_rate == 1:
    delta_p_psi = np.array([6, 8, 10, 12])
    flow_rate_nl = sfr.flow_rate(delta_p_psi, R=40e-6/2, eta=1.2e-3, L=0.30)
    print("The flow rate is: ", flow_rate_nl, " nl/min")


######################################################################################### Particle linear density
calculate_linear_density = 0
run_date_linear_density = "20230314"
run = "_10"
plateau_start = 400
plateau_end = 1400

# Calculate the linear density
if calculate_linear_density == 1:
    data = pld.load_data(run_date_linear_density, run)
    linear_density = pld.calculate_linear_density(data, plateau_start, plateau_end)
    print("The linear density is: ", linear_density, " particles/m")


######################################################################################## Transmission efficiency
transmission_efficiency_for_different_plateaus = 0

# Calculate the linear density of the particles
configuration = "orifice"
if configuration == "uv":
    run_list_density = [10, 2, 18, 22]
elif configuration == "xray":
    run_list_density = [30, 28, 32, 34]
elif configuration == "voltage":
    run_list_density = [12, 4, 20, 24]
elif configuration == "orifice":
    run_list_density = [39, 37, 43, 41]
elif configuration == "gas":
    run_list_density = [63, 65, 67, 69]
elif configuration == "multi":
    run_list_density = [78, 76, 74, 72]
else:
    raise ValueError("Invalid configuration: " + configuration)

run_date_transmission = "20230314"

plateau = "very_coarse"
if plateau == "fine":
    plateau_range = [10,20,30,40,50,100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600,1700,1800,1900,2000]
elif plateau == "coarse":
    plateau_range = [200,220,240,260,280,300,320,340,360,380,400]
elif plateau == "very_coarse":
    plateau_range = [400]
else:
    raise ValueError("Invalid plateau: " + plateau)

'''
- S75E80-2: Concentration (Upgrade): 1.51e+011 particles/ml
- S75E80-N: Concentration (Upgrade): 4.90e+010 particles/ml
- original Particle Concentration: 2.6E+11
'''
particle_concentration = 1.3e+010 # particles/ml 20 times diluted and measured with the nano sight
zoom_level = 3.68 # zoom 1=1.54, and zoom 2=3.12, and zoom 2.2=3.68
pixel_size = 6.5 # um

# Calculate transmission efficiency For different plateaus
if transmission_efficiency_for_different_plateaus == 1:
    te.calculate_and_plot_transmission_efficiency(configuration, run_date_transmission, run_list_density, particle_concentration, zoom_level, pixel_size, particle_velocity, bin_number, plateau_range)


