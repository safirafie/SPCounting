import numpy as np
import math
import plot_figures as p
import matplotlib.pyplot as plt
import particle_linear_density as pld


def calculate_transmission(particle_concentration: float, particle_linear_density: np.ndarray, zoom_level: float = 3.68, pixel_size: float = 6.5, particle_velocity: float = 29.1e6*60) -> np.ndarray:
    """
    Calculates the transmission efficiency of particles.

    Args:
    particle_concentration (float): The concentration of particles in the sample in particles/ml.
    uv_normal_counts (np.ndarray): An array of particle counts per frame under UV light.
    plateau_in_pixel (int): The plateau in pixels.
    zoom_level (float): The zoom level.
    pixel_size (float): The pixel size in um.
    particle_velocity (float): The average velocity of particles in um/min.

    Returns:
    float: The transmission efficiency of particles under UV light.
    """

    # Convert flow rate from nl/min to ml/min
    '''
    Flow rate: 6, 8, 10, 12 PSI
    Flow rate nl per min: 433.21054729, 577.61406305, 722.01757881, 866.42109457
    '''
    flow_rate_ml_min = p.flow_rate_nl_min*1e-6 # ml/min

    # Calculate inflow of particles (particles per min)
    '''
    inflow of particles: (particle concentration (particles/ml) * sample flow rate (ml/min))
    '''
    inflow_of_particles = particle_concentration*flow_rate_ml_min # particles per min

    # Calculate outflow of particles (particles per min) = (particle velocity (um/min) * particle linear density (particles/pixel) * zoom level  / pixel size)
    outflow_of_particles = particle_velocity*particle_linear_density * zoom_level / pixel_size 

    # Calculate transmission efficiency
    transmission = outflow_of_particles/inflow_of_particles*100

    return transmission


def calculate_and_plot_transmission_efficiency(configuration, run_date, run_list_density, particle_concentration, zoom_level, pixel_size, particle_velocity, bin_number, plateau_range):
    # Initialize the transmission and average transmission lists
    transmission_list = []
    average_transmission_list = []

    for plateau in plateau_range:
        # Calculate the linear density
        linear_density = []

        for run in run_list_density:
            run_1 = "_" + str(run)
            run_2 = "_" + str(run+1)
            # Load the data
            particle_data_1 = pld.load_data(run_date, run_1)
            particle_data_2 = pld.load_data(run_date, run_2)
            particle_data = np.concatenate((particle_data_1, particle_data_2), axis=0)

            # Plot the particles
            #pld.plot_particles(particle_data, plateau_start, plateau_end)

            # Plot histogram of y-coordinates of the particles
            mean, std, fwhm = pld.calculate_histogram_stats(particle_data, bin_number)
            plateau_start = int(mean - plateau/2)
            plateau_end = int(mean + plateau/2)

            # Calculate the linear density
            linear_density_of_flow = pld.calculate_linear_density(particle_data, plateau_start, plateau_end)
            linear_density.append(linear_density_of_flow)

        # convert linear_density to numpy array
        linear_density = np.array(linear_density)
        transmission = calculate_transmission(particle_concentration, linear_density, zoom_level, pixel_size, particle_velocity)
        print('Transmission efficiency: ', transmission, '%', ', for plateau: ', plateau)
        # Average transmission efficiency
        average_transmission = np.mean(transmission)
        print('Average transmission efficiency: ', average_transmission, '%', ', for plateau: ', plateau)

        # Append the transmission and average transmission values to the list
        transmission_list.append(transmission)

        # Append the transmission and average transmission values to the list
        average_transmission_list.append(average_transmission)

    # Save the transmission and average transmission values to a file
    np.savetxt('./txt/efficiency/transmission_efficiency_' + configuration + '.txt', np.concatenate(transmission_list))
    # Save the transmission and average transmission values to a file
    np.savetxt('./txt/efficiency/average_transmission_efficiency_' + configuration + '.txt', average_transmission_list)
    np.savetxt('./txt/efficiency/plateau_range_' + configuration + '.txt', plateau_range)

    average_transmission_all = np.mean(average_transmission_list)
    print('Average transmission efficiency all: ', average_transmission_all, '%')

    # Plot plateau range vs average transmission
    plt.plot(plateau_range, average_transmission_list)
    plt.title('Plateau Range vs Average Transmission')
    plt.xlabel('Plateau Range')
    plt.ylabel('Average Transmission (%)')
    plt.ylim([0, 100])
    plt.show()
    