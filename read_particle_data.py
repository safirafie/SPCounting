import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import optimize, stats
from typing import Dict, Tuple



# Define the Gaussian function
def gaussian(x, amplitude, mean, stddev):
    return amplitude * np.exp(-((x - mean) / (np.sqrt(2) * stddev)) ** 2)

def load_data(run_date, run):
    # Location of the TXT file 
    folder_location = "./" + run_date + "/" + run_date + run + "/"
    txt_file = folder_location + run_date + run+"_particles.txt"    
    # Load the data into a numpy array
    data = np.loadtxt(txt_file, delimiter=',')

    # Extract the columns of data
    frames = data[:, 0]
    particles = data[:, 1]
    x_coords = data[:, 2]
    y_coords = data[:, 3]
    print(len(particles))

    # Create a dictionary to store particle data for each frame
    frame_data = {}
    for frame, particle, x, y in zip(frames, particles, x_coords, y_coords):
        if frame not in frame_data:
            frame_data[frame] = []
        frame_data[frame].append((particle, x, y))

    return frame_data, y_coords


def plot_particles(run_date, run):
    # Plot the particles in a single plot, with separate legends for each frame
    frame_data, y_coords = load_data(run_date, run)
    # Plot the particles in a single plot, with separate legends for each frame
    fig, ax = plt.subplots()

    for frame, particle_data in frame_data.items():
        for particle, x, y in particle_data:
            ax.plot(x, y, 'o', label=f"Particle {int(particle)}, Frame {int(frame)}")

        # Set the limits of the axes
        plt.xlim(0, 2048)

    # Set the labels of the axes
    ax.set_xlabel('x (pixels)')
    ax.set_ylabel('y (pixels)')

    plt.show()

def histogram_fitting_y_coords(run_date, run):
    # Load the data into a numpy array
    frame_data, y_coords = load_data(run_date, run)
    # Create the histogram
    fig, ax = plt.subplots()
    hist, bins, _ = ax.hist(y_coords, bins=100)

    # Compute the bin centers
    bin_centers = (bins[:-1] + bins[1:]) / 2

    # Fit the histogram to a Gaussian distribution
    popt, _ = optimize.curve_fit(gaussian, bin_centers, hist, p0=[1, np.mean(y_coords), np.std(y_coords)], maxfev=10000)

    # Compute the FWHM
    fwhm = 2 * np.sqrt(2 * np.log(2)) * popt[2]

    # Plot the fitted Gaussian function
    x = np.linspace(np.min(y_coords), np.max(y_coords), 100)
    ax.plot(x, gaussian(x, *popt), 'r-', label='Gaussian fit')

    # Set the axis labels and title
    ax.set_xlabel('Value')
    ax.set_ylabel('Frequency')
    ax.set_title('Histogram of y_coords')

    # Add a legend
    ax.legend()

    # Print the FWHM
    print('FWHM:', fwhm)

    # Show the plot
    plt.show()

def read_counts(run_date, run):
    # Load the data into a numpy array
    frame_data, y_coords = load_data(run_date, run)
    
    # Create a dictionary to store the number of particles for each frame
    num_particles = {}
    for frame, particle_data in frame_data.items():
        num_particles[frame] = len(particle_data)

    # Plot histogram of the number of particles in each frame
    fig, ax = plt.subplots()
    ax.hist(num_particles.values(), bins=50)

    # Fit the histogram to a Gaussian distribution
    hist, bins, _ = ax.hist(num_particles.values(), bins=50)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    # Convert dict_values to a NumPy array
    num_particles_array = np.array(list(num_particles.values()))

    # Calculate mean and standard deviation
    mean_value = np.mean(num_particles_array)


    popt, _ = optimize.curve_fit(gaussian, bin_centers, hist, p0=[1, np.mean(num_particles_array), np.std(num_particles_array)], maxfev=10000)
    fwhm = 2 * np.sqrt(2 * np.log(2)) * popt[2]
    x = np.linspace(np.min(num_particles_array), np.max(num_particles_array), 100)
    ax.plot(x, gaussian(x, *popt), 'r-', label='Gaussian fit')
    ax.set_xlabel('Value')
    ax.set_ylabel('Frequency')
    ax.set_title('Histogram of number of particles in each frame')
    print('FWHM:', fwhm)
    # Print mean value
    print("Mean value:", mean_value)    
    print("1/2 Mean value:", mean_value/2)    
    plt.show()
