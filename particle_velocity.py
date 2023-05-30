import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize


# Define the Gaussian function
def gaussian(x, amplitude, mean, stddev):
    return amplitude * np.exp(-((x - mean) / (np.sqrt(2) * stddev)) ** 2)


def calculate_average_speed(run_date, run_list, bin_number, lower_speed_limit, upper_speed_limit):
    filtered_speeds = np.array([])
    bin_number = 100  # Number of bins for the histogram, keep the same for both run_list
    for r in run_list:
        run = r
        folder_location = "./" + run_date + "/" + run_date + run + "/"
        txt_file_all_speed = folder_location + run_date + run + "_all_speed.txt"
        # Read the input file and extract the values
        with open(txt_file_all_speed, "r") as infile:
            # Initialize an empty list to store the values
            values = []
        # Loop over each line in the file
            for line in infile:
                # Convert the line to a float and append it to the list of values
                values.append(float(line.strip()))

    # Convert the list of values to a NumPy array
        speeds = np.array(values)
    # Filter the speeds to remove values less than lower_speed_limit and greater than upper_speed_limit
        filtered_speeds_file = [
            speed for speed in speeds if speed >= lower_speed_limit and speed <= upper_speed_limit]
        filtered_speeds = np.concatenate(
            [filtered_speeds, filtered_speeds_file])

    # Plot the histogram
    fig, ax = plt.subplots()
    hist, bins, _ = ax.hist(filtered_speeds, bins=bin_number)

    # Compute the bin centers
    bin_centers = (bins[:-1] + bins[1:]) / 2
    # Find the index of the maximum frequency
    max_bin_index = np.argmax(hist)
    # Get the corresponding bin number
    max_bin_number = bin_centers[max_bin_index]

    # Fit the histogram to a Gaussian distribution
    popt, _ = optimize.curve_fit(gaussian, bin_centers, hist, p0=[1, np.mean(
        filtered_speeds), np.std(filtered_speeds)], maxfev=10000)

    # Compute the FWHM
    fwhm = 2 * np.sqrt(2 * np.log(2)) * popt[2]

    # Plot the fitted Gaussian function
    x = np.linspace(np.min(filtered_speeds), np.max(filtered_speeds), 100)
    ax.plot(x, gaussian(x, *popt), 'r-', label='Gaussian fit')

    # Set the axis labels and title
    ax.set_xlabel('Speed')
    ax.set_ylabel('Frequency')

    # Print the mean
    mean_speed = popt[1]
    print("Mean value of the fitted Gaussian:", mean_speed)
    print("Bin number with maximum frequency:", max_bin_number)
    # Draw a vertical line on the histogram at the bin with the maximum frequency
    plt.axvline(max_bin_number, color='r',
                linestyle='--', label='Max Frequency Bin')

    ax.set_title(
        f'Histogram of the speed (Mean: {mean_speed:.2f}), (Max frequency: {max_bin_number:.2f})')

    # Add a legend
    ax.legend()
    # Show the plot
    plt.show()
    
    return max_bin_number



def calculate_speed(run_date, run):
    # Location of the TXT file 
    folder_location = "./" + run_date + "/" + run_date + run + "/"
    txt_file = folder_location + run_date + run+"_particles.txt"

    txt_file_speed = folder_location + run_date + run + "_speed.txt"
    txt_file_all_speed = folder_location + run_date + run + "_all_speed.txt"
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

    # Calculate the speed of the particles
    # Zoom factor
    zoom_factor = 1.54 # zoom 1=1.54, and zoom 2=3.12, and zoom 2.2=3.68
    # the time between the pulses
    delta_time = 35 # us
    # pixel size
    pixel_size = 6.5 # um
    # factor
    factor = pixel_size/(delta_time*zoom_factor)
    # Calculate the difference in y_coords for all particles in each frame
    y_coord_diffs = {}

    for frame, particle_data in frame_data.items():
        y_coords_frame = [data[2] for data in particle_data]
        y_coord_diffs[frame] = [abs(y1 - y2) for idx1, y1 in enumerate(y_coords_frame) for idx2, y2 in enumerate(y_coords_frame) if idx1 != idx2]

    # Calculate the speed from the difference in y_coords for all particles in each frame
    speed = {}

    for frame, diffs in y_coord_diffs.items():
        scaled_diffs = [diff * factor for diff in diffs]
        speed[frame] = scaled_diffs



    # Create the histogram
    # Extract all the speed values
    all_speed = [diff for diffs in speed.values() for diff in diffs]

    # Plot the histogram
    fig, ax = plt.subplots()
    hist, bins, _ = ax.hist(all_speed, bins=50)

    # Compute the bin centers
    bin_centers = (bins[:-1] + bins[1:]) / 2

    # Fit the histogram to a Gaussian distribution
    popt, _ = optimize.curve_fit(gaussian, bin_centers, hist, p0=[1, np.mean(all_speed), np.std(all_speed)], maxfev=10000)

    # Compute the FWHM
    fwhm = 2 * np.sqrt(2 * np.log(2)) * popt[2]

    # Plot the fitted Gaussian function
    x = np.linspace(np.min(all_speed), np.max(all_speed), 100)
    ax.plot(x, gaussian(x, *popt), 'r-', label='Gaussian fit')

    # Set the axis labels and title
    ax.set_xlabel('Speed')
    ax.set_ylabel('Frequency')

    # Print the mean 
    print("Mean value of the fitted Gaussian:", popt[1])
    # Find the index of the maximum y value
    # Find the index of the maximum frequency
    max_bin_index = np.argmax(hist)
    # Get the corresponding bin number
    max_bin_number = bin_centers[max_bin_index]
    print("Bin number with maximum frequency:", max_bin_number)
    # Draw a vertical line on the histogram at the bin with the maximum frequency
    plt.axvline(max_bin_number, color='r', linestyle='--', label='Max Frequency Bin')

    ax.set_title(f'Histogram of the speed (Mean: {popt[1]:.2f}), (Max frequency: {max_bin_number:.2f})')

    # Save the output to a txt file
    with open(txt_file_speed, "w") as outfile:
        outfile.write(f"Speed for all particles in each frame (Mean: {popt[1]:.2f}), (Max frequency: {max_bin_number:.2f}):\n")
        for frame, diffs in speed.items():
            outfile.write(f"Frame {frame}: {diffs}\n")

    # save all_speed to a txt file
    np.savetxt(txt_file_all_speed, all_speed)

    # Add a legend
    ax.legend()
    # Show the plot
    plt.show()

    return speed
