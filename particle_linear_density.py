import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize



def load_data(run_date: str, run: str) -> np.ndarray:
    """
    Loads the particle data for a given run.

    Args:
    run_date (str): The date of the run in the format "YYYYMMDD".
    run (str): The run number.

    Returns:
    The particle data.
    """

    # Location of the TXT file 
    folder_location = "./" + run_date + "/" + run_date + run + "/"
    txt_file = folder_location + run_date + run+"_particles.txt"

    # Load the data into a numpy array
    data = np.loadtxt(txt_file, delimiter=',')

    return data

def save_data(run_date: str, run: str, name: str, data: np.ndarray) -> None:
    # Location of the TXT file
    folder_location = "./" + run_date + "/" + run_date + run + "/"
    saved_file = folder_location + run_date + run + name + ".txt"
    # Save the data to a file
    np.savetxt(saved_file, data)



def calculate_linear_density(data, plateau_start: int = 0, plateau_end: int = 2048) -> float:
    """
    Calculates the linear density of particles in a given particle_data and returns a linear density for the whole run.

    Args:
    plateau_start (int): The start of the plateau in pixels.
    plateau_end (int): The end of the plateau in pixels.

    Returns:
    The linear density.
    """

    # Extract the columns of data
    y_coords = data[:, 3]

    # Trim y_coords to only include the plateau
    y_coords = y_coords[(y_coords > plateau_start) & (y_coords < plateau_end)]

    # number of particles per frame
    particle_per_frame = len(y_coords)/1e3

    # Calculate the linear density
    linear_density = particle_per_frame / (plateau_end - plateau_start)

    return linear_density


def plot_particles(run_date: str, run: str, plateau_start: int = 0, plateau_end: int = 2048) -> None:
    """
    Plots the particles in a single plot, with separate legends for each frame.

    Args:
    frame_data (Dict[int, List[Tuple[int, float, float]]]]): A dictionary of particle data for each frame.
    plateau_start (int): The start of the plateau in pixels.
    plateau_end (int): The end of the plateau in pixels.

    Returns:
    None
    """
    # Load the data
    data = load_data(run_date, run)

    # Extract the columns of data
    frames = data[:, 0]
    particles = data[:, 1]
    x_coords = data[:, 2]
    y_coords = data[:, 3]

    # Create a dictionary to store particle data for each frame
    frame_data = {}
    for frame, particle, x, y in zip(frames, particles, x_coords, y_coords):
        if frame not in frame_data:
            frame_data[frame] = []
        frame_data[frame].append((particle, x, y))

    fig, ax = plt.subplots()

    for frame, particle_data in frame_data.items():
        for particle, x, y in particle_data:
            ax.plot(x, y, 'o', label=f"Particle {int(particle)}, Frame {int(frame)}")

        # Set the limits of the axes
        plt.xlim(0, 2048)
        plt.ylim(plateau_start, plateau_end)

    # Set the labels of the axes
    ax.set_xlabel('x (pixels)')
    ax.set_ylabel('y (pixels)')

    plt.show()



def calculate_histogram_stats(data: np.ndarray, bin_number: int = 100) -> tuple[float, float, float]:
    """
    Calculates the FWHM of a Gaussian fit to the histogram of the input data.

    Args:
    data (np.ndarray): The input data.
    bin_number (int): The number of bins for the histogram.

    Returns:
    float: The FWHM of the Gaussian fit.
    """

    def gaussian(x, a, x0, sigma):
        """
        Gaussian function.

        Args:
        x (float): The input value.
        a (float): The amplitude of the Gaussian.
        x0 (float): The mean of the Gaussian.
        sigma (float): The standard deviation of the Gaussian.

        Returns:
        float: The value of the Gaussian function at x.
        """
        return a * np.exp(-(x - x0)**2 / (2 * sigma**2))

    data = data[:, 3] # y-coordinates
    # Create the histogram
    fig, ax = plt.subplots()
    hist, bins, _ = ax.hist(data, bins=bin_number)

    # Convert hist to a proper array of floats
    hist = hist.astype(float)

    # Compute the bin centers
    bin_centers = (bins[:-1] + bins[1:]) / 2

    # Fit the histogram to a Gaussian distribution
    popt, _ = optimize.curve_fit(gaussian, bin_centers, hist, p0=[1, np.mean(data), np.std(data)], maxfev=10000)

    # Compute the FWHM
    fwhm = 2 * np.sqrt(2 * np.log(2)) * popt[2]

    # Find the index of the maximum frequency
    max_bin_index = np.argmax(hist)
    # Get the corresponding bin number
    max_bin_number = bin_centers[max_bin_index]


    # Plot the fitted Gaussian function
    x = np.linspace(np.min(data), np.max(data), 100)
    ax.plot(x, gaussian(x, *popt), 'r-', label='Gaussian fit')

    # Set the axis labels and title
    ax.set_xlabel('Value')
    ax.set_ylabel('Frequency')
    # Draw a vertical line on the histogram at the bin with the maximum frequency
    plt.axvline(max_bin_number, color='r',
                linestyle='--', label='Max Frequency Bin')

    ax.set_title(
        f'Histogram of the speed (Mean: {popt[1]:.2f}), fwhm: {fwhm:.2f}), (Max frequency: {max_bin_number:.2f})')

    # Mean and standard deviation of the data
    mean = popt[1]
    std = popt[2]
    # Add a legend
    ax.legend()

    # Show the plot
    #plt.show()
    plt.close(fig)

    return float(mean), float(std), float(fwhm)
