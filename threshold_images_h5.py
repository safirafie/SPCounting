import numpy as np
import matplotlib.pyplot as plt
import h5py

# plot_histogram: Plot the histogram of the image
def threshold_images(hdf5_file, i, threshold_multiple, plot_histogram=0):
    with h5py.File(hdf5_file, 'r') as hdf:
        image = hdf['images'][i]
    # Calculate the histogram
    histogram, bin_edges = np.histogram(image.ravel(), bins=2048)

    # Find the peak of the background
    background_peak_index = np.argmax(histogram)
    background_peak_value = bin_edges[background_peak_index]

    # Calculate the standard deviation of the background peak
    background_std = np.std(image)

    # Calculate the threshold
    threshold = background_peak_value + threshold_multiple * background_std
    
    # Subtract the median
    median_image = np.median(image)
    threshold_after_median = threshold - median_image
    print("Threshold after median:", threshold_after_median)

    if plot_histogram == 1:
        # Print the threshold
        print("Threshold before median:", threshold)
        # Plot the histogram
        plt.figure()
        plt.plot(bin_edges[:-1], histogram)
        plt.axvline(threshold, color='r', linestyle='--')
        plt.xlabel("Pixel intensity")
        plt.ylabel("Frequency")
        plt.title("Image Histogram")
        # Show the plot
        plt.show()
        
    return threshold_after_median