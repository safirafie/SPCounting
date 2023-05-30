import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import gaussian
import cv2
from scipy import ndimage
from skimage.measure import regionprops, label
import os
from threshold_images_h5 import threshold_images
import h5py


# Control (13, 41, 1)
run_list = []
""" for i in range(41, 49, 1):
    run_list.append("_" + str(i)) """
run_list = ["_28", "_29"]
run_date = "20230314"
threshold_multiple = 2.6

# limit analysis
limit_analysis = 0
# Limit the number of images to analyze
limit_num_images = 10
# Limit the number of images to plot
limit_num_images_plot = 0
# Plot original image
plot_original_image = 0
# Plot image number
plot_original_image_number = 0
# Plot debackgrounded image
plot_debackgrounded_image = 0
# Plot filtered image
plot_filtered_image = 0
# Plot thresholded image
plot_thresholded_image = 0
# optimize images
optimize_images = 0

############################################################################################################## Functions
def append_image_to_hdf5(hdf5_file, image, particle_data):
    if 3 < len(particle_data) < 13:
        with h5py.File(hdf5_file, 'a') as hdf:
            dataset = hdf['images']
            dataset.resize((dataset.shape[0] + 1,) + dataset.shape[1:])
            dataset[-1] = image
# Get the path to the file
def get_file_path(filename):
    script_dir = os.path.dirname(os.path.realpath(__file__))
    file_path = os.path.join(script_dir, filename)
    return file_path

############################################################################################################## Main
for r in run_list:
    # run number
    run = r


    ##############################################################################################################
    # Location of the H5 file
    folder_location = "./" + run_date + "/" + run_date + run + "/"
    hdf5_file_r = folder_location + run_date + run+".h5"
    hdf5_file = get_file_path(hdf5_file_r)
    optimized_hdf5_file_r = folder_location + run_date + run+"_optimized.h5"
    optimized_hdf5_file = get_file_path(optimized_hdf5_file_r)
    
    # read the first image from the h5 file
    with h5py.File(hdf5_file, 'r') as hdf:
        images = hdf['images'][0]
        
    ## Read 
    images_height = images.shape[0]
    images_width = images.shape[1]
    print(images.shape)
    print(images.dtype)
    with h5py.File(hdf5_file, 'r') as hdf:
        num_images = len(hdf['images'][:])
    print(num_images)

    if plot_original_image == 1:
        plt.figure(1)
        plt.imshow(images,vmin = 0, vmax = 200)
        plt.title('Original Image')
        plt.show(block=False)

    # initialize a variable to count total number of particle in the stack
    total_num_particles = 0
    num_particles = np.zeros(num_images)

    particles_file_r = folder_location + run_date + run+"_particles.txt"
    particles_file = get_file_path(particles_file_r)
    with open(particles_file, 'w') as _:
        pass

    counts_file_r = folder_location + run_date + run+"_counts.txt"
    counts_file = get_file_path(counts_file_r)
    with open(counts_file, 'w') as _:
        pass

    if limit_analysis == 1:
        num_images = limit_num_images

    # loop through all the images in the stack
    for i in range(num_images):
        # read the image number i from the h5 file
        with h5py.File(hdf5_file, 'r') as hdf:
            image = hdf['images'][i]
        
        # Subtract the background (median) for each image
        median_image = np.median(image)
        debackgrounded_image = image - median_image
        
        # plot the image after subtracting background
        if plot_debackgrounded_image == 1:
            plt.figure(2)
            plt.imshow(debackgrounded_image,vmin = 0, vmax = 200)
            plt.title('debackgrounded image')
            plt.show()

        # Apply Gaussian filter to denoise the debackgrounded image
        filtered_image = gaussian(debackgrounded_image, sigma=2)
        # plot the image after applying Gaussian filter
        if plot_filtered_image == 1:
            plt.figure(3)
            plt.imshow(filtered_image,vmin = 0, vmax = 200)
            plt.title('filtered image')
            plt.show(block=False)
        
        # Apply thresholding to the filtered image
        # the threshold value is determined from the histogram using threshold_images.py
        threshold_value = threshold_images(hdf5_file, i, threshold_multiple, 0)
        #threshold_value = 10
        _, thresholded_image = cv2.threshold(filtered_image, threshold_value, 255, cv2.THRESH_BINARY)
        
        # plot the image after applying thresholding
        if plot_thresholded_image == 1:
            plt.figure(4)
            plt.imshow(thresholded_image,vmin = 0, vmax = 200)
            plt.title('thresholded image')
            plt.show(block=False) 

        # Label the connected components in the binary image
        labeled_image = label(thresholded_image)
        # Extract the properties of each particle
        particle_data = regionprops(labeled_image, image)
        # Plot the binary image with particle centroids and bounding boxes
        if i in range(0, limit_num_images_plot):
            fig, ax = plt.subplots()
            ax.imshow(debackgrounded_image, cmap='gray', vmin=0, vmax=200)
            for prop in particle_data:
                ax.plot(prop.centroid[1], prop.centroid[0], 'r*')
                ax.add_patch(plt.Rectangle((prop.bbox[1], prop.bbox[0]), prop.bbox[3] - prop.bbox[1], prop.bbox[2] - prop.bbox[0], fill=False, edgecolor='red', linewidth=1))
            plt.show()
        
        if optimize_images == 1:
            # Append the image to the optimized hdf5 file
            append_image_to_hdf5(optimized_hdf5_file, image, particle_data)
        
        # Count the number of particles
        num_particles[i] = len(particle_data)

        # Total Number of particles
        total_num_particles += num_particles[i]

        # Print the result
        print(f'Number of particles: {num_particles[i]}, in image number: {i + 1}')

        # Store the centroid values for each particle in a text file
        with open(particles_file, 'a') as fid:
            for prop in particle_data:
                if prop.centroid[1] > 800 and prop.centroid[1] < 1500:
                    fid.write(f'{i},{prop.label},{prop.centroid[1]},{prop.centroid[0]}\n')


    print(f"Total number of particles: {total_num_particles}")   

    num_particles = np.concatenate(([total_num_particles], num_particles))

    # Write the number of particles in each loop to a text file
    np.savetxt(counts_file, num_particles, fmt='%d')

