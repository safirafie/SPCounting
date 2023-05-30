import h5py
import matplotlib.pyplot as plt
import os



# Get the path to the file
def get_file_path(filename):    
    script_dir = os.path.dirname(os.path.realpath(__file__))
    file_path = os.path.join(script_dir, filename)
    return file_path

# List the objects in the HDF5 file
def list_objects_in_hdf5(hdf5_file):
    with h5py.File(hdf5_file, 'r') as hdf:
        for key in hdf.keys():
            print(key)

# Display the first image in the stack
def display_first_image(image_stack):
    plt.imshow(image_stack[0], cmap='gray')
    plt.show()

# Print the metadata for the HDF5 file
def print_metadata(hdf5_file):
    with h5py.File(hdf5_file, 'r') as hdf:
        for key in hdf.keys():
            print(f"Metadata for object '{key}':")
            for attr_name in hdf[key].attrs:
                print(f"  {attr_name}: {hdf[key].attrs[attr_name]}")

# Read the images from the HDF5 file
def read_images_from_hdf5(run_date: str, run: str):
    # Get the path to h5 file
    folder_location = "./" + run_date + "/" + run_date + run + "/"
    hdf5_file_r = folder_location + run_date + run + ".h5"
    hdf5_file = get_file_path(hdf5_file_r)
    list_objects_in_hdf5(hdf5_file)
    with h5py.File(hdf5_file, 'r') as hdf:
        image_stack = hdf['images'][:]
    
    display_first_image(image_stack)
    print_metadata(hdf5_file)
    return image_stack

