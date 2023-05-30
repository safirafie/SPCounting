import h5py
import tifffile
import os
from typing import Tuple

############################################################################################################## Functions
# Get the path to the file
def get_file_path(filename):
    '''
    This function returns the path to a file.
    '''
    script_dir = os.path.dirname(os.path.realpath(__file__))
    file_path = os.path.join(script_dir, filename)
    return file_path

# Convert an OME-TIFF file to an HDF5 file
def convert_ometiff_to_hdf5(input_file, output_file):
    '''
    This function converts an OME-TIFF file to an HDF5 file.
    '''
    with tifffile.TiffFile(input_file) as tif:
        image_stack = tif.asarray()

    with h5py.File(output_file, 'w') as hdf:
        hdf.create_dataset('images', data=image_stack)

# Convert an HDF5 file to an OME-TIFF file
def convert_tif_to_h5(run_date: str, run: str) -> Tuple[str, str]:
    '''
    This function converts an OME-TIFF file to an HDF5 file.

    Parameters:
    run_date: the date of the run in the format "YYYYMMDD"
    run: the run number as a string, e.g. "_42"

    Returns:
    A tuple containing the paths to the HDF5 file and the OME-TIFF file, respectively.
    '''
    # Get the path to H5 file
    folder_location = "./" + run_date + "/" + run_date + run + "/"
    hdf5_file_r = folder_location + run_date + run + ".h5"
    hdf5_file = get_file_path(hdf5_file_r)

    # Get the path to the OME-TIFF file
    tif_file_r = folder_location + run_date + run + "_MMStack_Default.ome.tif"
    tif_file = get_file_path(tif_file_r)

    # Convert the OME-TIFF file to an HDF5 file
    convert_ometiff_to_hdf5(tif_file, hdf5_file)

    return (hdf5_file, tif_file)
