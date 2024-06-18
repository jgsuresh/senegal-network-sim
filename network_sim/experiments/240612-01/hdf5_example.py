import numpy as np
import h5py

def create_file():
    # Example dictionary with numpy arrays
    data = {
        'array1': np.array([1, 2, 3]),
        'array2': np.array([4, 5, 6])
    }

    # Save dictionary to an HDF5 file with compression
    with h5py.File('data_compressed.hdf5', 'w') as f:
        for key, array in data.items():
            f.create_dataset(key, data=array, compression='gzip', compression_opts=9)


if __name__ == "__main__":
    create_file()
    # # Load the HDF5 file
    # with h5py.File('data_compressed.hdf5', 'r') as f:
    #     for key in f.keys():
    #         print(key, f[key][:])