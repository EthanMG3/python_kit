import h5py

# Replace with your actual file path
filename = '/home/emgreen/simulations/stellarator/183281/pcms_gnet/gtc.h5'

def print_h5_structure(name, obj):
    print(name)
    if isinstance(obj, h5py.Dataset):
        print(f"  Dataset shape: {obj.shape}, dtype: {obj.dtype}")

with h5py.File(filename, 'r') as f:
    print("Contents of the HDF5 file:")
    f.visititems(print_h5_structure)

    # Optional: access a specific dataset
    dataset = f['/gtc_n6d']
    print(dataset[:])  # print contents
 