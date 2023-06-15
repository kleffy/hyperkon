import h5py

# Open the file
file = h5py.File('/vol/research/RobotFarming/Projects/hyperkon/database/test_train/train.hdf5', 'r')

# Print the names of the datasets in the file
print("Datasets in the file:", list(file.keys()))

# Check the shapes of the datasets
print("Shape of 'anchor_patches':", file['anchor_patches'].shape)
print("Shape of 'positive_patches':", file['positive_patches'].shape)

# Close the file
file.close()
