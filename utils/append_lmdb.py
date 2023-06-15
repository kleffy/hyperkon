import lmdb
import pickle
import os 
# Open the existing LMDB file in write mode
lmdb_file = '/vol/research/RobotFarming/Projects/hyperkon/database/test_val/val.lmdb'
env = lmdb.open(lmdb_file, map_size=1_555_627_776)  # Adjust the map_size as needed

# Create a new transaction to write the additional data
with env.begin(write=True) as txn:
    # Prepare the new key-value data
    new_key = 'num_samples'.encode()
    new_value = '55'

    # Convert the value to bytes using pickle serialization
    new_value_bytes = pickle.dumps(new_value)

    # Write the new key-value data to the transaction
    txn.put(new_key, new_value_bytes)

# Close the LMDB environment
env.close()
print('Done.')  


# # Open the LMDB environment in read-only mode
# env = lmdb.open(lmdb_file, readonly=True)

# # Get the file path of the LMDB database
# lmdb_file_path = env.path()

# # Get the size of the LMDB file
# lmdb_file_size = os.stat(lmdb_file_path).st_size

# # Close the LMDB environment
# env.close()

# # Print the size of the LMDB file
# print(f"LMDB file size: {lmdb_file_size} bytes")
