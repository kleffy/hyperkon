import lmdb
import numpy as np

# Open the LMDB environment
env = lmdb.open('/vol/research/RobotFarming/Projects/hyperkon/database/test_train/train.lmdb', readonly=True)

# Create a transaction to read from the LMDB environment
with env.begin() as txn:
    # Print the names of the databases in the LMDB environment
    databases = txn.list_dbs()
    print("Databases in the LMDB environment:", databases)

    # Open the database 'anchor_patches'
    anchor_db = txn.open_db(b'anchor_patches')

    # Read data from the 'anchor_patches' database
    anchor_data = txn.get(b'', db=anchor_db)
    anchor_patches = np.frombuffer(anchor_data, dtype=np.float32)
    anchor_patches = anchor_patches.reshape((-1, anchor_patches.shape[0] // anchor_patches.shape[1]))

    # Open the database 'positive_patches'
    positive_db = txn.open_db(b'positive_patches')

    # Read data from the 'positive_patches' database
    positive_data = txn.get(b'', db=positive_db)
    positive_patches = np.frombuffer(positive_data, dtype=np.float32)
    positive_patches = positive_patches.reshape((-1, positive_patches.shape[0] // positive_patches.shape[1]))

# Print the shapes of the datasets
print("Shape of 'anchor_patches':", anchor_patches.shape)
print("Shape of 'positive_patches':", positive_patches.shape)
