import numpy as np

no_patches = 144
width = 160
height = 160
num_channels = 224
bytes_per_element = 2  # float16 requires 2 bytes per element

file_size_bytes = width * height * num_channels * bytes_per_element
file_size_mb = file_size_bytes / (1024 * 1024)  # Convert bytes to megabytes

print(f"Estimated file size for 1 patch: {file_size_mb:.2f} MB")
print(f"Estimated file size for {no_patches} patches: {no_patches * file_size_mb:.2f} MB")
