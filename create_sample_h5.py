import numpy as np
import h5py
import os

# Create a sample .h5 file with 'img' dataset containing 14 channels
# This is the format the Streamlit app expects for demonstration

sample_dir = r'C:\Users\Pratham\Desktop\landlside UI'
os.makedirs(sample_dir, exist_ok=True)

filepath = os.path.join(sample_dir, 'sample_demo.h5')

with h5py.File(filepath, 'w') as f:
    # Create 14 channels of random data (simulating multispectral satellite imagery)
    # Shape: (14, 64, 64) - channel-first format, which the app accepts
    img_data = np.random.rand(14, 64, 64).astype(np.float32)
    f.create_dataset('img', data=img_data)

print(f'Created sample file: {filepath}')

# Verify the file structure
with h5py.File(filepath, 'r') as f:
    print(f'Dataset keys: {list(f.keys())}')
    img_shape = f['img'].shape
    print(f'img shape: {img_shape}')
    print(f'Format: (14 channels, H=64, W=64) - accepted by the app')
    print(f'Also accepted: (H, W, 14) format - transpose if needed')