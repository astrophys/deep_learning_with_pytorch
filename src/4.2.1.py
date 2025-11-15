import os
import torch
import imageio
import sys

data_dir = 'data/p1ch4/volumetric-dicom/2-LUNG 3.0  B70f-04083'
vol_arr = imageio.volread(data_dir, 'DICOM')
vol_arr.shape

vol = torch.from_numpy(vol_arr).float()
# Add extra dimension so torch expects it
vol = torch.unsqueeze(vol, 0)

sys.exit(0)


