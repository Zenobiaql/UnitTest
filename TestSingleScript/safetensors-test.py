import os
import torch
import safetensors.torch as st

random_tensor = torch.rand(3, 3)

file_path = '/mnt/data-qilin/0103-ResourceUsing/test-tensor.safetensors'

if not os.path.exists(file_path):
    with open(file_path, 'w') as f:
        pass

st.save_file = (random_tensor, file_path)