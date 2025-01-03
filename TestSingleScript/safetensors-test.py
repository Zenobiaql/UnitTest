import os
import torch
import safetensors.torch as st

random_tensor = torch.rand(10000, 10000)

file_path = '/mnt/data-qilin/0103-ResourceUsing/test-tensor.safetensors'

if not os.path.exists(file_path):
    with open(file_path, 'w') as f:
        pass

st.save_file = (random_tensor, file_path)

tensors = {}
with st.safe_open(file_path, framework = "pt", device = 0) as f:
    for k in f.keys():
        tensors[k] = f.get_tensor(k)
print(tensors)