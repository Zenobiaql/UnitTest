import torch
from safetensors.torch import save_file, load_file

# Step 1: Initialize a random tensor
random_tensor = torch.rand(2, 3)
print("Original Tensor:")
print(random_tensor)

# Step 2: Save the tensor to a safetensors file
tensors = {"random_tensor": random_tensor}
save_file(tensors, "/mnt/data-qilin/0103-ResourceUsing/random_tensor.safetensors")

# Step 3: Load the tensor back from the safetensors file
loaded_tensors = load_file("/mnt/data-qilin/0103-ResourceUsing/random_tensor.safetensors")
loaded_tensor = loaded_tensors["random_tensor"]

# Step 4: Print the loaded tensor
print("\nLoaded Tensor:")
print(loaded_tensor)