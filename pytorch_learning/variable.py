from tempfile import tempdir
import torch


tensor = torch.tensor([[1,2], [3,4]], dtype=torch.float)
tensor.requires_grad_(True)

v_out = torch.mean(tensor * tensor)
v_out.backward()

print(tensor.grad)
print(tensor.data.numpy())
