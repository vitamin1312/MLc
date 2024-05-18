import torch

batch_size = 512
epochs = 200
sample_size = 64
nz = 128
k = 1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')