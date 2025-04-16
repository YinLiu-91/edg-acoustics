import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"
dtype = torch.float64
