import numpy as np
import torch
from torchvision.transforms import Compose, ToTensor, Lambda, ToPILImage, CenterCrop, Resize

# forward diffusion (using the nice property)

image_size = 128
transform = Compose([
    Resize(image_size),
    CenterCrop(image_size),
    ToTensor(), # turn into torch Tensor of shape CHW, divide by 255
    Lambda(lambda t: (t * 2) - 1),
    
])

reverse_transform = Compose([
     Lambda(lambda t: (t + 1) / 2),
     Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
     Lambda(lambda t: t * 255.),
     Lambda(lambda t: t.numpy().astype(np.uint8)),
     ToPILImage(),
])

def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

def q_sample(x_start, cumprod_alphas, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x_start)
    cumprod_alphas = extract(cumprod_alphas, t, x_start.shape)
    noised = torch.sqrt(cumprod_alphas) * x_start + (1-cumprod_alphas) * noise
    return noised

def get_noisy_image(x_start, cumprod_alphas, t):
  # add noise
  x_noisy = q_sample(x_start, cumprod_alphas, t=t)

  # turn back into PIL image
  noisy_image = reverse_transform(x_noisy.squeeze())

  return noisy_image
