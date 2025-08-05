import os
import torch
from denoising_diffusion_pytorch import Unet, GaussianDiffusion


base_dir = os.path.dirname(__file__)
diffusion_model_registry = {}

def register_diffusion_model(name):
    def decorator(cls):
        diffusion_model_registry[name] = cls
        return cls
    return decorator

def get_diffusion_model(name, **kwargs):
    if name not in diffusion_model_registry:
        raise ValueError(f"Diffusion model '{name}' is not registered.")
    return diffusion_model_registry[name](**kwargs)

@register_diffusion_model('dummy_diffusion')
class DummyDiffusionModel:
    def __init__(self):
        print("Initialized DummyDiffusionModel")

### REQUIREMENTS for a diffusion model class ###
# - Must have a `num_timesteps` attribute
# - Must have a `score` method that takes an image tensor and a timestep : score(x_t, t)

@register_diffusion_model('day')
class DayGaussianDiffusion(GaussianDiffusion):
    def __init__(self, weights_path=None, *args, **kwargs):

        if weights_path is None:
            weights_path = os.path.join(base_dir, 'pretrained_weights/day/checkpoint.pt')

        model = Unet(
            dim = 64,
            dim_mults = (1, 2, 4),
            flash_attn = False
        )

        super().__init__(model=model, image_size=64, timesteps=1000, sampling_timesteps=250, *args, **kwargs)

        self.load_state_dict(torch.load(weights_path))
    
    def score(self, x_t, t):
        """
        Override the score method to return the model's prediction.
        """
        return self.model(x_t, t)

@register_diffusion_model('night')
class NightGaussianDiffusion(GaussianDiffusion):
    def __init__(self, weights_path=None, *args, **kwargs):

        if weights_path is None:
            weights_path = os.path.join(base_dir, 'pretrained_weights/night/checkpoint.pt')
        
        model = Unet(
            dim = 64,
            dim_mults = (1, 2, 4),
            flash_attn = False
        )

        super().__init__(model=model, image_size=64, timesteps=1000, sampling_timesteps=250, *args, **kwargs)

        self.load_state_dict(torch.load(weights_path))
    
    def score(self, x_t, t):
        """
        Override the score method to return the model's prediction.
        """
        return self.model(x_t, t)