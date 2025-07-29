diffusion_model_registry = {}

def register_diffusion_model(name):
    def decorator(cls):
        diffusion_model_registry[name] = cls
        return cls
    return decorator

def get_diffusion_model(name, **kwargs):
    if name not in diffusion_model_registry:
        raise ValueError(f"Diffusion model '{name}' is not registered.")
    return diffusion_model_registry[name]

@register_diffusion_model('dummy_diffusion')
class DummyDiffusionModel:
    def __init__(self):
        print("Initialized DummyDiffusionModel")

### REQUIREMENTS for a diffusion model class ###
# - Must have a `num_timesteps` attribute
# - Must have a `data_shape` attribute (tuple of (C, H, W))
# - Must have a `score` method that takes an image tensor and a timestep : score(x_t, t)
