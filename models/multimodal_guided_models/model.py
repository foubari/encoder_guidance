import torch


model_registry = {}

def register_model(name):
    def decorator(cls):
        model_registry[name] = cls
        return cls
    return decorator

def get_multimodal_guided_model(name, **kwargs):
    if name not in model_registry:
        raise ValueError(f"Model '{name}' is not registered.")
    return model_registry[name]

@register_model('encoder_guided')
class EncoderGuidedMultimodalDiffusion:
    def __init__(self, model_x, model_y, guidance_model, device):
        self.model_x = model_x
        self.model_y = model_y
        self.encoder = guidance_model
        self.device = device

        assert self.model_x.num_timesteps == self.model_y.num_timesteps, \
            "Both models must have the same number of timesteps for joint sampling."
        
        self.num_timesteps = self.model_x.num_timesteps
        self.data_shape_x = self.model_x.data_shape # (C, H, W)
        self.data_shape_y = self.model_y.data_shape

    def sample(self, num_samples=1):
        print(f"Sampling {num_samples} samples using EncoderGuidedMultimodalDiffusion...")
        
        # put all the models to device and to eval mode
        self.model_x.to(self.device).eval()
        self.model_y.to(self.device).eval()
        self.encoder.to(self.device).eval()

        # Initialize the modalities with noise
        x_t = torch.randn((num_samples, *self.data_shape_x), device=self.device)
        y_t = torch.randn((num_samples, *self.data_shape_y), device=self.device)

        for t in reversed(range(0, self.num_timesteps)):
            # Get the scores from both models
            score_x_t = self.model_x.score(x_t, t)
            score_y_t = self.model_y.score(y_t, t)

            # Get the guidance from the encoder
            guidance_term = self.encoder(x_t, y_t, t)

            #TO DO:update the images with these terms
        
        return x_t, y_t
