import math
import torch
import torch.nn as nn

guidance_model_registry = {}

def register_guidance_model(name):
    def decorator(cls):
        guidance_model_registry[name] = cls
        return cls
    return decorator

def get_guidance_model(name, **kwargs):
    if name not in guidance_model_registry:
        raise ValueError(f"Guidance model '{name}' is not registered.")
    return guidance_model_registry[name]

def get_model(name, **kwargs):
    """
    Get a registered guidance model by name.
    """
    if name not in guidance_model_registry:
        raise ValueError(f"Model '{name}' is not registered.")
    return guidance_model_registry[name](**kwargs)

@register_guidance_model('dummy_encoder')
class DummyEncoder:
    def __init__(self):
        print("Initialized DummyEncoder")

@register_guidance_model('cnn_enc')
class Encoder(nn.Module):
    """
    Two tiny CNN encoders → GAP → concat → MLP → *logit*  (no sigmoid!)
    """
    def __init__(self, in_ch=3, base=48, n_steps=1000):
        super().__init__()

        def enc():
            return nn.Sequential(
                nn.Conv2d(in_ch,   base,   4, 2, 1), nn.SiLU(),
                nn.Conv2d(base,  base*2, 4, 2, 1), nn.GroupNorm(8, base*2), nn.SiLU(),
                nn.Conv2d(base*2, base*4, 4, 2, 1), nn.GroupNorm(8, base*4), nn.SiLU(),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten())

        self.enc_x = enc()
        self.enc_y = enc()
        self.t_embed = nn.Embedding(n_steps, base * 4)

        self.head = nn.Sequential(
            nn.Linear(base * 12, base * 4),
            nn.SiLU(),
            nn.Linear(base * 4, 1))

        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, a=math.sqrt(5))

    def forward(self, x_t, y_t, t):
        h = torch.cat([self.enc_x(x_t),
                       self.enc_y(y_t),
                       self.t_embed(t)], dim=1)
        return self.head(h).squeeze(1)

@register_guidance_model('cnn_disc')
class Discriminator(nn.Module):
    """
    Two tiny CNN encoders → GAP → concat → MLP → *logit*  (no sigmoid!)
    """
    def __init__(self, in_ch=3, base=48, n_steps=1000):
        super().__init__()

        def enc():
            return nn.Sequential(
                nn.Conv2d(in_ch,   base,   4, 2, 1), nn.SiLU(),
                nn.Conv2d(base,  base*2, 4, 2, 1), nn.GroupNorm(8, base*2), nn.SiLU(),
                nn.Conv2d(base*2, base*4, 4, 2, 1), nn.GroupNorm(8, base*4), nn.SiLU(),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten())

        self.enc_x = enc()
        self.enc_y = enc()
        self.t_embed = nn.Embedding(n_steps, base * 4)

        self.head = nn.Sequential(
            nn.Linear(base * 12, base * 4),
            nn.SiLU(),
            nn.Linear(base * 4, 1))

        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, a=math.sqrt(5))

    def forward(self, x_t, y_t, t):
        h = torch.cat([self.enc_x(x_t),
                       self.enc_y(y_t),
                       self.t_embed(t)], dim=1)
        return self.head(h).squeeze(1)