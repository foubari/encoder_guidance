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

@register_guidance_model('dummy_encoder')
class DummyEncoder:
    def __init__(self):
        print("Initialized DummyEncoder")
