import argparse
from models.diffusion_models import get_diffusion_model
from models.guidance_models import get_guidance_model
from models.multimodal_guided_models import get_multimodal_guided_model

def main():
    parser = argparse.ArgumentParser(description="Multimodal Joint Sampling")
    parser.add_argument('--model_x', type=str, required=True, help='Name of the first diffusion model')
    parser.add_argument('--model_y', type=str, required=True, help='Name of the second diffusion model')
    parser.add_argument('--guidance', type=str, required=True, choices=['encoder_guided'], help='Type of guidance for joint generation')
    parser.add_argument('--guidance_model', type=str, required=True, help='Name of the guidance model')
    parser.add_argument('--num_samples', type=int, default=1, help='Number of samples to generate')
    
    args = parser.parse_args()

    # Load diffusion models
    model_x = get_diffusion_model(args.model_x)
    model_y = get_diffusion_model(args.model_y)

    # Load guidance model
    guidance_model = get_guidance_model(args.guidance_model)

    # Load the multimodal model
    model = get_multimodal_guided_model(args.guidance)(model_x=model_x, model_y=model_y, guidance_model=guidance_model)

    # Perform sampling
    model.sample(num_samples=args.num_samples)

if __name__ == "__main__":
    main()
