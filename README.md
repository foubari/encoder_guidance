Encoder Guided Multimodal Diffusion

The generic command to do multimodal sampling:

python joint_sampling.py --model_x dummy_diffusion --model_y dummy_diffusion --guidance 'encoder_guided' --guidance_model 'dummy_encoder' --num_samples 4