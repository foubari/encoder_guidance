import argparse
import torch
import random
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from model import get_model  # Ensure model.py is in the same directory
from dataset import get_data

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) #adds models/
from diffusion_models.model import get_diffusion_model

def main():

    # -------------------------------
    # Argument Parser Initialization
    # -------------------------------
    parser = argparse.ArgumentParser(description="Train Discriminator on real and fake image pairs, with the BCE loss.")
    parser.add_argument('--model_x', type=str, required=True, help='Name of the first diffusion model')
    parser.add_argument('--model_y', type=str, required=True, help='Name of the second diffusion model')
    parser.add_argument('--disc_model', type=str, default='cnn_disc', help='Discriminator model to use')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset to use for training')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train the discriminator')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for training")
    parser.add_argument("--save_every", type=int, default=10000, help="Checkpoint save frequency")
    parser.add_argument("--ckpt_path", type=str, default="disc_mm_disco.pt", help="Path to save checkpoint")
    args = parser.parse_args()

    device = args.device
    # Load diffusion models
    model_x = get_diffusion_model(args.model_x).to(device)
    model_y = get_diffusion_model(args.model_y).to(device)
    model_x.eval().requires_grad_(False)
    model_y.eval().requires_grad_(False)

    # dataloader
    dataloader = get_data(args.dataset, batch_size=args.batch_size, split="train")

    # discriminator model
    disc_model = get_model(args.disc_model).to(device)

    # optimizer and scaler
    opt = torch.optim.Adam(disc_model.parameters(), lr=args.lr, betas=(0.5, 0.999))
    scaler = GradScaler()

    # loss
    bce_logits = torch.nn.BCEWithLogitsLoss()

    for epoch in range(1, args.epochs+1):
        for x, y, labels in dataloader:

            x = x.to(device)
            y = y.to(device)
            labels = labels.to(device)

            B = x.size(0)
            t = torch.randint(0, 1000, (B,), device=device).long()

            eps_x = torch.randn_like(x)
            eps_y = torch.randn_like(y)
            x_t = model_x.q_sample(x, t, eps_x).requires_grad_(True)
            y_t = model_y.q_sample(y, t, eps_y).requires_grad_(True)

            with autocast(device_type="cuda"):
                logits = disc_model(x_t, y_t, t)
                L_disc = bce_logits(logits, labels)

            ratio = logits
            grad_x, grad_y = torch.autograd.grad(ratio.sum(), (x_t, y_t), create_graph=True)

            eps_pred_x = model_x.model(x_t, t)
            eps_pred_y = model_y.model(y_t, t)

            c_t = model_x.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
            L_denoise = ((eps_x - eps_pred_x + c_t * grad_x) ** 2).mean() + \
                        ((eps_y - eps_pred_y + c_t * grad_y) ** 2).mean()

            loss = L_disc + L_denoise

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

        if epoch % 1 == 0:
            print(f"epoch {epoch:6d} | L_disc {L_disc.item():.3f} | L_den {L_denoise.item():.3f}")

        if epoch % args.save_every == 0:
            torch.save({
                "epoch": epoch,
                "state_dict": disc_model.state_dict(),
                "opt": opt.state_dict()
            }, args.ckpt_path)

        print("⇨ discriminator training finished")

if __name__ == "__main__":
    main()