import os.path
import shutil
from config import get_config
from scheduler import MipLRDecay
from loss import NeRFLoss, mse_to_psnr
from model import MipNeRF
import torch
import torch.optim as optim
import wandb
from os import path
from datasets import get_dataloader, cycle
import numpy as np
from tqdm import tqdm


def train_model(config):
    
    if config.with_wandb:
        print("Logging to wandb")
        run = wandb.init(
            project="mip-nerf",
            entity="rp-nerf",
            config=config,
            resume="allow",
            save_code=True
        )
    else:
        print("No logging")
        run = wandb.init(mode="disabled")

    data = iter(
        cycle(
            get_dataloader(
                dataset_name=config.dataset_name,
                base_dir=config.base_dir,
                split="train",
                factor=config.factor,
                batch_size=config.batch_size,
                shuffle=True,
                device=config.device,
            )
        )
    )
    eval_data = None
    if config.do_eval:
        print("Evaluating model every", config.save_every, "steps")
        eval_data = iter(
            cycle(
                get_dataloader(
                    dataset_name=config.dataset_name,
                    base_dir=config.base_dir,
                    split="test",
                    factor=config.factor,
                    batch_size=config.batch_size,
                    shuffle=True,
                    device=config.device,
                )
            )
        )

    model = MipNeRF(
        use_viewdirs=config.use_viewdirs,
        randomized=config.randomized,
        ray_shape=config.ray_shape,
        white_bkgd=config.white_bkgd,
        num_levels=config.num_levels,
        num_samples=config.num_samples,
        hidden=config.hidden,
        density_noise=config.density_noise,
        density_bias=config.density_bias,
        rgb_padding=config.rgb_padding,
        resample_padding=config.resample_padding,
        min_deg=config.min_deg,
        max_deg=config.max_deg,
        viewdirs_min_deg=config.viewdirs_min_deg,
        viewdirs_max_deg=config.viewdirs_max_deg,
        device=config.device,
    )
    optimizer = optim.AdamW(
        model.parameters(), lr=config.lr_init, weight_decay=config.weight_decay
    )
    if config.continue_training:
        raise NotImplementedError

    scheduler = MipLRDecay(
        optimizer,
        lr_init=config.lr_init,
        lr_final=config.lr_final,
        max_steps=config.max_steps,
        lr_delay_steps=config.lr_delay_steps,
        lr_delay_mult=config.lr_delay_mult,
    )
    loss_func = NeRFLoss(config.coarse_weight_decay)
    model.train()

    for step in tqdm(range(0, config.max_steps), ncols=100):
        rays, pixels = next(data)
        comp_rgb, _, _ = model(rays)
        pixels = pixels.to(config.device)

        # Compute loss and update model weights.
        loss_val, psnr = loss_func(comp_rgb, pixels, rays.lossmult.to(config.device))
        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()
        scheduler.step()

        psnr = psnr.detach().cpu().numpy()
        step_results = {}
        step_results["train/loss"] = float(loss_val.detach().cpu().numpy())
        step_results["train/coarse_psnr"] = float(np.mean(psnr[:-1]))
        step_results["train/fine_psnr"] = float(psnr[-1])
        step_results["train/avg_psnr"] = float(np.mean(psnr))
        step_results["train/lr"] = float(scheduler.get_last_lr()[-1])
        run = wandb.log(step_results, step=step)

        if step % config.save_every == 0:
            if eval_data:
                del rays
                del pixels
                psnr = eval_model(config, model, eval_data)
                psnr = psnr.detach().cpu().numpy()
                step_results = {}
                step_results["eval/coarse_psnr"] = float(np.mean(psnr[:-1]))
                step_results["eval/fine_psnr"] = float(psnr[-1])
                step_results["eval/avg_psnr"] = float(np.mean(psnr))
                run = wandb.log(step_results, step=step)

            torch.save(model.state_dict(), os.path.join(wandb.run.dir, "model.pt"))
            torch.save(optimizer.state_dict(), os.path.join(wandb.run.dir, "optim.pt"))

        torch.save(model.state_dict(), os.path.join(wandb.run.dir, "model.pt"))
        torch.save(optimizer.state_dict(), os.path.join(wandb.run.dir, "optim.pt"))


def eval_model(config, model, data):
    model.eval()
    rays, pixels = next(data)
    with torch.no_grad():
        comp_rgb, _, _ = model(rays)
    pixels = pixels.to(config.device)
    model.train()
    return torch.tensor(
        [mse_to_psnr(torch.mean((rgb - pixels[..., :3]) ** 2)) for rgb in comp_rgb]
    )


if __name__ == "__main__":
    config = get_config()
    train_model(config)
