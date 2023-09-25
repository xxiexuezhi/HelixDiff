from pathlib import Path
import score_sde_pytorch.losses as losses
import score_sde_pytorch.sampling as sampling
import argparse
from score_sde_pytorch.models.ema import ExponentialMovingAverage
import score_sde_pytorch.sde_lib as sde_lib
import torch
from torch.utils import tensorboard
from score_sde_pytorch.utils import save_checkpoint, restore_checkpoint, get_model, recursive_to
from dataset import ProteinDataset, PaddingCollate
import pickle as pkl
import yaml
from easydict import EasyDict
import time
from utils import random_mask_batch, get_condition_from_batch_hot4, selected_mask_batch,random_mask_given_all_except_4hotspots_batch
import shutil

import pickle

with open("protein_dataset_for_proteinclass_numpy_filter_rosettascore100_trainingdata_0512_64_64_for_protein_sgm.pkl","rb") as f:
    dataset= pickle.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('--resume', type=str, default="helix_1_64_64_hot4")
    args = parser.parse_args()
    mask_info ="18:28,50:60,100:110"
    with open(args.config, 'r') as f:
        config = EasyDict(yaml.safe_load(f))

    ss_constraints = True if config.data.num_channels == 8 else False
   # dataset = ProteinDataset(config.data.dataset_path, config.data.min_res_num,
    #                         config.data.max_res_num, ss_constraints)
    train_size = int(0.95 * len(dataset))
    test_size = len(dataset) - train_size
    train_ds, test_ds = torch.utils.data.random_split(dataset, [train_size, test_size],
                                                      generator=torch.Generator().manual_seed(config.seed))

    train_sampler = torch.utils.data.RandomSampler(
        train_ds,
        replacement=True,
        num_samples=config.training.n_iters * config.training.batch_size
    )
    train_dl = torch.utils.data.DataLoader(
        train_ds,
        sampler=train_sampler,
        batch_size=config.training.batch_size,
        collate_fn=PaddingCollate(config.data.max_res_num)
    )
    train_iter = iter(train_dl)

    test_sampler = torch.utils.data.RandomSampler(
        test_ds,
        replacement=True,
        num_samples=config.training.n_iters * config.training.batch_size
    )
    test_dl = torch.utils.data.DataLoader(
        test_ds,
        sampler=test_sampler,
        batch_size=config.training.batch_size,
        collate_fn=PaddingCollate(config.data.max_res_num)
    )
    test_iter = iter(test_dl)


    # Create directories for experimental logs
    if args.resume is not None:
        workdir = Path(args.resume)
        workdir.mkdir(exist_ok=True,parents=True)
    else:
        workdir = Path("training", Path(args.config).stem, time.strftime('%Y_%m_%d__%H_%M_%S', time.localtime()))
        workdir.mkdir(exist_ok=True,parents=True)
        # Save config to workdir
        shutil.copy(args.config, workdir.joinpath("config.yml"))

    sample_dir = workdir.joinpath("samples")
    sample_dir.mkdir(exist_ok=True)

    tb_dir = workdir.joinpath("tensorboard")
    tb_dir.mkdir(exist_ok=True)
    writer = tensorboard.SummaryWriter(tb_dir)

    # Initialize model.
    score_model = get_model(config)
    ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
    optimizer = losses.get_optimizer(config, score_model.parameters())
    state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)

    # Create checkpoints directory
    checkpoint_dir = workdir.joinpath("checkpoints")
    # Intermediate checkpoints to resume training after pre-emption in cloud environments
    checkpoint_meta_dir = workdir.joinpath("checkpoints-meta", "checkpoint.pth")
    checkpoint_dir.mkdir(exist_ok=True)
    checkpoint_meta_dir.parent.mkdir(exist_ok=True)
    # Resume training when intermediate checkpoints are detected
    if checkpoint_meta_dir.is_file():
        state = restore_checkpoint(checkpoint_meta_dir, state, config.device)
        initial_step = int(state['step'])
    else:
        initial_step = 0

    print(f"Starting from step {initial_step}...")

    # Setup SDEs
    if config.training.sde.lower() == 'vpsde':
        sde = sde_lib.VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
        sampling_eps = 1e-3
    elif config.training.sde.lower() == 'vesde':
        sde = sde_lib.VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max,
                            N=config.model.num_scales)
        sampling_eps = 1e-5
    else:
        raise NotImplementedError(f"SDE {config.training.sde} unknown.")

    # Build one-step training and evaluation functions
    optimize_fn = losses.optimization_manager(config)
    train_step_fn = losses.get_step_fn(sde, train=True, optimize_fn=optimize_fn)
    eval_step_fn = losses.get_step_fn(sde, train=False, optimize_fn=optimize_fn)

    # # # Building sampling functions
    if config.training.snapshot_sampling:
        sampling_shape = (config.training.batch_size, config.data.num_channels,
                          config.data.max_res_num, config.data.max_res_num)
        sampling_fn = sampling.get_sampling_fn(config, sde, sampling_shape, sampling_eps)

    for step in range(initial_step, config.training.n_iters + 1):
        batch = recursive_to(next(train_iter), config.device)
        # Execute one training step
        #batch = selected_mask_batch(batch, mask_info, config)
        # the below generted mask matrix only given 3 hotspot seq info. model need to generate all the rest
        batch = random_mask_given_all_except_4hotspots_batch(batch, config)
        #batch = random_mask_batch(batch, config)
        loss = train_step_fn(state, batch, condition=config.model.condition)
        if step % config.training.log_freq == 0:
            writer.add_scalar("training_loss", loss, step)

        # Save a temporary checkpoint to resume training after pre-emption periodically
        if step != 0 and step % config.training.snapshot_freq_for_preemption == 0:
            save_checkpoint(checkpoint_meta_dir, state)

        # Report the loss on an evaluation dataset periodically
        if step % config.training.eval_freq == 0:
            eval_batch = recursive_to(next(test_iter), config.device)
            # the below generted mask matrix only given 3 hotspot seq info. model need to generate all the rest
            eval_batch = random_mask_given_all_except_4hotspots_batch(eval_batch, config)
           #eval_batch = selected_mask_batch(eval_batch, mask_info, config)
            #eval_batch = random_mask_batch(eval_batch, config)
            eval_loss = eval_step_fn(state, eval_batch, condition=config.model.condition)
            writer.add_scalar("eval_loss", eval_loss.item(), step)

        # Save a checkpoint periodically and generate samples if needed
        if step != 0 and step % config.training.snapshot_freq == 0 or step == config.training.n_iters:
            # Save the checkpoint.
            save_step = step // config.training.snapshot_freq
            save_checkpoint(checkpoint_dir.joinpath(f'checkpoint_{save_step}.pth'), state)

            # Generate and save samples
            if config.training.snapshot_sampling:
                ema.store(score_model.parameters())
                ema.copy_to(score_model.parameters())
                condition = get_condition_from_batch_hot4(config, eval_batch)
                sample, n = sampling_fn(score_model, condition=condition)
                ema.restore(score_model.parameters())
                this_sample_dir = sample_dir.joinpath(f"iter_{step}")
                this_sample_dir.mkdir(exist_ok=True)

                with open(str(this_sample_dir.joinpath("sample.pkl")), "wb") as fout:
                    pkl.dump(sample.cpu(), fout)

                # save_grid(sample.cpu().numpy(), this_sample_dir.joinpath("sample.png"))

if __name__ == "__main__":
    main()
