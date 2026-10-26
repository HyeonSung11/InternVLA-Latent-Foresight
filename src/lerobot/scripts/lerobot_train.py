#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
import os
import time
from contextlib import nullcontext
from pprint import pformat
from typing import Any

import torch
import multiprocessing as mp
from accelerate import Accelerator
from accelerate.utils import send_to_device
from termcolor import colored
from torch.optim import Optimizer

from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_dataset, make_dataloader
from lerobot.datasets.utils import cycle
from lerobot.optim.factory import make_optimizer_and_scheduler
from lerobot.policies.factory import make_policy
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.rl.wandb_utils import WandBLogger
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.logging_utils import AverageMeter, MetricsTracker, format_time
from lerobot.utils.random_utils import set_seed
from lerobot.utils.train_utils import (
    get_step_checkpoint_dir,
    get_step_identifier,
    load_training_state,
    save_checkpoint,
    update_last_checkpoint,
)
from lerobot.utils.utils import (
    format_big_number,
    has_method,
    init_logging,
    gather_object, 
)

def update_policy(
    train_metrics: MetricsTracker,
    policy: PreTrainedPolicy,
    batch: Any,
    optimizer: Optimizer,
    grad_clip_norm: float,
    accelerator: Accelerator,
    lr_scheduler=None,
    lock=None,
) -> tuple[MetricsTracker, dict]:
    """
    Performs a single training step to update the policy's weights.

    This function executes the forward and backward passes, clips gradients, and steps the optimizer and
    learning rate scheduler. Accelerator handles mixed-precision training automatically.

    Args:
        train_metrics: A MetricsTracker instance to record training statistics.
        policy: The policy model to be trained.
        batch: A batch of training data.
        optimizer: The optimizer used to update the policy's parameters.
        grad_clip_norm: The maximum norm for gradient clipping.
        accelerator: The Accelerator instance for distributed training and mixed precision.
        lr_scheduler: An optional learning rate scheduler.
        lock: An optional lock for thread-safe optimizer updates.

    Returns:
        A tuple containing:
        - The updated MetricsTracker with new statistics for this step.
        - A dictionary of outputs from the policy's forward pass, for logging purposes.
    """
    start_time = time.perf_counter()
    policy.train()

    # 09-15: accelerator.accumulate() 로 감싸야 gradient accumulation 이 실제로 동작한다.
    # accumulation_steps=1(기본)이면 sync_gradients 가 항상 True 라 기존 동작과 완전히 동일하다.
    # backward 안에서 loss 가 accumulation_steps 로 나뉘고, 옵티마이저는 마지막 micro-step 에서만 갱신된다.
    with accelerator.accumulate(policy):
        with accelerator.autocast():
            loss, output_dict = policy.forward(batch)

        accelerator.backward(loss)

        if accelerator.sync_gradients:
            # Clip gradients if specified
            if grad_clip_norm > 0:
                grad_norm = accelerator.clip_grad_norm_(policy.parameters(), grad_clip_norm)
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    policy.parameters(), float("inf"), error_if_nonfinite=False
                )
        else:
            grad_norm = torch.zeros((), device=loss.device)

        # Optimizer step
        with lock if lock is not None else nullcontext():
            optimizer.step()

        optimizer.zero_grad()

    # Step through pytorch scheduler at every batch instead of epoch
    if lr_scheduler is not None:
        lr_scheduler.step()

    # Update internal buffers if policy has update method
    if has_method(accelerator.unwrap_model(policy, keep_fp32_wrapper=True), "update"):
        accelerator.unwrap_model(policy, keep_fp32_wrapper=True).update()

    train_metrics.loss = loss.item()
    if "loss_action" in output_dict:
        train_metrics.loss_action = output_dict["loss_action"]
    if "loss_video" in output_dict:
        train_metrics.loss_video = output_dict["loss_video"]
    if "loss_vqa" in output_dict:
        train_metrics.loss_vqa = output_dict["loss_vqa"]
    if "loss_fast" in output_dict:
        train_metrics.loss_fast = output_dict["loss_fast"]
    if "loss_subtask" in output_dict:
        train_metrics.loss_subtask = output_dict["loss_subtask"]
    if "loss_prior" in output_dict:  # Foresight Residual Prior (anchored 모드) 의 g 지도 손실
        train_metrics.loss_prior = output_dict["loss_prior"]
    if "loss_tpc" in output_dict:  # 다중 표본 FM 의 대조쌍 일관성 항 (fm_tpc_weight>0 일 때만 non-zero)
        train_metrics.loss_tpc = output_dict["loss_tpc"]
    if "loss_spatial" in output_dict:  # Spatial query tokens (9.29) 의 기하 교사 회귀 손실 (가중·warm-up 포함)
        train_metrics.loss_spatial = output_dict["loss_spatial"]
    train_metrics.grad_norm = grad_norm.item()
    train_metrics.lr = optimizer.param_groups[0]["lr"]
    train_metrics.update_s = time.perf_counter() - start_time
    return train_metrics, output_dict


@parser.wrap()
def train(cfg: TrainPipelineConfig, accelerator: Accelerator | None = None):
    """
    Main function to train a policy.

    This function orchestrates the entire training pipeline, including:
    - Setting up logging, seeding, and device configuration.
    - Creating the dataset, evaluation environment (if applicable), policy, and optimizer.
    - Handling resumption from a checkpoint.
    - Running the main training loop, which involves fetching data batches and calling `update_policy`.
    - Periodically logging metrics, saving model checkpoints, and evaluating the policy.
    - Pushing the final trained model to the Hugging Face Hub if configured.

    Args:
        cfg: A `TrainPipelineConfig` object containing all training configurations.
        accelerator: Optional Accelerator instance. If None, one will be created automatically.
    """
    # mp.set_start_method("spawn", force=True)
    
    # Create Accelerator if not provided
    # It will automatically detect if running in distributed mode or single-process mode
    # We set step_scheduler_with_optimizer=False to prevent accelerate from adjusting the lr_scheduler steps based on the num_processes
    # We set find_unused_parameters=True to handle models with conditional computation
    if accelerator is None:
        from accelerate.utils import DistributedDataParallelKwargs

        # 09-15: 2-GPU DDP 성능 조정을 환경변수로 노출 (기본값은 기존 동작 그대로).
        #   find_unused_parameters=True 는 매 iteration autograd 그래프 전체를 훑고 DDP 최적화를 막는다.
        #   이 모델은 미사용 파라미터가 없다("did not find any unused parameters" 경고)므로 끌 수 있고,
        #   gradient checkpointing 을 쓸 때는 static_graph=True 가 PyTorch 권장 설정이다.
        #   gradient_as_bucket_view=True 는 gradient 버킷을 별도 복사하지 않아 VRAM 을 아낀다.
        _b = lambda k, d: os.environ.get(k, str(d)).lower() in ("1", "true", "yes")
        ddp_kwargs = DistributedDataParallelKwargs(
            find_unused_parameters=_b("DDP_FIND_UNUSED", True),
            static_graph=_b("DDP_STATIC_GRAPH", False),
            gradient_as_bucket_view=_b("DDP_BUCKET_VIEW", False),
        )
        logging.info(
            f"[DDP] find_unused_parameters={ddp_kwargs.find_unused_parameters} "
            f"static_graph={ddp_kwargs.static_graph} gradient_as_bucket_view={ddp_kwargs.gradient_as_bucket_view}"
        )
        # 09-15: 진짜 gradient accumulation. accelerate launch 의 --gradient_accumulation_steps 는
        # DeepSpeed 전용이라 반영되지 않으므로, 여기서 직접 넘긴다. 기본 1 = 기존 동작과 동일.
        _accum_steps = max(int(os.environ.get("REAL_GRAD_ACCUM", "1")), 1)
        accelerator = Accelerator(
            step_scheduler_with_optimizer=False,
            gradient_accumulation_steps=_accum_steps,
            kwargs_handlers=[ddp_kwargs],
        )

    init_logging(accelerator=accelerator)

    # 09-15: 실제 유효배치를 반드시 로그에 남긴다.
    #   accelerate launch --gradient_accumulation_steps 는 DeepSpeed 를 쓸 때만 반영된다(공식 도움말).
    #   또 이 스크립트의 update_policy 는 accelerator.accumulate() 를 쓰지 않고 매 iteration
    #   optimizer.step() 을 호출하므로, 누적은 실제로 일어나지 않는다 → 유효배치 = batch_size × num_processes.
    #   이 사실을 모르고 "유효배치 16" 으로 기록해온 이력이 있어 런타임 계측으로 못박는다.
    _accum = accelerator.gradient_accumulation_steps  # REAL_GRAD_ACCUM 로 설정 (기본 1)
    _eff = cfg.batch_size * _accum * accelerator.num_processes
    logging.info(
        f"[BATCH] batch_size={cfg.batch_size} × accum={_accum} × procs={accelerator.num_processes} "
        f"→ 실제 유효배치 {_eff}"
    )

    # Determine if this is the main process (for logging and checkpointing)
    # When using accelerate, only the main process should log to avoid duplicate outputs
    is_main_process = accelerator.is_main_process

    # Only log on main process
    if is_main_process:
        logging.info(pformat(cfg.to_dict()))
    
    try:
        cfg.validate()
    except FileExistsError:
        if is_main_process:
            raise
        logging.warning(f"Ignoring existing output_dir on non-main process: {cfg.output_dir}")
    accelerator.wait_for_everyone()

    # Initialize wandb only on main process
    if cfg.wandb.enable and cfg.wandb.project and is_main_process:
        wandb_logger = WandBLogger(cfg)
    else:
        wandb_logger = None
        if is_main_process:
            logging.info(colored("Logs will be saved locally.", "yellow", attrs=["bold"]))

    # Pin this rank to Accelerate's device before any model/CUDA initialization.
    device = accelerator.device
    if device.type == "cuda":
        torch.cuda.set_device(device.index if device.index is not None else 0)
    cfg.policy.device = str(device)

    if cfg.seed is not None:
        set_seed(cfg.seed, accelerator=accelerator)

    # # Use accelerator's device
    # device = accelerator.device
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    # Dataset loading synchronization: main process downloads first to avoid race conditions
    if is_main_process:
        logging.info("Creating dataset")
        dataset, data_stats = make_dataset(cfg)

    accelerator.wait_for_everyone()

    # Now all other processes can safely load the dataset
    if not is_main_process:
        dataset, data_stats = make_dataset(cfg)

    accelerator.wait_for_everyone()

    if accelerator.num_processes>1:
        all_data_stats = gather_object(data_stats, accelerator)
    else:
        all_data_stats = [data_stats]
        
    if is_main_process:
        merged_data_stats = {}
        for rank_stats in all_data_stats:
            merged_data_stats.update(rank_stats)
        data_stats = merged_data_stats
    else:
        data_stats = None

    accelerator.wait_for_everyone()

    if is_main_process:
        logging.info("Creating policy")
    policy = make_policy(
        cfg=cfg.policy,
    )

    # Wait for all processes to finish policy creation before continuing
    accelerator.wait_for_everyone()

    if is_main_process:
        logging.info("Creating optimizer and scheduler")
    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)

    step = 0  # number of policy updates (forward + backward + optim)

    if cfg.resume:
        step, optimizer, lr_scheduler = load_training_state(cfg.checkpoint_path, optimizer, lr_scheduler)

    if cfg.dataset.dist_loading:
        num_frames = sum(gather_object(dataset.num_frames, accelerator))
        num_episodes = sum(gather_object(dataset.num_episodes, accelerator))
    else:
        num_frames = dataset.num_frames
        num_episodes = dataset.num_episodes
    num_processes = accelerator.num_processes
    effective_bs = cfg.batch_size * num_processes

    if is_main_process:
        logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")
        logging.info(f"{cfg.steps=} ({format_big_number(cfg.steps)})")
        logging.info(f"\033[91m\033[1mnum_frames={num_frames} ({format_big_number(num_frames)})\033[0m")
        logging.info(f"\033[91m\033[1mnum_episodes={num_episodes} ({format_big_number(num_episodes)})\033[0m")
        logging.info(f"Effective batch size: {cfg.batch_size} x {num_processes} = {effective_bs}")
        logging.info(f"policy info:\n{policy}")

    # create dataloader for offline training
    dataloader, dl_self_managed = make_dataloader(cfg, dataset)

    # When the dataloader manages its own sampling (MixedMultimodalDataset,
    # dist_loading), it must NOT be wrapped by accelerator.prepare and
    # batches are sent to device manually in the training loop.
    accelerator.wait_for_everyone()
    if cfg.dataset.dist_loading:
        policy, optimizer, lr_scheduler = accelerator.prepare(
            policy, optimizer, lr_scheduler
        )
    else:
        policy, optimizer, dataloader, lr_scheduler = accelerator.prepare(
            policy, optimizer, dataloader, lr_scheduler
        )
    dl_iter = cycle(dataloader)

    policy.train()

    if cfg.policy.type == "internvla_a1_5":
        train_metrics = {
            "loss": AverageMeter("loss", ":.3f"),
            "loss_action": AverageMeter("loss_action", ":.3f"),
            "grad_norm": AverageMeter("grdn", ":.3f"),
            "lr": AverageMeter("lr", ":0.1e"),
            "update_s": AverageMeter("updt_s", ":.3f"),
            "dataloading_s": AverageMeter("data_s", ":.3f"),
        }
    else:
        train_metrics = {
            "loss": AverageMeter("loss", ":.3f"),
            "grad_norm": AverageMeter("grdn", ":.3f"),
            "lr": AverageMeter("lr", ":0.1e"),
            "update_s": AverageMeter("updt_s", ":.3f"),
            "dataloading_s": AverageMeter("data_s", ":.3f"),
        }
  
    if getattr(cfg.policy, "enable_vqa_loss", False):
        train_metrics["loss_vqa"] = AverageMeter("loss_vqa", ":.3f")
        train_metrics["loss_action"] = AverageMeter("loss_action", ":.3f")
    
    if cfg.policy.type == "internvla_a1_5":
        train_metrics["loss_video"] = AverageMeter("loss_video", ":.3f")

    if cfg.policy.type == "internvla_a1_5":
        train_metrics["loss_fast"] = AverageMeter("loss_fast", ":.3f")
        train_metrics["loss_subtask"] = AverageMeter("loss_subtask", ":.3f")
        train_metrics["loss_prior"] = AverageMeter("loss_prior", ":.3f")
        train_metrics["loss_tpc"] = AverageMeter("loss_tpc", ":.3f")
        train_metrics["loss_spatial"] = AverageMeter("loss_spatial", ":.3f")


    # Use effective batch size for proper epoch calculation in distributed training
    effective_batch_size = cfg.batch_size * accelerator.num_processes
    train_tracker = MetricsTracker(
        effective_batch_size,
        num_frames,
        num_episodes,
        train_metrics,
        initial_step=step,
        accelerator=accelerator,
    )

    if is_main_process:
        logging.info("Start offline training on a fixed dataset")
        training_start_time = time.perf_counter()

    for _ in range(step, cfg.steps):
        start_time = time.perf_counter()
        batch = next(dl_iter)
        if cfg.dataset.dist_loading or dl_self_managed:
            batch = send_to_device(batch, accelerator.device, non_blocking=True)
        train_tracker.dataloading_s = time.perf_counter() - start_time

        train_tracker, output_dict = update_policy(
            train_tracker,
            policy,
            batch,
            optimizer,
            cfg.optimizer.grad_clip_norm,
            accelerator=accelerator,
            lr_scheduler=lr_scheduler,
        )

        # Note: eval and checkpoint happens *after* the `step`th training update has completed, so we
        # increment `step` here.
        step += 1
        train_tracker.step()
        is_log_step = cfg.log_freq > 0 and step % cfg.log_freq == 0 and is_main_process
        is_saving_step = step % cfg.save_freq == 0 or step == cfg.steps

        if is_log_step:
            avg_update_time = train_tracker.update_s.avg if hasattr(train_tracker.update_s, 'avg') else train_tracker.update_s.val
            steps_per_second = 1.0 / avg_update_time if avg_update_time > 0 else 0

            elapsed_time = time.perf_counter() - training_start_time if training_start_time else 0
            remaining_steps = cfg.steps - step
            estimated_remaining_time = remaining_steps * avg_update_time if avg_update_time > 0 else 0

            elapsed_str = format_time(elapsed_time)
            remaining_str = format_time(estimated_remaining_time)

            logging.info(f" \033[92m\033[1m{elapsed_str} << {remaining_str}\033[0m | \033[96m\033[1m{steps_per_second:.2f} iters/s\033[0m | {train_tracker}")
            if wandb_logger:
                wandb_log_dict = train_tracker.to_dict()
                if output_dict:
                    wandb_log_dict.update(output_dict)
                wandb_logger.log_dict(wandb_log_dict, step)
            train_tracker.reset_averages()

        if cfg.save_checkpoint and is_saving_step:
            if is_main_process:
                logging.info(f"Checkpoint policy after step {step}")
                checkpoint_dir = get_step_checkpoint_dir(cfg.output_dir, cfg.steps, step)
                logging.info(colored("Checkpoint saved at:", "cyan", attrs=["bold"]) + f" {checkpoint_dir}")
                save_checkpoint(
                    checkpoint_dir=checkpoint_dir,
                    step=step,
                    cfg=cfg,
                    policy=accelerator.unwrap_model(policy),
                    optimizer=optimizer,
                    scheduler=lr_scheduler,
                    data_stats=data_stats, 
                )
                update_last_checkpoint(checkpoint_dir)
                if wandb_logger:
                    wandb_logger.log_policy(checkpoint_dir)

            accelerator.wait_for_everyone()

    if is_main_process:
        logging.info("End of training")

        if cfg.policy.push_to_hub:
            unwrapped_policy = accelerator.unwrap_model(policy)
            unwrapped_policy.push_model_to_hub(cfg)

    # Properly clean up the distributed process group
    accelerator.wait_for_everyone()
    accelerator.end_training()


def main():
    register_third_party_plugins()
    train()


if __name__ == "__main__":
    main()
