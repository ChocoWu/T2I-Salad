import argparse
import torch
from omegaconf import OmegaConf
import numpy as np
import random
# from trainer import Trainer
from trainer import Trainer
from distributed import synchronize
import os
import torch.multiprocessing as multiprocessing

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if __name__ == "__main__":

    multiprocessing.set_start_method('spawn')

    parser = argparse.ArgumentParser()
    parser.add_argument("--DATA_ROOT", type=str, default="./DATA", help="path to DATA")
    parser.add_argument("--OUTPUT_ROOT", type=str, default="./OUTPUT", help="path to OUTPUT")

    parser.add_argument("--name", type=str, default="test", help="experiment will be stored in OUTPUT_ROOT/name")
    parser.add_argument("--seed", type=int, default=123, help="used in sampler")
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--yaml_file", type=str, default="configs/vg.yaml", help="paths to base configs.")
    parser.add_argument("--max_relations", type=int, default=10, help="the numer of the relations contained in one prompt.")

    parser.add_argument("--base_learning_rate", type=float, default=3e-5, help="")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="")
    parser.add_argument("--warmup_steps", type=int, default=1000, help="")
    parser.add_argument("--scheduler_type", type=str, default='cosine', help="cosine or constant")
    parser.add_argument("--batch_size", type=int, default=1, help="")
    parser.add_argument("--workers", type=int, default=1, help="")
    parser.add_argument("--official_ckpt_name", type=str, default="diffusion_pytorch_model.bin",
                        help="SD ckpt name and it is expected in DATA_ROOT, thus DATA_ROOT/official_ckpt_name must exists")
    parser.add_argument("--ckpt", type=lambda x: x if type(x) == str and x.lower() != "none" else None, default=None,
                        help=("If given, then it will start training from this ckpt"
                              "It has higher prioty than official_ckpt_name, but lower than the ckpt found in autoresuming (see trainer.py) "
                              "It must be given if inpaint_mode is true")
                        )

    parser.add_argument('--enable_ema', default=False, type=lambda x: x.lower() == "true")
    parser.add_argument("--ema_rate", type=float, default=0.9999, help="")
    parser.add_argument("--total_iters", type=int, default=500000, help="")
    parser.add_argument("--save_every_iters", type=int, default=5000, help="")
    parser.add_argument("--disable_inference_in_training", type=lambda x: x.lower() == "true", default=False,
                        help="Do not do inference, thus it is faster to run first a few iters. It may be useful for debugging ")

    args = parser.parse_args()
    assert args.scheduler_type in ['cosine', 'constant']

    n_gpu = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = n_gpu > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()

    config = OmegaConf.load(args.yaml_file)
    config.update(vars(args))
    print("n_gpu", n_gpu)
    config.total_batch_size = config.batch_size * n_gpu
    if args.inpaint_mode:
        config.model.params.inpaint_mode = True

    trainer = Trainer(config)
    synchronize()
    trainer.start_training()

    # CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 main.py  --yaml_file=configs/ade_sem.yaml  --DATA_ROOT=../../DATA   --batch_size=4
