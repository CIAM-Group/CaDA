import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # PSDNCO for utils
import time
import yaml
import json
import random
import torch
import wandb
import logging
import argparse
# import numpy as np

from trainer import VRPTrainer as Trainer
from utils.utils import copy_all_src
##########################################################################################
# torch.multiprocessing.set_start_method("spawn", force=True)

def init_seeds():
    random.seed(args.seed)
    # np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

def get_logger():
    args.result_dir = os.path.join('result', args.start_time)
    os.makedirs(args.result_dir, exist_ok=True)
    args.log_file = os.path.join(args.result_dir, f'log_{args.rank}.txt')
    logging.basicConfig(filename=args.log_file, format='%(asctime)-15s %(message)s')
    logger_ = logging.getLogger()
    logger_.setLevel(logging.INFO)
    console = logging.StreamHandler(sys.stdout)
    args.mute = (args.ddp and args.rank !=0)
    if not args.mute: logger_.addHandler(console)
    return logger_

def set_test_params():
    args.env['test_size']=[args.all_test_size[idx] for idx in args.env['test_size_idx']]
    args.env['test_problem']=[args.all_test_problem[idx] for idx in args.env['test_problem_idx']]
    args.env['test_distribution']=[ args.all_test_distribution[idx] for idx in args.env['test_distribution_idx']]

    size_l=['s/{:02d}-{}-a8gap%'.format(idx, args.all_test_size[idx]) for idx in args.env['test_size_idx']]
    size_l+=['s/{:02d}-{}-gap%'.format(idx+len(args.all_test_size), args.all_test_size[idx]) for idx in args.env['test_size_idx']]
    pro_l=['p/{:02d}-{}-a8gap%'.format(idx, args.all_test_problem[idx]) for idx in args.env['test_problem_idx']]
    pro_l+=['p/{:02d}-{}-gap%'.format(idx+len(args.all_test_problem), args.all_test_problem[idx]) for idx in args.env['test_problem_idx']]
    distr_l=['d/{:02d}-{}-a8gap%'.format(idx, args.all_test_distribution[idx]) for idx in args.env['test_distribution_idx']]
    distr_l+=['d/{:02d}-{}-gap%'.format(idx+len(args.all_test_distribution), args.all_test_distribution[idx]) for idx in args.env['test_distribution_idx']]
    args.test_metric_label = size_l + pro_l + distr_l

def main():
    init_seeds()  # no, do not need same seed # all ddp.workers use the same seed to produce the same initial param value
    trainer = Trainer(args)
    if not args.mute: copy_all_src(args.result_dir)
    if args.ddp: torch.distributed.barrier()
    args.log('finish copy_all_src.')
    trainer.run()
    # trainer.test_multi_epochs()

if __name__ == "__main__":
    # get params
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=64, help="training batch_size per gpu")
    parser.add_argument("--n_size", type=int, default=50, help="training n size")
    parser.add_argument("--num_workers", type=int, default=0, help="num_workers") # must set to 0
    parser.add_argument("--seed", type=int, default=7, help="seed")
    parser.add_argument("--resume", dest='resume', action='store_true', help="load model")
    parser.add_argument("--epoch", type=int, help="model checkpoint epoch")
    parser.add_argument("--path_id", type=str, help="model checkpoint folder id")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument('--wandb', type=str, default='', help='wandb_id https://wandb.ai/')
    parser.add_argument('--warmup', dest='warmup', action='store_true', help="warm up")
    parser.add_argument('--warmup_steps', type=int, default=3140, help="warm up steps") # 157*2=314 157*5=785 157*10=1570 157*20=3140
    parser.add_argument('--test', dest='test', action='store_true', help="test during train")
    parser.add_argument('--test_lib', dest='test_lib', action='store_true', help="test on lib")
    # parser.add_argument('--use_p', dest='use_p', action='store_true', help="use prompt")
    parser.add_argument('--skip', dest='skip', action='store_true', help="every epoch only 3 step")
    parser.add_argument('--test_only', dest='test_only', action='store_true', help="test_only")
    parser.add_argument('--warmup_step', type=int, default=-1, help="warm up steps")
    parser.add_argument('--tune_p', dest='tune_p', action='store_true', help="only tune prompt")
    # parser.add_argument('--use_p', dest='use_p', action='store_true', help="use_p")
    # parser.add_argument('--two_b', dest='two_b', action='store_true', help="two_b")
    # parser.add_argument('--no_log', dest='no_log', action='store_true', help="no_log")
    args = parser.parse_args()
    # get time for result log
    args.start_time = time.strftime("%Y-%m%d-%H%M", time.localtime())
    # load config.yaml
    with open('config.yaml', "r") as f:
        cfg_yaml = yaml.safe_load(f)
    for key, value in cfg_yaml.items():
        assert not hasattr(args, key), f"Cfg key already exists {key}"
        setattr(args, key, value)
    # ddp
    args.world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1  # num of ddp workers
    args.rank = int(os.environ['RANK']) if 'RANK' in os.environ else 0  # global rank of current ddp worker not use beacuse only use one machine
    args.local_rank = int(os.environ["LOCAL_RANK"]) if "LOCAL_RANK" in os.environ else 0 # local rank is the gpu id
    args.ddp = "LOCAL_RANK" in os.environ.keys()  # running in dist mode
    if args.ddp:
        torch.cuda.set_device(args.local_rank)  # set CUDA_VISIBLE_DEVICES to choose GPU
        torch.distributed.init_process_group(backend='nccl')
    else:
        torch.cuda.set_device(0)
    # set other params
    # args.env['seed'] = 493245309779795149
    assert args.n_size in [50,100]
    args.env['generator_params']['num_loc'] = args.n_size
    args.env['test_size_idx'] = [0] if args.n_size == 50 else [1] # 100
    args.model_params['sqrt_embedding_dim'] = args.model_params['embedding_dim'] ** (1 / 2)
    # args.model_params['use_p'] = args.usb_p
    # args.model_params['two_b'] = args.two_b
    args.optimizer_params['optimizer']['lr'] = args.lr
    args.optimizer_params['optimizer']['weight_decay'] = float(args.optimizer_params['optimizer']['weight_decay'])
    args.trainer_params['train_batch_size'] = args.batch_size
    args.trainer_params['model_load'] = {'enable': args.resume}
    if args.resume:
        args.trainer_params['model_load']['path'] = os.path.dirname(os.path.abspath(__file__)) + f'/result/{args.path_id}'
        args.trainer_params['model_load']['epoch'] = args.epoch
    # set task set
    args.p_set = ['C', 'O', 'L', 'B', 'TW', 'LB', 'TWO','LTW','OB','LO', 'TWB', 'LOB', 'LTWB', 'LTWO', 'TWOB', 'LTWOB'] # 16  
    # args.dist_set = [(0, 0), (1, 1)] + [(c, d) for c in [3, 5, 7] for d in [10, 30, 50]] # 11
    # args.n_set = list(range(50, 201, 5))  # 31
    # args.task_set = [(p, n, c, d) for p in args.p_set for n in args.n_set for (c, d) in args.dist_set] # 16*31*11
    # reset test metric label and test set
    set_test_params()
    logger = get_logger()
    do_not_log = ["p_set", "task_set", "dist_set", "n_set", "metric_label", "test_metric_label",
                  "all_test_size", "all_test_problem", "all_test_distribution"]
    if args.ddp:
        args.seed = args.seed + args.rank + (int(time.time()) if args.resume else 0)
    # args.env['seed'] = args.seed # set ebv's seed,
    logger.info(json.dumps({k:v for k,v in vars(args).items() if not k in do_not_log}, indent=4))
    if args.wandb != '' and (not args.ddp or args.rank == 0):
        wandb.init(project=f"vrp{args.n_size}", config=args.__dict__, id=args.wandb)  # , resume="must")
    args.log = logger.info
    main()

