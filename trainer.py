import os
import random
import wandb
import torch
import math
import pickle
import pandas as pd
from tqdm import tqdm
import torch.nn as nn
# from torch.optim import Adam, AdamW
# from torch.optim.lr_scheduler import MultiStepLR
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from utils.utils import *
from utils.functions import *
from model import VRPModel
from envs.env import MTVRPEnv, get_dataloader
from envs.transformer import StateAugmentation


def metric2str(metric_label, metric_list):
    metric_info = '|'.join([f'{metric_label[i]} {metric_list[i]:.4f}' for i in range(len(metric_label))])
    return metric_info

def transform_dict_to_mean(dict_):
    for k, v in dict_.items():
        dict_[k] = torch.tensor(v).mean().item()

def warm_up(step, warmup_step, lr, optimizer):
    # alpha = (step + 1.) / args.warmup_step
    # for group_idx in range(len(self.optimizer.param_groups)):
    #     self.optimizer.param_groups[group_idx]['lr'] = alpha * self.optimizer.param_groups[group_idx]['initial_lr']
    # alpha = float(step + 1) / args.warmup_step
    # current_lr = (args.lr - 1e-7) * alpha + 1e-7
    # for param_group in self.optimizer.param_groups:
    #     param_group['lr'] = current_lr
    alpha = float(step + 1) / warmup_step
    current_lr = (lr - 1e-7) * alpha + 1e-7
    for param_group in optimizer.param_groups:
        param_group['lr'] = current_lr

def cal_model_size(model, args):
    param_count = sum(param.nelement() for param in model.parameters())
    buffer_count = sum(buffer.nelement() for buffer in model.buffers())
    args.log('Total number of parameters: {}'.format(param_count))
    args.log('Total number of buffer elements: {}'.format(buffer_count))
    # #
    # param_size = 0
    # for param in model.parameters():
    #     param_size += param.nelement() * param.element_size()
    # buffer_size = 0
    # for buffer in model.buffers():
    #     buffer_size += buffer.nelement() * buffer.element_size()
    # size_all_mb = (param_size + buffer_size) / 1024 ** 2
    # args.log('model size: {:.3f}MB'.format(size_all_mb))

class VRPTrainer:
    def __init__(self,args):
        self.args=args
        # cuda
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        # Main Components
        self.model = VRPModel(args) # args.log(f'current device:{self.model.device}')
        cal_model_size(self.model, args) #
        # exit(0)
        self.env = MTVRPEnv(**args.env)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr = args.optimizer_params['optimizer']['lr'],
            weight_decay=args.optimizer_params['optimizer']['weight_decay'] if 'weight_decay' in args.optimizer_params['optimizer'] else 0,
        )
        if args.optimizer_params['scheduler']['name'] == 'MultiStepLR':
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer,
                milestones = args.optimizer_params['scheduler']['milestones'],
                gamma = args.optimizer_params['scheduler']['gamma']
            )
        else:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=int(args.optimizer_params['scheduler']['T_max_epoch']*args.trainer_params['train_step']),
                eta_min=args.optimizer_params['scheduler']['eta_min_factor'] * args.optimizer_params['optimizer']['lr']
            )
        # Restore
        self.start_epoch = 1
        model_load = args.trainer_params['model_load']
        if model_load['enable']:
            checkpoint_fullname = '{path}/checkpoint-{epoch}.pt'.format(**model_load)
            checkpoint = torch.load(checkpoint_fullname, map_location='cuda')
            model_state_dict = checkpoint['model_state_dict']
            if args.tune_p: # only load encoder and decoder
                encoder_params = {k[len("encoder."):]:v for k,v in model_state_dict.items() if k.startswith('encoder.')}
                decoder_params = {k[len("decoder."):]:v for k,v in model_state_dict.items() if k.startswith('decoder.')}
                self.model.encoder.load_state_dict(encoder_params)
                self.model.decoder.load_state_dict(decoder_params)
                for param in self.model.encoder.parameters():
                    param.requires_grad = False
                for param in self.model.decoder.parameters():
                    param.requires_grad = False
                args.log(f'Freeze Encoder and Decoder.')
            else:
                self.model.load_state_dict(model_state_dict, strict=True)
                self.start_epoch = 1 + model_load['epoch']
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.scheduler.last_epoch = model_load['epoch']-1
            args.log(f'Saved Model Loaded from {checkpoint_fullname}.')
            # set random state
            if not args.ddp: # ddp
                self.env.__setstate__(checkpoint['env_state_dict'])
                torch.set_rng_state(checkpoint['rng_state_dict']['torch.rng_state'].cpu())
                torch.cuda.set_rng_state(checkpoint['rng_state_dict']['torch.cuda.rng_state'].cpu())
                random.setstate(checkpoint['rng_state_dict']['random.state'])
            else:
                self.env.__setstate__(checkpoint['env_state_dict'], set_seed=False)
        # ddp model
        if args.ddp:
            torch.distributed.barrier()
            self.model = DistributedDataParallel(self.model)  # , find_unused_parameters=True)
            # assure different ddp workers have the same param
            for param in self.model.parameters(): # if all ddp.worker load the same ckpt, this may be redundant
                dist.broadcast(param.data, src=0)
            args.log(f'use ddp, current device:{torch.cuda.current_device()}')
        # test dataloader
        if args.test:
            self.test_dataloader = get_dataloader(
                self.env.dataset(phase='test', data_size=args.env['test_episodes']),
                batch_size=args.env['test_batch_size'],
                ddp=args.ddp,
                num_workers=args.num_workers
            )
            self.augmentation = StateAugmentation()
        # utility
        self.time_estimator = TimeEstimator()

    def run(self):
        args = self.args
        self.time_estimator.reset(self.start_epoch)
        # test before train
        if args.test: self.test(self.start_epoch - 1)
        if args.test_only: exit(0)
        # begin train
        step_i = 0 # args.n_set+list(reversed(args.n_set))[1:-1]
        # n_set = torch.tensor(args.n_set).repeat_interleave(128)
        for epoch in range(self.start_epoch, args.trainer_params['epochs']+1):
            args.log('=================================================================')
            if args.wandb != '' and not args.mute: wandb.log({f'lr': self.optimizer.param_groups[0]['lr']}, step=epoch) #
            ########################## one epoch ###########################
            # prepare
            # dataloader = get_dataloader(
            #     self.env.dataset(data_size=args.trainer_params['train_episodes'], phase='train'),
            #     batch_size=args.trainer_params['train_batch_size'], num_workers=args.num_workers
            # )
            # train_pbar = tqdm(dataloader, bar_format='{desc}|{elapsed}+{remaining}|{n_fmt}/{total_fmt}', leave=False)
            self.model.train()
            train_pbar = tqdm(range(args.trainer_params['train_step']) , bar_format='{desc}|{elapsed}+{remaining}|{n_fmt}/{total_fmt}', leave=False)
            train_label = f"Train|Epoch{str(epoch).zfill(3)}/{str(args.trainer_params['epochs']).zfill(3)}"
            all_metric = [] # metric: [score loss grad] # metric: [score loss grad, vrpbscore]
            for step in train_pbar: # task_param (p,n)
                if args.skip and step > 2: break
                self.optimizer.zero_grad()
                # warm_up lr
                if step_i+1 <= args.warmup_step:
                    warm_up(step_i, args.warmup_step, args.lr, self.optimizer)
                n_loc = args.n_size
                batch_size = args.batch_size
                self.env.generator.reset_n_loc(n_loc)
                td = self.env.reset(batch_size=batch_size).to('cuda') # sample
                if args.ddp: torch.distributed.barrier()
                out = self.model(td, self.env)
                reward = out["reward"].view(-1, batch_size) # repeat_num, batch_size
                log_likelihood = out["log_likelihood"].view(-1, batch_size)  # repeat_num, batch_size
                # loss
                advantage = reward - reward.mean(dim=0, keepdims=True) # repeat_num, batch_size
                loss = -(advantage * log_likelihood).mean()
                # Score
                max_pomo_reward, _ = reward.max(dim=0)  # bs   get best results from pomo
                score_mean = -max_pomo_reward.float().mean()  # negative sign to make positive value
                vrpb_score_mean = -max_pomo_reward[(td['p_s_tag'][:,:5] == torch.tensor([1, 0, 0, 0, 1])).all(dim=1)].float().mean() # TODO:
                vrptw_score_mean = -max_pomo_reward[(td['p_s_tag'][:,:5] == torch.tensor([1, 0, 1, 0, 0])).all(dim=1)].float().mean() # TODO:
                if args.ddp: torch.distributed.barrier()
                loss.backward()
                grad_norms, grad_norms_clipped = clip_grad_norms(self.optimizer.param_groups, 1.)
                self.optimizer.step()
                # CosineAnnealingLR decay
                if args.optimizer_params['scheduler']['name'] == 'CosineAnnealingLR' and step_i<self.scheduler.T_max:
                    self.scheduler.step()
                # update step_i
                if step_i+1 <= args.warmup_step or (args.optimizer_params['scheduler']['name'] == 'CosineAnnealingLR' and step_i<self.scheduler.T_max):
                    step_i = step_i + 1
                # Log
                metric_list = [loss.item(), 
                               score_mean.item(), vrpb_score_mean.item(), vrptw_score_mean.item(),
                               grad_norms[0].item()]
                # metric_list = [loss.item(), score_mean.item(), 1.]
                all_metric.append(metric_list)
                metric_info = '|'.join([f'{args.metric_label[i]} {metric_list[i]:.4f}' for i in range(len(args.metric_label))])
                train_pbar.set_description(f"🙏> {train_label}|{metric_info}")
            if args.ddp: torch.distributed.barrier()
            # Log Once, for each epoch
            ## on each device
            metric_tensor = torch.tensor(all_metric).mean(dim=0)
            metric_list = metric_tensor.tolist()
            metric_info = '|'.join([f'{args.metric_label[i]} {metric_list[i]:.4f}' for i in range(len(args.metric_label))])
            elapsed = f"{tqdm.format_interval(train_pbar.format_dict['elapsed'])}"
            args.log(f"{train_label}|{elapsed}|{metric_info}|LR {self.optimizer.param_groups[0]['lr']:.2e}")
            ## on all devices
            if args.ddp:
                metric_tensor_ = metric_tensor.to('cuda')
                dist.reduce(metric_tensor_, dst=0)  # op=ReduceOp.SUM
                if args.rank == 0:
                    metric_avg = metric_tensor_ / dist.get_world_size()  # average over all workers
                    metric_info = '|'.join([f'{args.metric_label[i]} {metric_avg[i]:.4f}' for i in range(len(args.metric_label))])
                    args.log(f'***ddp_reduce*** {train_label}|{elapsed}|{metric_info}')
                    if args.wandb != '' and args.rank == 0:
                        wandb.log({f'{args.metric_label[i]}_train': metric_list[i] for i in range(len(args.metric_label))}, step= epoch) # epoch*2
                torch.distributed.barrier()
            elif args.wandb != '':
                wandb.log({f'{args.metric_label[i]}_train': metric_list[i] for i in range(len(args.metric_label))}, step= epoch) #  epoch*2
            # test during train
            if args.test and (epoch % args.env['test_interval'] == 0 or epoch in args.env['test_epoch']): self.test(epoch)
            ########################## one epoch end ###########################
            # MultiStepLR LR Decay
            if args.optimizer_params['scheduler']['name'] == 'MultiStepLR':
                self.scheduler.step()
            # Remain times
            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(epoch, args.trainer_params['epochs'])
            args.log("Epoch {:3d}/{:3d}: Time Est.: Elapsed[{}], Remain[{}]".format(epoch, args.trainer_params['epochs'], elapsed_time_str, remain_time_str))
            # Save Model
            if ((epoch == args.trainer_params['epochs']) or (epoch % args.trainer_params['model_save_interval']) == 0):
                if not args.mute:
                    args.log("Saving trained_model")
                    checkpoint_dict = {
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict() if not self.args.ddp else self.model.module.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'scheduler_state_dict': self.scheduler.state_dict(),
                        'env_state_dict': self.env.__getstate__(),
                        'rng_state_dict': {'torch.rng_state': torch.get_rng_state(),
                                           'torch.cuda.rng_state': torch.cuda.get_rng_state(),
                                           'random.state': random.getstate()}
                    }
                    torch.save(checkpoint_dict, '{}/checkpoint-{}.pt'.format(args.result_dir, epoch))
            # end of epoch
            if args.ddp: torch.distributed.barrier()
        args.log(" *** Training Done *** ")

    @torch.no_grad()
    def test(self, epoch):
        args = self.args
        self.model.eval()
        dataset_num = len(list(self.test_dataloader.keys()))
        # aug gap dict
        s_a8gap_dict = {i: [] for i in args.env['test_size']}
        p_a8gap_dict = {i: [] for i in args.env['test_problem']}
        d_a8gap_dict = {i: [] for i in args.env['test_distribution']}
        # gap dict
        s_gap_dict = {i: [] for i in args.env['test_size']}
        p_gap_dict = {i: [] for i in args.env['test_problem']}
        d_gap_dict = {i: [] for i in args.env['test_distribution']}
        tmp_test_metric_label = ['1g/0gap', '2ag/0aug_gap']
        # 按照 size 分类， 用于保存为excel
        s_a8gap_dict_excel = {
            i: dict(
                {'problem': args.env['test_problem']},
                **{j: [0.]*len(args.env['test_problem']) for j in args.env['test_distribution']}
            ) for i in args.env['test_size']
        }
        s_gap_dict_excel = {
            i: dict(
                {'problem': args.env['test_problem']},
                **{j: [0.]*len(args.env['test_problem']) for j in args.env['test_distribution']}
            ) for i in args.env['test_size']
        }
        problem_to_idx = {j: i for i,j in enumerate(args.env['test_problem'])}
        for data_idx, (dataset_name, test_dataloader) in enumerate(self.test_dataloader.items()):
            all_metric = []
            eval_label = f"Eval {dataset_name:7s} {str(data_idx).zfill(3)}/{str(dataset_num).zfill(3)} |Epoch{str(epoch).zfill(3)}/{str(args.trainer_params['epochs']).zfill(3)}|rank{args.rank}"
            eval_phar = tqdm(test_dataloader, bar_format='{desc}|{elapsed}+{remaining}|{n_fmt}/{total_fmt}', leave=False)
            # begin one dataset
            for step, inp in enumerate(eval_phar):
                if args.skip and step > 1: break
                batch_size = inp.batch_size[0]
                td = self.env.reset(td=inp.to('cuda'))
                td = self.augmentation(td) #=td.expand(8, batch_size).contiguous().view(8*batch_size,)
                if args.ddp: torch.distributed.barrier()
                out = self.model(td, self.env) # repeat_num, 8, batch_size
                all_reward = out["reward"].view(-1, self.augmentation.num_augment, batch_size)  # repeat_num, 8, batch_size
                all_reward, _ = all_reward.max(dim=0)  # aug, batch_size
                score = -all_reward[0, :].float() # batch_size
                aug_reward, _ = all_reward.max(dim=0)  # batch
                aug_score = -aug_reward.float() # batch
                # compute gap
                opt_score = inp['opt_cost'].to('cuda')  # cpu or cuda ?
                gap = ((score - opt_score) * 100 / opt_score).mean().item()
                aug_gap = ((aug_score - opt_score) * 100 / opt_score).mean().item()
                metric_list = [gap, aug_gap]
                # collection result
                all_metric.append(metric_list)
                # log
                metric_info = metric2str(tmp_test_metric_label, metric_list)
                eval_phar.set_description(f"🙏> {eval_label}|{metric_info}")
            # finish solve one dataset
            # log one dataset
            if args.ddp: torch.distributed.barrier()
            metric_mean = torch.tensor(all_metric).mean(dim=0).tolist()  # gap, aug_gap
            metric_info = metric2str(tmp_test_metric_label, metric_mean)
            elapsed = f"{tqdm.format_interval(eval_phar.format_dict['elapsed'])}"
            args.log(f"{eval_label}|{elapsed}|{metric_info}")
            size,problem,distribution = dataset_name.split('_')
            size = int(size)
            if args.ddp:  # compute average metric over all ddp workers
                # different workers may have different number of samples so the mean metric is not accurate!
                val_tensor = torch.tensor(metric_mean).cuda()
                dist.reduce(val_tensor, dst=0) # sum to 0
                if args.rank == 0:
                    num_workers = dist.get_world_size()
                    metric_avg = (val_tensor / num_workers).tolist()
                    metric_info = metric2str(tmp_test_metric_label, metric_avg)
                    args.log(f'***ddp_reduce*** {eval_label}|{elapsed}|{metric_info}')
                    # update dict
                    s_a8gap_dict[size].append(metric_avg[1])
                    p_a8gap_dict[problem].append(metric_avg[1])
                    d_a8gap_dict[distribution].append(metric_avg[1])
                    s_gap_dict[size].append(metric_avg[0])
                    p_gap_dict[problem].append(metric_avg[0])
                    d_gap_dict[distribution].append(metric_avg[0])
                    #
                    s_a8gap_dict_excel[size][distribution][problem_to_idx[problem]]=metric_avg[1]
                    s_gap_dict_excel[size][distribution][problem_to_idx[problem]]=metric_avg[0]
                torch.distributed.barrier()
            else:
                # update dict
                s_a8gap_dict[size].append(metric_mean[1])
                p_a8gap_dict[problem].append(metric_mean[1])
                d_a8gap_dict[distribution].append(metric_mean[1])
                s_gap_dict[size].append(metric_mean[0])
                p_gap_dict[problem].append(metric_mean[0])
                d_gap_dict[distribution].append(metric_mean[0])
                #
                s_a8gap_dict_excel[size][distribution][problem_to_idx[problem]] = metric_mean[1]
                s_gap_dict_excel[size][distribution][problem_to_idx[problem]] = metric_mean[0]
        if not args.mute:
            # wandb log all dataset
            metric_list = []
            for dict_ in [s_a8gap_dict, s_gap_dict, p_a8gap_dict, p_gap_dict, d_a8gap_dict, d_gap_dict]:
                metric_list.extend(torch.tensor(list(dict_.values())).cuda().mean(dim=1).tolist())
            metric_info = metric2str(args.test_metric_label, metric_list)
            args.log(metric_info)
            if args.wandb != '':
                wandb.log(
                    {f'{args.test_metric_label[i]}': metric_list[i] for i in range(len(args.test_metric_label))},
                    step=epoch
                ) # epoch*2
            # save to excel
            save_file_name = [
                os.path.join(args.result_dir, f'a8gap_{epoch}.xlsx'),
                os.path.join(args.result_dir, f'gap_{epoch}.xlsx')
            ]
            for file_name, sheet_ in zip(save_file_name, [s_a8gap_dict_excel, s_gap_dict_excel]):
                writer = pd.ExcelWriter(file_name)
                for sheet_name_, sheet_data_ in sheet_.items():
                    pd.DataFrame(sheet_data_).to_excel(writer, sheet_name=str(sheet_name_))
                writer.close()
        if args.ddp: torch.distributed.barrier()

    @torch.no_grad()
    def test_multi_epochs(self):
        args = self.args
        test_epochs = list(range(100,1201,400)) # +[1200]
        # path_id = ['2024-0711-2331', '2024-0714-0024'] # 800
        # path_id_ = '2024-0723-1110' # pomo 1
        # path_id = ['2024-0723-1007', '2024-0723-1630'] # 100 pomo2 1gpu
        path_id_ = '2024-0723-1057' #  pomo2 3gpu
        for epoch in test_epochs:
            # path_id_ = path_id[0] if epoch <= 100 else path_id[1]
            model_load = {}
            model_load['path'] = os.path.dirname(os.path.abspath(__file__)) + f'/result/{path_id_}'
            model_load['epoch'] = epoch
            checkpoint_fullname = '{path}/checkpoint-{epoch}.pt'.format(**model_load)
            checkpoint = torch.load(checkpoint_fullname, map_location='cuda')
            model_state_dict = checkpoint['model_state_dict']
            self.model.module.load_state_dict(model_state_dict) if args.ddp else self.model.load_state_dict(model_state_dict)
            args.log(f'Saved Model Loaded from {checkpoint_fullname}.')
            # test
            if args.ddp: torch.distributed.barrier()
            self.test(epoch)
