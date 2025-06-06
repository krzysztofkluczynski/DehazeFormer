import os
import argparse
import json
import yaml
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm
import wandb

from utils import AverageMeter
from datasets.loader import PairLoader, CityScapesPairLoader
from models import *


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='dehazeformer-s', type=str, help='model name')
    parser.add_argument('--num_workers', default=16, type=int, help='number of workers')
    parser.add_argument('--no_autocast', action='store_false', default=True, help='disable autocast')
    parser.add_argument('--save_dir', default='./saved_models/', type=str, help='path to models saving')
    parser.add_argument('--ckpt', type=str, required=False, default=None, help='if resuming training, path to model checkpoint')
    parser.add_argument('--data_dir', default='./data/', type=str, help='path to dataset')
    parser.add_argument('--log_dir', default='./logs/', type=str, help='path to logs')
    parser.add_argument('--dataset', default='RESIDE-IN', type=str, help='dataset name')
    parser.add_argument('--config', default='configs/indoor/default.json', type=str, help='path to training config file')
    parser.add_argument('--gpu', default='0,1,2,3', type=str, help='GPUs used for training')


    parser.add_argument("--ignore_previous_best_score", action='store_true', default=False, \
        help="Save the best model based only on the score acvhieved in the current run (ignore pretrained model's score)")

    parser.add_argument('--n_epochs', default=None, type=int, help='override number of epochs to train')
    parser.add_argument('--batch_size', default=None, type=int, help='override number of epochs to train')

    # Wandb options
    parser.add_argument("--enable_wandb", action='store_true', default=False, help="Use Weights & Biases for logging")
    parser.add_argument("--wandb_team", type=str, default=None, help="Weights & Biases team name")
    parser.add_argument("--wandb_project", type=str, default=None, help="Weights & Biases project name")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="Weights & Biases current run name")
    parser.add_argument("--wandb_restore_ckpt", type=str, default=None, help="Weights & Biases checkpoint")
    parser.add_argument("--wandb_restore_run_path", type=str, default=None, help="Weights & Biases current run path")

    parser.add_argument("--wandb_sweep_config", type=str, help="Weights & Biases sweep config file path")
    parser.add_argument("--wandb_sweep_id", type=str, default=None, help="Weights & Biases sweep id. If provided, an existing sweep will be used instead of creating a new one.")

    args = parser.parse_args()
    return args


def train(train_loader, network, criterion, optimizer, scaler, epoch, wandb_run=None):
    losses = AverageMeter()

    torch.cuda.empty_cache()
    
    network.train()

    for i, batch in enumerate(tqdm(train_loader, desc=f'Training epoch {epoch}', leave=False)):
        source_img = batch['source'].cuda()
        target_img = batch['target'].cuda()

        with autocast(args.no_autocast):
            output = network(source_img)
            loss = criterion(output, target_img)

        losses.update(loss.item())


        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if wandb_run is not None:
            wandb.log({
                "train_loss": loss.item(),
                "epoch": epoch
            }, step=(epoch) * len(train_loader) + i)

    return losses.avg


def valid(val_loader, network, step, wandb_run=None):
    PSNR = AverageMeter()

    torch.cuda.empty_cache()

    network.eval()

    for i, batch in enumerate(tqdm(val_loader, desc='Validating', leave=False)):
        source_img = batch['source'].cuda()
        target_img = batch['target'].cuda()

        with torch.no_grad():                            # torch.no_grad() may cause warning
            output = network(source_img).clamp_(-1, 1)        

        mse_loss = F.mse_loss(output * 0.5 + 0.5, target_img * 0.5 + 0.5, reduction='none').mean((1, 2, 3))
        psnr = 10 * torch.log10(1 / mse_loss).mean()
        PSNR.update(psnr.item(), source_img.size(0))

    if wandb_run is not None:
        wandb.log({
            "val_psnr": psnr.item()
        }, step=step)

    return PSNR.avg


def get_wandb_run(args, settings, ckpt):
    assert args.enable_wandb and args.wandb_project is not None and args.wandb_team is not None, \
        "get_wandb_run was called, but not all required arguments were provided."

    if args.wandb_run_name is None:
         wandb_run_name = "_".join([
            "sweep" if args.wandb_sweep_config or args.wandb_sweep_id else "",
            args.model,
            "ckpt-" + (ckpt or "scratch").split('/')[-1].split('.')[0],
            f"batch-{settings['batch_size']}",
            f"lr-{settings['lr']}-{settings['optimizer']}",
            f"edge-{settings['edge_decay']}",
            f"patch-{settings['patch_size']}",
            f"batch-{settings['batch_size']}",
            f"epochs-{settings['epochs']}",
         ])
    else:
         wandb_run_name = args.wandb_run_name

    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_team,
        name=wandb_run_name,
        config=dict(**(vars(args) | settings)),
    )
    return run


if __name__ == '__main__':
    args = get_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    if args.enable_wandb:
        WANDB_TOKEN = os.getenv("WANDB_TOKEN")
        assert WANDB_TOKEN, "WANDB_TOKEN environment variable not set. Please set it to your Weights & Biases API key."
        wandb.login(key=WANDB_TOKEN, verify=True)

    if args.wandb_sweep_config is not None:
        assert args.enable_wandb, "You must enable Weights & Biases to use sweeps."
        assert args.wandb_project is not None, "You must specify a wandb project name to use sweeps."
        assert args.wandb_team is not None, "You must specify a wandb team name to use sweeps."
        with open(args.wandb_sweep_config, "r") as f:
            sweep_config = yaml.safe_load(f)
        sweep_id = wandb.sweep(sweep_config, project=args.wandb_project, entity=args.wandb_team)
        print(f"Created sweep with id: {sweep_id}")
    elif args.wandb_sweep_id is not None:
        assert args.enable_wandb, "You must enable Weights & Biases to use sweeps."
        assert args.wandb_project is not None, "You must specify a wandb project name to use sweeps."
        assert args.wandb_team is not None, "You must specify a wandb team name to use sweeps."
        sweep_id = args.wandb_sweep_id
    else:
        sweep_id = None

    def main():
        if not os.path.exists(args.config):
            raise FileNotFoundError(f"Config file not found: {args.config}")

        with open(args.config, 'r') as f:
            setting = json.load(f)

        if args.n_epochs is not None:
            setting['epochs'] = args.n_epochs

        if args.batch_size is not None:
            setting['batch_size'] = args.batch_size

        print(f"Training with settings: {setting}")

        if args.enable_wandb:
            wandb_run = get_wandb_run(args, setting, args.ckpt or (args.wandb_restore_ckpt.split('/')[-1] if args.wandb_restore_ckpt else None))
        else:
            wandb_run = None

        if args.wandb_sweep_config or args.wandb_sweep_id:
            setting["lr"] = wandb.config.get('lr')
            setting["loss_type"] = wandb.config.get('loss_type')
        else:
            setting['loss_type'] = "l1"

        if args.enable_wandb:
            wandb.config.update(setting, allow_val_change=True)

        # Extract exp name from config path for logging and saving
        exp_name = os.path.basename(os.path.dirname(args.config))

        network = eval(args.model.replace('-', '_'))()
        network = nn.DataParallel(network).cuda()

        if setting['loss_type'] == 'l1':
            criterion = nn.L1Loss()
        elif setting['loss_type'] == 'l2':
            criterion = nn.MSELoss()
        else:
            raise Exception(f"ERROR: unsupported loss type {setting['loss_type']}")

        if setting['optimizer'] == 'adam':
            optimizer = torch.optim.Adam(network.parameters(), lr=setting['lr'])
        elif setting['optimizer'] == 'adamw':
            optimizer = torch.optim.AdamW(network.parameters(), lr=setting['lr'])
        else:
            raise Exception(f"ERROR: unsupported optimizer {setting['optimizer']}") 

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=setting['epochs'], eta_min=setting['lr'] * 1e-2)
        scaler = GradScaler()

        if args.dataset == 'cityscapes_foggy':
            train_dataset = CityScapesPairLoader(
                data_dir=args.data_dir,
                mode='train',
                size=setting['patch_size'],
                edge_decay=setting['edge_decay'],
                only_h_flip=setting['only_h_flip']
            )
            val_dataset = CityScapesPairLoader(
                data_dir=args.data_dir, 
                mode=setting['valid_mode'], 
                size=setting['patch_size']
            )
        else:
            dataset_dir = os.path.join(args.data_dir, args.dataset)
            train_dataset = PairLoader(
                data_dir=dataset_dir,
                sub_dir='train',
                mode='train',
                size=setting['patch_size'],
                edge_decay=setting['edge_decay'],
                only_h_flip=setting['only_h_flip']
            )
            val_dataset = PairLoader(
                data_dir=dataset_dir,
                sub_dir='test',
                mode=setting['valid_mode'],
                size=setting['patch_size']
            )

        train_loader = DataLoader(train_dataset,
                                batch_size=setting['batch_size'],
                                shuffle=True,
                                num_workers=args.num_workers,
                                persistent_workers=True,
                                pin_memory=True,
                                drop_last=True)
        val_loader = DataLoader(val_dataset,
                                batch_size=setting['batch_size'],
                                num_workers=args.num_workers,
                                persistent_workers=True,
                                pin_memory=True)

        save_dir = os.path.join(args.save_dir, exp_name)
        os.makedirs(save_dir, exist_ok=True)

        assert not ((args.ckpt and (args.wandb_restore_ckpt or args.wandb_restore_run_path))), "Cannot restore from both checkpoint file and wandb"
        assert bool(args.wandb_restore_ckpt) == bool(args.wandb_restore_run_path), "Must provide either wandb_restore_ckpt and wandb_restore_run_path or neither of them"

        if args.ckpt:
            assert os.path.isfile(args.ckpt), "--ckpt %s does not exist" % args.ckpt
            print(f"==> Resuming training from local checkpoint: {args.ckpt}")
            ckpt = args.ckpt
        elif args.wandb_restore_ckpt is not None:
            print(f"==> Resuming training from wandb checkpoint: {args.wandb_restore_run_path}:{args.wandb_restore_ckpt}")
            wandb_restored = wandb.restore(
                name=args.wandb_restore_ckpt,
                run_path=args.wandb_restore_run_path,
                replace=True,
                root=Path("checkpoints") / args.wandb_restore_run_path,
            )
            ckpt = wandb_restored.name
            wandb_restored.close()
        else:
            ckpt = None
            print('==> Start training, current model name: ' + args.model)

        if ckpt is not None:
            checkpoint = torch.load(ckpt)

            network.load_state_dict(checkpoint['state_dict'])

            if 'optimizer' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer'])
            if 'scheduler' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler'])
            if 'scaler' in checkpoint:
                scaler.load_state_dict(checkpoint['scaler'])
            start_epoch = checkpoint.get('epoch', 0)

            if args.ignore_previous_best_score:
                best_psnr = 0
            else:
                best_psnr = checkpoint.get('best_psnr', 0)
        else:
            best_psnr = 0
            start_epoch = 0

        writer = SummaryWriter(log_dir=os.path.join(args.log_dir, exp_name, args.model))

        print(list(range(start_epoch, setting['epochs'])))
        for epoch in tqdm(range(start_epoch, setting['epochs']), desc=f"Training for {setting['epochs']} epochs"):
            loss = train(train_loader, network, criterion, optimizer, scaler, epoch=epoch, wandb_run=wandb_run)

            writer.add_scalar('train_loss', loss, epoch)

            scheduler.step()

            if epoch % setting['eval_freq'] == 0:
                avg_psnr = valid(val_loader, network, step=(epoch + 1) * len(train_loader) - 1, wandb_run=wandb_run)

                writer.add_scalar('valid_psnr', avg_psnr, epoch)

                if avg_psnr > best_psnr:
                    best_psnr = avg_psnr
                    ckpt_path = os.path.join(save_dir, "best_" + args.model + '.pth')
                    torch.save({
                        'state_dict': network.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'scaler': scaler.state_dict(),
                        'epoch': epoch,
                        'best_psnr': best_psnr
                    }, ckpt_path)
                    wandb.save(ckpt_path, policy="live")
                    tqdm.write("Model saved as %s" % ckpt_path)

                writer.add_scalar('best_psnr', best_psnr, epoch)

            ckpt_path = os.path.join(save_dir, "last_" + args.model + '.pth')
            torch.save({
                'state_dict': network.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'scaler': scaler.state_dict(),
                'epoch': epoch,
                'best_psnr': best_psnr
            }, ckpt_path)
            wandb.save(ckpt_path, policy="live")
            tqdm.write("Model saved as %s" % ckpt_path)

        if wandb_run:
            wandb_run.finish()

    if sweep_id is not None:
        wandb.agent(sweep_id, function=main, project=args.wandb_project, entity=args.wandb_team)
    else:
        main()
