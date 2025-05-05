import os
import argparse
import json
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

# Wandb options
parser.add_argument("--enable_wandb", action='store_true', default=False, help="Use Weights & Biases for logging")
parser.add_argument("--wandb_team", type=str, default=None, help="Weights & Biases team name")
parser.add_argument("--wandb_project", type=str, default=None, help="Weights & Biases project name")
parser.add_argument("--wandb_run_name", type=str, default=None, help="Weights & Biases current run name")

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu


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

		with torch.no_grad():							# torch.no_grad() may cause warning
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

	WANDB_TOKEN = os.getenv("WANDB_TOKEN")
	assert WANDB_TOKEN, "WANDB_TOKEN environment variable not set. Please set it to your Weights & Biases API key."
	wandb.login(key=WANDB_TOKEN, verify=True)

	if args.wandb_run_name is None:
		 wandb_run_name = "_".join([
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
		config=dict(**vars(args), **settings),
	)
	return run


if __name__ == '__main__':
	if not os.path.exists(args.config):
		raise FileNotFoundError(f"Config file not found: {args.config}")
	
	with open(args.config, 'r') as f:
		setting = json.load(f)

	# Extract exp name from config path for logging and saving
	exp_name = os.path.basename(os.path.dirname(args.config))

	network = eval(args.model.replace('-', '_'))()
	network = nn.DataParallel(network).cuda()

	criterion = nn.L1Loss()

	if setting['optimizer'] == 'adam':
		optimizer = torch.optim.Adam(network.parameters(), lr=setting['lr'])
	elif setting['optimizer'] == 'adamw':
		optimizer = torch.optim.AdamW(network.parameters(), lr=setting['lr'])
	else:
		raise Exception("ERROR: unsupported optimizer") 

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

	if args.enable_wandb:
		wandb_run = get_wandb_run(args, setting, args.ckpt)
	else:
		wandb_run = None

	if not args.ckpt:
		print('==> Start training, current model name: ' + args.model)
		best_psnr = 0
		start_epoch = 0
	else:
		print(f"==> Resuming training from checkpoint: {args.ckpt}")
		checkpoint = torch.load(args.ckpt)

		network.load_state_dict(checkpoint['state_dict'])

		if 'optimizer' in checkpoint:
			optimizer.load_state_dict(checkpoint['optimizer'])
		if 'scheduler' in checkpoint:
			scheduler.load_state_dict(checkpoint['scheduler'])
		if 'scaler' in checkpoint:
			scaler.load_state_dict(checkpoint['scaler'])
		start_epoch = checkpoint.get('epoch', 0)

		best_psnr = checkpoint.get('best_psnr', 0)

	writer = SummaryWriter(log_dir=os.path.join(args.log_dir, exp_name, args.model))

	for epoch in tqdm(range(start_epoch, setting['epochs'] + 1), desc=f"Training for {setting['epochs']} epochs"):
		loss = train(train_loader, network, criterion, optimizer, scaler, epoch=epoch, wandb_run=wandb_run)

		writer.add_scalar('train_loss', loss, epoch)

		scheduler.step()

		if epoch % setting['eval_freq'] == 0:
			avg_psnr = valid(val_loader, network, step=(epoch + 1) * len(train_loader) - 1, wandb_run=wandb_run)
		
			writer.add_scalar('valid_psnr', avg_psnr, epoch)

			if avg_psnr > best_psnr:
				best_psnr = avg_psnr
				torch.save({
					'state_dict': network.state_dict(),
					'optimizer': optimizer.state_dict(),
					'scheduler': scheduler.state_dict(),
					'scaler': scaler.state_dict(),
					'epoch': epoch,
					'best_psnr': best_psnr
				}, os.path.join(save_dir, args.model + '.pth'))
			
			writer.add_scalar('best_psnr', best_psnr, epoch)
