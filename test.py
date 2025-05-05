import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import ssim
from torch.utils.data import DataLoader
from collections import OrderedDict
import json

import wandb
from tqdm import tqdm

from utils import AverageMeter, write_img, chw_to_hwc
from datasets.loader import PairLoader, CityScapesPairLoader
from models import *


parser = argparse.ArgumentParser()
parser.add_argument('--model', default='dehazeformer-s', type=str, help='model name')
parser.add_argument('--num_workers', default=16, type=int, help='number of workers')
parser.add_argument('--data_dir', default='./data/', type=str, help='path to dataset')
parser.add_argument('--ckpt', type=str, help='path to models checkpoint')
parser.add_argument('--result_dir', default='./results/', type=str, help='path to results saving')
parser.add_argument('--dataset', default='RESIDE-IN', type=str, help='dataset name')
parser.add_argument('--config', default='configs/indoor/default.json', type=str, help='path to training config file')

# Wandb options
parser.add_argument("--enable_wandb", action='store_true', default=False, help="Use Weights & Biases for logging")
parser.add_argument("--wandb_team", type=str, default=None, help="Weights & Biases team name")
parser.add_argument("--wandb_project", type=str, default=None, help="Weights & Biases project name")
parser.add_argument("--wandb_run_name", type=str, default=None, help="Weights & Biases current run name")

args = parser.parse_args()


def single(ckpt):
	state_dict = torch.load(ckpt, weights_only=False)['state_dict']
	new_state_dict = OrderedDict()

	for k, v in state_dict.items():
		name = k[7:]
		new_state_dict[name] = v

	print('==> Loading checkpoint %s' % ckpt)
	return new_state_dict


def test(test_loader, network, result_dir, wandb_run=None):
	PSNR = AverageMeter()
	SSIM = AverageMeter()

	torch.cuda.empty_cache()

	network.eval()

	os.makedirs(os.path.join(result_dir, 'imgs'), exist_ok=True)
	f_result = open(os.path.join(result_dir, 'results.csv'), 'w')

	for idx, batch in enumerate(tqdm(test_loader)):
		input = batch['source'].cuda()
		target = batch['target'].cuda()

		filename = batch['filename'][0]

		with torch.no_grad():
			output = network(input).clamp_(-1, 1)

			# [-1, 1] to [0, 1]
			output = output * 0.5 + 0.5
			target = target * 0.5 + 0.5

			psnr_val = 10 * torch.log10(1 / F.mse_loss(output, target)).item()

			_, _, H, W = output.size()
			down_ratio = max(1, round(min(H, W) / 256))		# Zhou Wang
			ssim_val = ssim(F.adaptive_avg_pool2d(output, (int(H / down_ratio), int(W / down_ratio))), 
							F.adaptive_avg_pool2d(target, (int(H / down_ratio), int(W / down_ratio))), 
							data_range=1, size_average=False).item()				

		PSNR.update(psnr_val)
		SSIM.update(ssim_val)

		tqdm.write('Test: [{0}]\t'
			  'PSNR: {psnr.val:.02f} ({psnr.avg:.02f})\t'
			  'SSIM: {ssim.val:.03f} ({ssim.avg:.03f})'
			  .format(idx, psnr=PSNR, ssim=SSIM))

		f_result.write('%s,%.02f,%.03f\n'%(filename, psnr_val, ssim_val))

		out_img = chw_to_hwc(output.detach().cpu().squeeze(0).numpy())
		write_img(os.path.join(result_dir, 'imgs', filename), out_img)

	f_result.close()

	os.rename(os.path.join(result_dir, 'results.csv'), 
			  os.path.join(result_dir, '%.02f | %.04f.csv'%(PSNR.avg, SSIM.avg)))

	if wandb_run is not None:
		wandb_run.log({
			"test_psnr": PSNR.avg,
			"test_ssim": SSIM.avg
		})

def get_wandb_run(args, settings, ckpt):
	assert args.enable_wandb and args.wandb_project is not None and args.wandb_team is not None, \
		"get_wandb_run was called, but not all required arguments were provided."

	WANDB_TOKEN = os.getenv("WANDB_TOKEN")
	assert WANDB_TOKEN, "WANDB_TOKEN environment variable not set. Please set it to your Weights & Biases API key."
	wandb.login(key=WANDB_TOKEN, verify=True)

	if args.wandb_run_name is None:
		 wandb_run_name = "_".join([
       		"test_",
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
	network = eval(args.model.replace('-', '_'))()
	network.cuda()
	exp_name = os.path.basename(os.path.dirname(args.config))
	
	if os.path.exists(args.ckpt):
		print('==> Start testing, current model name: ' + args.model)
		network.load_state_dict(single(args.ckpt))
	else:
		print('==> No existing trained model!')
		exit(0)

	
	# Select the appropriate loader based on dataset
	if args.dataset == 'cityscapes_foggy':
		test_dataset = CityScapesPairLoader(
			data_dir=args.data_dir,
   			mode='test'
		)
	else:
		dataset_dir = os.path.join(args.data_dir, args.dataset)
		test_dataset = PairLoader(
			data_dir=dataset_dir,
			sub_dir='test',
			mode='test'
		)
		
	test_loader = DataLoader(test_dataset,
							 batch_size=1,
							 num_workers=args.num_workers,
							 pin_memory=True)

	
	with open(args.config, 'r') as f:
		setting = json.load(f)
  
	if args.enable_wandb:
		wandb_run = get_wandb_run(args, setting, args.ckpt)
	else:
		wandb_run = None

	result_dir = os.path.join(args.result_dir, args.dataset, args.model)
	test(test_loader, network, result_dir, wandb_run)
