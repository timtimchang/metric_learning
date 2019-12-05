#!/usr/bin/env bash
#SBATCH --job-name deep-metric-learning
#SBATCH --cpus-per-task 4
#SBATCH --mem 16G
#SBATCH --qos masterlow
#SBATCH --partition mhigh,mlow
#SBATCH --gres gpu:1
#SBATCH --chdir /home/grupo06/deep-metric-learning
#SBATCH --output ../logs/%x_%j.out

"""
	parser = argparse.ArgumentParser()
	parser.add_argument('dataset_dir', type=str)
	parser.add_argument('--min-images', type=int, default=10)
	parser.add_argument('--epochs', type=int, default=20)
	parser.add_argument('--input-size', type=int, default=224)
	parser.add_argument('--batch-size', type=int, default=128)
	parser.add_argument('--dims', type=int, default=32)
"""

#python src/main.py datasets/lfwcrop_color_by_dirs --min-images 20 
python src/main_cifar.py datasets/cifar-10-batches-py --epochs 5 | tee result/cifar_e5.txt 

