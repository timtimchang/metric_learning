import os
import sys
import argparse
import io

import numpy as np
from PIL import Image
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn import metrics
from tqdm import tqdm

import torch
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import torchvision
import torchvision.transforms as transforms

from sklearn.manifold import TSNE
from matplotlib import pyplot as plt

from dataloader import Dataset, BalancedBatchSampler
from network import EmbeddingNet
from loss import OnlineTripletLoss


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('dataset_dir', type=str)
	parser.add_argument('--min-images', type=int, default=10)
	parser.add_argument('--epochs', type=int, default=20)
	parser.add_argument('--input-size', type=int, default=224)
	parser.add_argument('--batch-size', type=int, default=128)
	parser.add_argument('--dims', type=int, default=32)
	return parser.parse_args()


def fit(train_loader, test_loader, model, criterion, optimizer, scheduler, n_epochs, cuda):
	for epoch in range(1, n_epochs + 1):
		scheduler.step()

		train_loss = train_epoch(train_loader, model, criterion, optimizer, cuda)
		print('Epoch: {}/{}, Average train loss: {:.4f}'.format(epoch, n_epochs, train_loss))

		if test_loader is not None:
			accuracy = test_epoch(train_loader, test_loader, model, cuda)
			print('Epoch: {}/{}, Accuracy: {:.4f}'.format(epoch, n_epochs, accuracy))


def train_epoch(train_loader, model, criterion, optimizer, cuda):
	model.train()

	losses = []
	for batch_idx, data in tqdm(enumerate(train_loader), total=len(train_loader), desc='Training', file=sys.stdout):
		samples, targets = data
		if cuda:
			samples = samples.cuda()
			targets = targets.cuda()

		optimizer.zero_grad()
		outputs = model(samples)

		loss = criterion(outputs, targets)
		loss.backward()
		optimizer.step()

		losses.append(loss.item())

	return np.mean(losses)


def test_epoch(train_loader, test_loader, model, cuda):
	train_embeddings, train_targets = extract_embeddings(train_loader, model, cuda)
	test_embeddings, test_targets = extract_embeddings(test_loader, model, cuda)

	knn = KNeighborsClassifier(n_neighbors=5, n_jobs=4).fit(train_embeddings, train_targets)
	predicted = knn.predict(test_embeddings)
	accuracy = metrics.accuracy_score(test_targets, predicted)

	return accuracy


def extract_embeddings(loader, model, cuda):
	model.eval()

	embeddings = []
	targets = []
	with torch.no_grad():
		for sample, target in tqdm(loader, total=len(loader), desc='Testing', file=sys.stdout):
			if cuda:
				sample = sample.cuda()

			output = model.get_embedding(sample)

			embeddings.append(output.cpu().numpy())
			targets.append(target)
	embeddings = np.vstack(embeddings)
	targets = np.concatenate(targets)

	return embeddings, targets


def plot_embeddings(dataset, embeddings, targets, title=''):
	embeddings = TSNE(n_components=2).fit_transform(embeddings)
	plt.figure(figsize=(10, 10))
	for cls in np.random.choice(dataset.classes, 10):
		i = dataset.class_to_idx[cls]
		inds = np.where(targets == i)[0]
		plt.scatter(embeddings[inds, 0], embeddings[inds, 1], alpha=0.5)
	plt.legend(dataset.classes)
	plt.title(title)
	plt.savefig('result/{}_embeddings.png'.format(title))


def predict(train_loader, predict_loader, model, cuda):
	train_embeddings, train_targets = extract_embeddings(train_loader, model, cuda)
	predict_embeddings, predict_targets = extract_embeddings(predict_loader, model, cuda)

	nbrs = NearestNeighbors(n_neighbors=5, n_jobs=4).fit(train_embeddings, train_targets)
	predicted = nbrs.kneighbors(predict_embeddings)

	return predicted


def main():
	args = parse_args()
	print(vars(args))

	cuda = torch.cuda.is_available()
	if cuda:
		print('Device: {}'.format(torch.cuda.get_device_name(0)))

	# 超参数设置
	num_classes = 10
	batch_size = 32
	learning_rate = 0.001


	# cifar10 分类索引
	classes = ('plane', 'car', 'bird', 'cat',
			   'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

	# 数据增广方法
	transform = transforms.Compose([
		# +4填充至36x36
		transforms.Pad(4),
		# 随机水平翻转
		transforms.RandomHorizontalFlip(), 
		# 随机裁剪至32x32
		transforms.RandomCrop(32), 
		# 转换至Tensor
		transforms.ToTensor(),
		#  归一化
		#   transforms.Normalize(mean=(0.5, 0.5, 0.5),   # 3 for RGB channels
		#                        std=(0.5, 0.5, 0.5))
		])


	# cifar10路径
	cifar10Path = './datasets/cifar-10-batches-py/'

	#  训练数据集
	train_dataset = torchvision.datasets.CIFAR10(root=cifar10Path,
												 train=True, 
												 transform=transform,
												 download=True)

	# 测试数据集
	test_dataset = torchvision.datasets.CIFAR10(root=cifar10Path,
												train=False, 
												transform=transform)

	# 生成数据加载器
	train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
											   batch_size=batch_size, 
											   shuffle=True)
	# 测试数据加载器
	test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
											  batch_size=batch_size, 
											  shuffle=False)

	model = EmbeddingNet(args.dims)
	if cuda:
		model = model.cuda()
	print(model)

	criterion = OnlineTripletLoss(margin=1.0)
	optimizer = Adam(model.parameters(), lr=1e-4)
	scheduler = StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)

	fit(train_loader, test_loader, model, criterion, optimizer, scheduler, args.epochs, cuda)

	train_embeddings, train_targets = extract_embeddings(train_loader, model, cuda)
	plot_embeddings(train_loader.dataset, train_embeddings, train_targets, title='train')

	test_embeddings, test_targets = extract_embeddings(test_loader, model, cuda)
	plot_embeddings(test_loader.dataset, test_embeddings, test_targets, title='test')


if __name__ == '__main__':
	main()
