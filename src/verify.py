from main import * 

from scipy import stats
import matplotlib.pyplot as plt

def cdf(kde,x):
	return kde.integrate_box_1d(-np.inf, x)

def gaussian_kde(data):
	dim = len(data[0])
	
	kde_list = []
	for i in range(dim):
		var = [ x[i] for x in data]
		dist = stats.kde.gaussian_kde(var)
		kde_list.append(dist)
	
	return kde_list

def generate_particles(kde,num,dim):
	particles = []
			
	for i in range(num):
		particle = []
		for j in range(dim):
			var = np.random.rand(1)
			val = cdf(kde[j],var)
			particle.append(val)
		particles.append(aprticle)

	return np.array(particles)

def verifiability(data):
	dim = len(data[0])
	num = len(data)
	time = 1000
	num_particles = 1000	
	ver = 0

	kde = gaussian_kde(data) 
	particles = generate_particles(kde, num_particles, dim)

	for i in range(num):
		coeff = np.random.rand(dim)
		p = np.dot(data[i], coeff) / np.dot(coeff, coeff)
		
		for j in range(0,num_particlse,2):
			p1 = np.dot(particles[j], coeff) / np.dot(coeff, coeff)
			p2 = np.dot(particles[j+1], coeff) / np.dot(coeff, coeff)
			if p1 > p2:
				if p1 > p and p > p2:
					ver += 1
			else:
				if p1 < p and p < p2:
					ver += 1
	
	return ver / (times*num_particles)

def main():
	cuda = torch.cuda.is_available()
	if cuda:
		print('Device: {}'.format(torch.cuda.get_device_name(0)))

	# 超参数设置
	num_classes = 10
	batch_size = 32
	learning_rate = 1e-3
	dim = 32


	# cifar10 分类索引
	classes = ('plane', 'car', 'bird', 'cat',
			   'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

	# cifar10路径
	cifar10Path = './datasets/cifar-10-batches-py/'

	# 数据增广方法
	transform = transforms.Compose([
		transforms.ToTensor(),
		])

	#  训练数据集
	train_dataset = torchvision.datasets.CIFAR10(root=cifar10Path,
												 train=True, 
												 transform=transform,
												 download=True)

	# 测试数据集
	test_dataset = torchvision.datasets.CIFAR10(root=cifar10Path,
												transform=transform,
												train=False) 

	# 生成数据加载器
	train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
											   batch_size=batch_size, 
											   shuffle=True)
	# 测试数据加载器
	test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
											  batch_size=batch_size, 
											  shuffle=False)

	# model part
	model = EmbeddingNet(dim)
	model.load_state_dict(torch.load('model/params_300.pkl'))
	if cuda:
		model = model.cuda()
	print(model)

	# for plot train embedding
	#train_embeddings, train_targets = extract_embeddings(train_loader, model, cuda)
	#plot_embeddings(train_loader.dataset, train_embeddings, train_targets, title='train_verify', num=2)

	# for plot test emebedding
	test_embeddings, test_targets = extract_embeddings(test_loader, model, cuda)
	
	x = [ i for i in list(zip(test_embeddings, test_targets)) if i[1] == test_targets[0] ]
	ver = verifiability(x)
	print(ver)
			
	
		
		
	#plot_embeddings(test_loader.dataset, test_embeddings, test_targets, title='test_verify', num=2)

	


if __name__ == '__main__':
	main()
