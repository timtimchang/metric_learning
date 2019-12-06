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
		particles.append(particle)

	return np.array(particles)

def verifiability(data):
	data = [d[0] for d in data]
	dim = len(data[0])
	num = len(data)
	kde = gaussian_kde(data) 
	time = 1000
	num_particles = 10000	
	ver = 0


	for i in range(num):
		coeff = np.random.rand(dim)
		#print(data[0])
		p_data = [ np.dot(d, coeff) / np.dot(coeff, coeff) for d in data ] 

		particles = generate_particles(kde, num_particles, dim)
		p_particles = [ np.dot(particle, coeff) / np.dot(coeff, coeff) for particle in particles ] 
		
		max_inter, min_inter = max(p_data), min(p_data)
		for j in range(num_particles):
			if max_inter > p_particles[j] and p_particles[j] < min_inter:
				ver += 1
		
		"""
		coeff = np.random.rand(dim)
		p = np.dot(data[i], coeff) / np.dot(coeff, coeff)
		
		for j in range(0,num_particlse,2):
			# triple pick
			p1 = np.dot(particles[j], coeff) / np.dot(coeff, coeff)
			p2 = np.dot(particles[j+1], coeff) / np.dot(coeff, coeff)
			if p1 > p2:
				if p1 > p and p > p2:
					ver += 1
			else:
				if p1 < p and p < p2:
					ver += 1
		"""
			
			
			
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
	ver = verifiability(x)
	print(ver)
	ver = verifiability(x)
	print(ver)
	ver = verifiability(x)
	print(ver)
	ver = verifiability(x)
	print(ver)
			
	
		
		
	#plot_embeddings(test_loader.dataset, test_embeddings, test_targets, title='test_verify', num=2)

	


if __name__ == '__main__':
	main()
