import numpy as np
import cv2
import datetime as dt
import load_mnist

# only for testing use
sobel_kernels = np.array([[[1,1],[2,0],[1,-1]],[[0,2],[0,0],[0,-2]],[[-1,1],[-2,0],[-1,-1]]])

def load_image(img_path):
	img = cv2.imread(img_path, 0)
	if img.shape ==(92, 60):
		img = cv2.resize(img, 1/3.0)
	if img.shape == (28, 20):
		img_padded = np.zeros((32,32))
		img_padded[2:-2,6:-6] = img
		img = img_padded

	return (img-img.mean())/img.std()

def test_on_image(img_path):
	return np.argmax(forward_prop(load_image(img_path), 0)[1])

def test_on_image2(img_path):
	output = forward_prop(load_image(img_path), 0)[1]
	probabilities = [(i,p) for i,p in enumerate((output+1)/np.sum(output+1))]
	sp = sorted(probabilities, key = lambda x: x[1], reverse=True)
	for i in range(3):
		print('{}: {:.3g}%'.format(sp[i][0], 100*sp[i][1][0]))

# Cross-correlation on 2D in_img with 3D array representing multiple 2D kernels
def corr_multi_kernel(in_img, kernels):
	assert(in_img.ndim == 2)
	assert(kernels.ndim == 3)

	in_img = in_img.reshape(in_img.shape + (1,))

	output = np.zeros((in_img.shape[0] - kernels.shape[0] + 1, in_img.shape[1] - kernels.shape[1] + 1, kernels.shape[2]))

	# vectorization trick to compute convolutions with all the kernels at once, only looping over the kernel shape
	for i in range(kernels.shape[0]):
		for j in range(kernels.shape[1]):
			output += kernels[i,j,:].reshape((1,1,kernels.shape[2])) * in_img[i:, j:][:output.shape[0], :output.shape[1]]
	return output

# Cross-correlation when the kernel is an image a few pixels smaller than the image (inefficient otherwise)
def corr_giant_kernel(in_img, kernel_img):
	out_shape = tuple(np.array(in_img.shape)-np.array(kernel_img.shape)+1)
	output_mat = np.zeros(out_shape)
	for i in range(out_shape[0]):
		for j in range(out_shape[1]):
			output_mat[i,j] = np.sum(in_img[i:,j:][:kernel_img.shape[0],:kernel_img.shape[1]]*kernel_img)
	return output_mat

# Cross-correlation on 3D stack of 2D images with 4D array of 3D kernels
def corr_multi_image_multi_kernel(in_imgs, kernels):
	assert(in_imgs.ndim == 3)
	assert(kernels.ndim == 4)

	output = np.zeros((in_imgs.shape[0] - kernels.shape[0] + 1, in_imgs.shape[1] - kernels.shape[1] + 1, kernels.shape[3]))
	# for each kernel num
	for k in range(kernels.shape[3]):
		# compute 3D convolution where 3rd dimension matches, giving a 2D result
		for i in range(kernels.shape[0]):
			for j in range(kernels.shape[1]):
				output[:,:,k] += np.sum(kernels[i,j,:,k] * in_imgs[i:, j:, :][:output.shape[0], :output.shape[1], :], axis=2)
	return output

# n x n max pooling
def max_pool(in_imgs, n):
	assert(in_imgs.shape[0]%n == 0 and in_imgs.shape[1]%n == 0)
	max_vals = np.full((in_imgs.shape[0]/n, in_imgs.shape[1]/n, in_imgs.shape[2]), -np.inf)
	slices = [in_imgs[i::n,j::n,:] for i in range(n) for j in range(n)]
	for s in slices:
		max_vals = np.maximum(max_vals, s)
	return max_vals

# ReLU clipping
def relu(in_data):
	out_data = in_data.copy()
	out_data[out_data < 0] = 0
	return out_data

# Fully connected layer - matrix multiplication
def fully_connected_layer(in_vec, weight_mat, biases):
	return weight_mat.dot(in_vec) + biases

def pool_input_grad(input_mat, output_mat, output_grad, n):
	ismax = input_mat == np.repeat(np.repeat(output_mat, n, axis=0), n, axis=1)
	return ismax*np.repeat(np.repeat(output_grad, n, axis=0), n, axis=1)

def conv_grads(input_mat, output_grad, kernels):
	khalf = np.array([kernels.shape[i]-1 for i in range(2)])

	output_grad_temp = np.zeros((output_grad.shape[0]+2*khalf[0],output_grad.shape[1]+2*khalf[1],kernels.shape[3]))
	output_grad_temp[khalf[0]:-khalf[0],khalf[1]:-khalf[1],:] = output_grad

	input_grad = np.zeros_like(input_mat)
	kernels_grad = np.zeros_like(kernels)
	bias_grad = np.zeros((kernels.shape[3],))

	for i in range(kernels.shape[3]):
		input_grad += corr_multi_kernel(output_grad_temp[:,:,i], kernels[::-1,::-1,:,i])
		bias_grad[i] = np.sum(output_grad[:,:,i])
		for j in range(kernels.shape[2]):
			kernels_grad[:,:,j,i] = corr_giant_kernel(input_mat[:,:,j], output_grad[:,:,i])

	return input_grad, kernels_grad, bias_grad

# Architecture based on LeNet architecture from http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf
c1_k = np.random.normal(0,1/5,(5,5,6))
c1_b = np.random.normal(0,1,(1,1,6))
c2_k = np.random.normal(0,1/np.sqrt(5*5*6),(5,5,6,16))
c2_b = np.random.normal(0,1,(1,1,16))

fc1_w = np.random.normal(0,1/np.sqrt(400),(120,400)) # 5*5*16 = 400
fc1_b = np.random.normal(0,1/np.sqrt(400),(120,1))
fc2_w = np.random.normal(0,1/np.sqrt(120),(84,120))
fc2_b = np.random.normal(0,1/np.sqrt(120),(84,1))
fc3_w = np.random.normal(0,1/np.sqrt(84),(10,84))


def forward_prop(image, label):
	expected = np.zeros((10,1))-1
	expected[label] = 1

	conv1_result = corr_multi_kernel(image, c1_k) + c1_b # 28 x 28 x 6
	pool1_result = max_pool(conv1_result, 2) # 14 x 14 x 6
	conv2_result = corr_multi_image_multi_kernel(pool1_result, c2_k) + c2_b # 10 x 10 x 16
	pool2_result = max_pool(conv2_result, 2) # 5 x 5 x 16
	fc1_result = fully_connected_layer(pool2_result.reshape((-1,1)), fc1_w, fc1_b) # 120 x 1
	relu1_result = relu(fc1_result) # 120 x 1
	fc2_result = fully_connected_layer(relu1_result, fc2_w, fc2_b) # 84 x 1
	tanh1_result = np.tanh(fc2_result) #84 x 1
	fc3_result = fully_connected_layer(tanh1_result, fc3_w, 0) # 10 x 1
	tanh2_result = np.tanh(fc3_result) # 10 x 1
	E = np.sum((tanh2_result-expected)**2)
	return E, tanh2_result

def get_gradients(image, label):
	expected = np.zeros((10,1))-1
	expected[label] = 1

	conv1_result = corr_multi_kernel(image, c1_k) + c1_b # 28 x 28 x 6
	pool1_result = max_pool(conv1_result, 2) # 14 x 14 x 6
	conv2_result = corr_multi_image_multi_kernel(pool1_result, c2_k) + c2_b # 10 x 10 x 16
	pool2_result = max_pool(conv2_result, 2) # 5 x 5 x 16
	fc1_result = fully_connected_layer(pool2_result.reshape((-1,1)), fc1_w, fc1_b) # 120 x 1
	relu1_result = relu(fc1_result) # 120 x 1
	fc2_result = fully_connected_layer(relu1_result, fc2_w, fc2_b) # 84 x 1
	tanh1_result = np.tanh(fc2_result) #84 x 1
	fc3_result = fully_connected_layer(tanh1_result, fc3_w, 0) # 10 x 1
	tanh2_result = np.tanh(fc3_result) # 10 x 1
	E = np.sum((tanh2_result-expected)**2)

	tanh2_g = 2*(tanh2_result-expected) # 10 x 1
	fc3_g = tanh2_g*(1-tanh2_result**2) # 10 x 1
	fc3_w_g = np.outer(fc3_g, tanh1_result) # 10 x 84
	tanh1_g = (fc3_g.T.dot(fc3_w)).T # 84 x 1
	fc2_g = tanh1_g*(1-tanh1_result**2) # 84 x 1
	fc2_w_g = np.outer(fc2_g, relu1_result) # 84 x 120
	fc2_b_g = fc2_g # 84 x 1
	relu1_g = (fc2_g.T.dot(fc2_w)).T # 120 x 1
	fc1_g = relu1_g*(fc1_result > 0) # 120 x 1
	fc1_w_g = np.outer(fc1_g, pool2_result) # 120 x 400
	fc1_b_g = fc1_g # 120 x 1
	pool2_g = (fc1_g.T.dot(fc1_w)).T # 400 x 1
	conv2_g = pool_input_grad(conv2_result, pool2_result, pool2_g.reshape(5,5,16), 2) # 10 x 10 x 16
	pool1_g, c2_k_g, c2_b_g = conv_grads(pool1_result, conv2_g, c2_k) # 14 x 14 x 6 and 5 x 5 x 6 x 16 and 16
	conv1_g = pool_input_grad(conv1_result, pool1_result, pool1_g, 2) # 28 x 28 x 6
	input_g, c1_k_g, c1_b_g = conv_grads(image.reshape((32,32,1)), conv1_g, c1_k.reshape(5,5,1,6)) # 32 x 32 x 1 and 5 x 5 x 6 and 6

	return c1_k_g, c1_b_g, c2_k_g, c2_b_g, fc1_w_g, fc1_b_g, fc2_w_g, fc2_b_g, fc3_w_g

images,labels = load_mnist.load_mnist_data()

mats = [c1_k, c1_b, c2_k, c2_b, fc1_w, fc1_b, fc2_w, fc2_b, fc3_w]

good_networks = []
Es = []
epoch_Es = []
Esum = np.Inf
nu = 1e-4 # learning rate
mu = 0.75 # momentum term
batch_size = 50
last_grads = [0 for i in range(9)]
for epoch in range(20):
	order = np.random.choice(50000,(50000,),replace=False)
	last_mats = [mat.copy() for mat in mats]
	for j in range(1000):
		grads = [0 for i in range(9)]
		for i in range(batch_size):
			idx = order[j*batch_size+i]
			ind_grads = get_gradients(images[idx], labels[idx])
			for k in range(len(grads)):
				grads[k] += ind_grads[k]
		for k in range(len(grads)):
			grads[k] += mu*last_grads[k]

		Esum = 0
		for i in range(10):
			Esum += forward_prop(images[-i],labels[-i])[0]

		print(Esum/10)
		Es.append(Esum/10)

		for mat,grad in zip(mats, grads):
			mat -= grad.reshape(mat.shape)*nu

		#pvec = np.hstack([g.flatten() for g in grads])

		last_grads = [g.copy() for g in grads]
	Esum = 0
	for i in range(10000):
		Esum += forward_prop(images[-i],labels[-i])[0]
	epoch_Es.append(Esum/10000)
	print('Validation mean error after epoch {}: {}'.format(epoch, Esum/10000))
	if epoch >= 2:
		if epoch_Es[epoch-2] > epoch_Es[epoch-1] < epoch_Es[epoch]:
			good_networks.append(last_mats)


# nu = 1e-3
# for i in range(10):
# 	E1,output1 = forward_prop(images[i], 0)
# 	grads = get_gradients(images[i], labels[i])
# 	for mat,grad in zip([c1_k, c1_b, c2_k, c2_b, fc1_w, fc1_b, fc2_w, fc2_b, fc3_w], grads):
# 		grads[j] += grad.reshape(*mat.shape)*nu
#
# 	E2, output = forward_prop(images[i], 0)
# 	print(output)