import numpy as np

train_images_path = r"train-images.idx3-ubyte"
train_labels_path = r"train-labels.idx1-ubyte"

def load_mnist_data():
	images = []

	with open(train_images_path, 'rb') as train_images_fp:
		b = bytearray(train_images_fp.read(16))
		headerdata = np.ndarray((4,), dtype='>i4', buffer=b)
		assert(headerdata[0] == 2051)
		n_images = headerdata[1]
		image_size = tuple(headerdata[2:])
		for i in range(n_images):
			b = bytearray(train_images_fp.read(np.prod(image_size)))
			img = np.zeros((32,32), dtype=np.uint8)
			img[2:-2,2:-2] = np.ndarray(image_size, dtype=np.uint8, buffer=b)
			images.append((img-img.mean())/img.std())

	with open(train_labels_path, 'rb') as train_labels_fp:
		b = bytearray(train_labels_fp.read(8))
		headerdata = np.ndarray((2,), dtype='>i4', buffer=b)
		assert(headerdata[0] == 2049)
		n_labels = headerdata[1]
		b = bytearray(train_labels_fp.read(n_labels))
		labels = np.ndarray((n_labels,), dtype=np.uint8, buffer=b)

	return images, labels