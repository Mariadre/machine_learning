import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data


mnist = input_data.read_data_sets('/data', one_hot=True)

images, labels = mnist.train.next_batch(10)
print(images.shape)
print(labels[0])


fig = plt.figure(figsize=(8, 4))
for c, (image, label) in enumerate(zip(images, labels)):
    ax = fig.add_subplot(2, 5, c+1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('{}'.format(np.argmax(label)))
    ax.imshow(image.reshape(28, 28), vmin=0, vmax=1,
              cmap='gray_r', interpolation='nearest')

plt.show()
