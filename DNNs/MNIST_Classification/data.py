from torch.utils.data import TensorDataset
import matplotlib.pyplot as plt
import torchvision, torch
from itertools import islice
from torchvision import transforms

def load_raw_MNIST():
    image_path = './DNNs/MNIST_Classification/'
    mnist_train_dataset = torchvision.datasets.MNIST(image_path, train=True,  download=True)
    mnist_test_dataset = torchvision.datasets.MNIST(image_path, train=False,  download=True)
    return mnist_train_dataset, mnist_test_dataset
    
def load_MNIST_dataset():
    image_path = './DNNs/MNIST_Classification/'
    transform = transforms.Compose([transforms.ToTensor()])
    mnist_train_dataset = torchvision.datasets.MNIST(image_path, train=True, transform=transform, download=True)
    mnist_test_dataset = torchvision.datasets.MNIST(image_path, train=False, transform=transform, download=True)
    assert isinstance(mnist_train_dataset, torch.utils.data.Dataset)
    assert isinstance(mnist_train_dataset, torch.utils.data.Dataset)
    return mnist_train_dataset, mnist_test_dataset
    
def plot_MNIST_sample(n, mnist_dataset):
    fig = plt.figure(figsize=(16,8))
    if(n > 14):
        n = 14
    for i, (img, label) in islice(enumerate(mnist_dataset), n):
        ax = fig.add_subplot(2, n//2, i+1)
        ax.imshow(img)
        ax.set_title(label)
    plt.show()

#mnist_train, _, mnist_test = load_MNIST_dataset()
#plot_MNIST_sample(20, load_raw_MNIST()[0])
