from data import load_MNIST_dataset
from model import MNISTClassifier
import torch.nn as nn
import torch
from torchvision import io, transforms
from torch.utils.data import DataLoader, random_split
import os
import matplotlib.pyplot as plt
import time

device = torch.device(
    'cuda' if torch.cuda.is_available()
    else 'mps' if torch.backends.mps.is_available()
    else 'cpu'
)

print(f'device: {device}')

def plot_accurecies(train_hist, eval_hist):
    plt.plot(range(0,len(train_hist)), train_hist, c='green', marker='*', label='train accurecy')
    plt.plot(range(0,len(eval_hist)), eval_hist, c='red', marker='o', label='eval accurecy')
    plt.legend()
    plt.show()

def eval(model, dl, msg, print_acc):
    acc_hist = []
    with torch.no_grad():
        accuracy = 0
        for x_batch, y_batch in dl:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            pred = model(x_batch)
            accuracy += (torch.argmax(pred, dim=1) == y_batch).float().sum()
        accuracy /= len(dl.dataset)
        accuracy = accuracy.item()
        acc_hist.append(accuracy)
        if(print_acc):
            print(f'{msg} evaluation accuracy: {accuracy}')
    return acc_hist
        
    

def train(epoches, model, optimizer, loss_fn, train_dl, eval_dl):
    acc_hist , eval_acc_hist= [], []
    for epoch in range(epoches):
        accuracy = 0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            accuracy += (torch.argmax(pred, dim=1) == y).float().sum()
        accuracy /= len(train_dl.dataset)
        accuracy = accuracy.item()
        acc_hist.append(accuracy)
        if(epoch%4 == 0):
            print(f'epoch {epoch} accuracy: {accuracy}')
        eval_acc_hist.append(eval(model, eval_dl, 'Training ', False))
    return acc_hist, eval_acc_hist

torch.manual_seed(111)
classifier = MNISTClassifier().to(device)
mnist_train_dataset, mnist_test_dataset = load_MNIST_dataset()
batch_size = 1024
train_len = int(len(mnist_train_dataset) *0.9)
val_len = int(len(mnist_train_dataset)- train_len)
train_dataset, val_dataset = random_split(mnist_train_dataset,[train_len, val_len], generator=torch.Generator().manual_seed(111))
train_dl = DataLoader(train_dataset, batch_size, shuffle=True)
eval_dl = DataLoader(val_dataset, batch_size, shuffle=True)
test_dl = DataLoader(mnist_test_dataset, batch_size, shuffle=True)
allow_loading = True
model_path = './DNNs/MNIST_Classification/mnist_classifier.pth'
if(os.path.exists(model_path)) and allow_loading:
    classifier.load_state_dict(torch.load(model_path))
    print('Loaded existing mnist classifier weights')
    eval(classifier, test_dl, 'Test', True)
else:
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=0.001, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    start_time = time.time()
    acc_hist, eval_acc_hist = train(20, classifier, optimizer, loss_fn, train_dl, eval_dl)
    training_time = time.time() - start_time
    print(f'training done in {training_time:.2f} seconds')
    test_acc_hist = eval(classifier, test_dl, 'Test', True)
    torch.save(classifier.state_dict(), model_path)
    plot_accurecies(acc_hist, eval_acc_hist)


