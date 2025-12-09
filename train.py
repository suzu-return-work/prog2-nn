import time

import matplotlib.pyplot as plt

import torch
from torchvision import datasets
import torchvision.transforms.v2 as transforms

import models

ds_transform=transforms.Compose([
    transforms.ToImage(),
    transforms.ToDtype(torch.float32,scale=True)
])

ds_train=datasets.FashionMNIST(
    root='data',
    train=True,
    download=True,
    transform=ds_transform
)

ds_test=datasets.FashionMNIST(
    root='data',
    train=False,
    download=True,
    transform=ds_transform
)

batch_size=64
dataloader_train=torch.utils.data.DataLoader(
    ds_train,
    batch_size=batch_size,
    shuffle=True
)
dataloader_test=torch.utils.data.DataLoader(
    ds_test,
    batch_size=batch_size
)

for image_batch,label_batch in dataloader_test:
    print(image_batch.shape)
    print(label_batch.shape)
    break

model=models.MyModel()


#acc_test=models.test_accuracy(model,dataloader_test)
#print(f'test accuracy: {acc_test*100:.3f}%')

loss_fn=torch.nn.CrossEntropyLoss()

learning_rate=1e-3
optimizer=torch.optim.SGD(model.parameters(),lr=learning_rate)

n_epochs=20

train_loss_log=[]
val_loss_log=[]
train_acc_log=[]
val_acc_log=[]

for epoch in range(n_epochs):
    print(f'epoch {epoch+1}/{n_epochs}')

    time_start=time.time()
    train_loss=models.train(model, dataloader_train,loss_fn,optimizer)
    time_end=time.time()
    print(f'training loss: {train_loss}({time_end-time_start}s)')
    #print(f'training loss:{train_loss}')
    train_loss_log.append(train_loss)

    val_loss = models.test(model,dataloader_test,loss_fn)
    print(f'validation loss: {val_loss}')
    val_loss_log.append(val_loss)

    train_acc=models.test_accuracy(model, dataloader_train)
    print(f'training accuracy: {train_acc*100:.3f}%')
    train_acc_log.append(train_acc)

    val_acc=models.test_accuracy(model,dataloader_test)
    print(f'validation accuracy: {val_acc*100:.3f}%')
    val_acc_log.append(val_acc)


plt.subplot(1,2,1)
plt.plot(range(1,n_epochs+1),train_loss_log,label='train')
plt.plot(range(1,n_epochs+1), val_loss_log, label='validation')
plt.xlabel('epochs')
plt.xticks(range(1,n_epochs+1))
plt.ylabel('loss')
plt.grid()
plt.legend()

plt.subplot(1,2,2)
plt.plot(range(1,n_epochs+1), train_acc_log, label='train')
plt.plot(range(1,n_epochs+1),val_acc_log,label='validation')
plt.xlabel('epochs')
plt.xticks(range(1,n_epochs+1))
plt.ylabel('accuracy')
plt.grid()
plt.legend()

plt.show()


#plt.plot(train_loss_log)
#plt.xlabel('epochs')
#plt.ylabel('loss')
#plt.grid()
#plt.show()
