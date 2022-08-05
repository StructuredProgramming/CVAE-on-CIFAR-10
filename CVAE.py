import torch 
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn. functional as F
import torch.optim as optim
import os
import sys


batch_size = 100
learning_rate = 0.001
max_epoch = 10000
device = torch.device("cpu")
num_workers = 5
load_epoch = -1
generate = True

class CVAE(nn.Module):
    def __init__(self,latent_size=32,num_classes=10):
        super(CVAE,self).__init__()
        self.latent_size = latent_size
        self.num_classes = num_classes

        
        self.conv1 = nn.Conv2d(3+1, 16, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.linear1 = nn.Linear(5*5*32,300)
        self.mu = nn.Linear(300, self.latent_size)
        self.logvar = nn.Linear(300, self.latent_size)

        
        self.linear2 = nn.Linear(self.latent_size + self.num_classes, 300)
        self.linear3 = nn.Linear(300,4*4*32)
        self.conv3 = nn.ConvTranspose2d(32, 16, kernel_size=5,stride=2)
        self.conv4 = nn.ConvTranspose2d(16, 1, kernel_size=5, stride=2)
        self.conv5 = nn.ConvTranspose2d(1, 3, kernel_size=8)

    def encoder(self,x,y):
        y = torch.argmax(y, dim=1).reshape((y.shape[0],1,1,1))
        y = y.expand(-1, -1, x.size(2), x.size(3))
        t = torch.cat((x,y),dim=1)
        
        t = F.relu(self.conv1(t))
        t = F.relu(self.conv2(t))
        t = t.reshape((x.shape[0], -1))

        t = F.relu(self.linear1(t))
        mu = self.mu(t)
        logvar = self.logvar(t)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        epsilon = torch.randn_like(std).to(device)
        return epsilon*std + mu
    
    def unFlatten(self, x):
        return x.reshape((x.shape[0], 32, 4, 4))

    def decoder(self, z):
        decoderoutput = F.relu(self.linear2(z))
        decoderoutput = F.relu(self.linear3(decoderoutput))
        decoderoutput = self.unFlatten(decoderoutput)
        decoderoutput = F.relu(self.conv3(decoderoutput))
        decoderoutput = F.relu(self.conv4(decoderoutput))
        decoderoutput = F.relu(self.conv5(decoderoutput))
        return decoderoutput


    def forward(self, x, y):
        mu, logvar = self.encoder(x,y)
        z = self.reparameterize(mu,logvar)

        z = torch.cat((z, y.float()), dim=1)
        pred = self.decoder(z)
        return pred, mu, logvar


def plot(epoch, pred, y,name='test_'):
    if not os.path.isdir('./images'):
        os.mkdir('./images')
    fig = plt.figure(figsize=(16,16))
    for i in range(6):
        ax = fig.add_subplot(3,2,i+1)
        ax.imshow(pred[i,0])
        ax.axis('off')
        ax.title.set_text(str(y[i]))
    plt.show()
    plt.savefig("./images/{}epoch_{}.jpg".format(name, epoch))
    plt.close()


def loss_function(x, pred, mu, logvar):
    reconstructionloss = F.mse_loss(pred, x, reduction='sum')
    kldivergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return reconstructionloss, kldivergence


def train(epoch, model, train_loader, optim):
    reconstruction_loss = 0
    kldivergence_loss = 0
    total_loss = 0
    for i,(x,y) in enumerate(train_loader):
        try:
            label = np.zeros((x.shape[0], 10))
            label[np.arange(x.shape[0]), y] = 1
            label = torch.tensor(label)

            optim.zero_grad()   
            pred, mu, logvar = model(x.to(device),label.to(device))
            
            reconstructionloss, kldivergence = loss_function(x.to(device),pred, mu, logvar)
            loss = reconstructionloss + kldivergence
            loss.backward()
            optim.step()

            total_loss += loss.cpu().data.numpy()*x.shape[0]
            reconstruction_loss += reconstructionloss.cpu().data.numpy()*x.shape[0]
            kldivergence_loss += kldivergence.cpu().data.numpy()*x.shape[0]
            if i == 0:
                
                for name,param in model.named_parameters():
                    if "bias" in name:
                        var=0
                    else:
                        var=1
                    
        except Exception as e:
            traceback.print_exe()
            torch.cuda.empty_cache()
            continue
    
    reconstruction_loss /= len(train_loader.dataset)
    kldivergence_loss /= len(train_loader.dataset)
    total_loss /= len(train_loader.dataset)
    return total_loss, kldivergence_loss,reconstruction_loss

def test(epoch, model, test_loader):
    reconstruction_loss = 0
    kldivergence_loss = 0
    total_loss = 0
    with torch.no_grad():
        for i,(x,y) in enumerate(test_loader):
            try:
                label = np.zeros((x.shape[0], 10))
                label[np.arange(x.shape[0]), y] = 1
                label = torch.tensor(label)

                pred, mu, logvar = model(x.to(device),label.to(device))
                reconstructionloss, kldivergence = loss_function(x.to(device),pred, mu, logvar)
                loss = reconstructionloss + kldivergence

                total_loss += loss.cpu().data.numpy()*x.shape[0]
                reconstruction_loss += reconstructionloss.cpu().data.numpy()*x.shape[0]
                kldivergence_loss += kldivergence.cpu().data.numpy()*x.shape[0]
                if i == 0:
                    # print("gr:", x[0,0,:5,:5])
                    # print("pred:", pred[0,0,:5,:5])
                    plot(epoch, pred.cpu().data.numpy(), y.cpu().data.numpy())
            except Exception as e:
                traceback.print_exe()
                torch.cuda.empty_cache()
                continue
    reconstruction_loss /= len(test_loader.dataset)
    kldivergence_loss /= len(test_loader.dataset)
    total_loss /= len(test_loader.dataset)
    return total_loss, kldivergence_loss,reconstruction_loss        



def generate_image(epoch,z, y, model):
    with torch.no_grad():
        label = np.zeros((y.shape[0], 10))
        label[np.arange(z.shape[0]), y] = 1
        label = torch.tensor(label)

        pred = model.decoder(torch.cat((z.to(device),label.float().to(device)), dim=1))
        plot(epoch, pred.cpu().data.numpy(), y.cpu().data.numpy(),name='Eval_')



def load_data():
    transform = torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor()])
    train_loader = torch.utils.data.DataLoader(torchvision.datasets.CIFAR10('./data/', train=True, download=True,
                             transform=transform),batch_size=batch_size, num_workers=num_workers, shuffle=True)
    test_loader = torch.utils.data.DataLoader(torchvision.datasets.CIFAR10('./data/', train=False, download=True,
                             transform=transform),batch_size=batch_size, num_workers=num_workers, shuffle=True)
    
    return train_loader, test_loader

def save_model(model, epoch):
    if not os.path.isdir("./checkpoints"):
        os.mkdir("./checkpoints")
    file_name = './checkpoints/model_{}.pt'.format(epoch)
    torch.save(model.state_dict(), file_name)



if __name__ == "__main__":
    train_loader, test_loader = load_data()
    model = CVAE().to(device)
    
    if load_epoch > 0:
        model.load_state_dict(torch.load('./checkpoints/model_{}.pt'.format(load_epoch), map_location=torch.device('cpu')))

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.001)


    train_loss_list = []
    test_loss_list = []
    for i in range(load_epoch+1, max_epoch):
        model.train()
        train_total, train_kldivergence, train_loss = train(i, model, train_loader, optimizer)
        with torch.no_grad():
            model.eval()
            test_total, test_kldivergence, test_loss = test(i, model, test_loader)
            if generate:
                z = torch.randn(6, 32).to(device)
                y = torch.tensor([1,2,3,4,5,6]) - 1
                generate_image(i,z, y, model)
            
        print("Epoch: {}/{} Train loss: {}, Train KLDivergence: {}, Train Reconstruction Loss:{}".format(i, max_epoch,train_total, train_kldivergence, train_loss))
        print("Epoch: {}/{} Test loss: {}, Test KLDivergence: {}, Test Reconstruction Loss:{}".format(i, max_epoch, test_loss, test_kldivergence, test_loss))

        save_model(model, i)
        train_loss_list.append([train_total, train_kldivergence, train_loss])
        test_loss_list.append([test_total, test_kldivergence, test_loss])
        np.save("train_loss", np.array(train_loss_list))
        np.save("test_loss", np.array(test_loss_list))
