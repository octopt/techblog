# -*- coding:utf-8 -*-
from torch.autograd import Variable
import os
import torch
import torch.utils.data
from torch import nn
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.nn import functional as F

class Encoder( nn.Module ):
    def __init__( self ):
        super().__init__()
        self.common = nn.Sequential(
            nn.Linear( 784, 400 ),
            nn.ReLU(),
            )
        self.model1 = nn.Sequential(
            self.common,
            nn.Linear( 400, 20 )
            )
        self.model2 = nn.Sequential(
            self.common,
            nn.Linear( 400, 20 )
            )
    def forward( self, img ):
        img_flat = img.view( img.size( 0 ), -1 )
        return self.model1( img_flat ), self.model2( img_flat )

class Decoder( nn.Module ):
    def __init__( self ):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear( 20, 400 ),
            nn.ReLU(),
            nn.Linear( 400, 784 ),
            nn.Sigmoid(),
            )
    def forward( self, z ):
        return self.model( z )
    
class VAE( nn.Module ):
    def __init__( self ):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        
    def _reparameterization_trick( self, mu, logvar ):
        std = torch.exp( 0.5 * logvar )
        eps = torch.randn_like( std )
        return mu + eps * std
    
    def forward( self, _input ):
        mu, sigma = self.encoder( _input )
        z         = self._reparameterization_trick( mu, sigma )
        return self.decoder( z ), mu, sigma

def get_dataloader():
    location = "data/mnist"
    os.makedirs(location, exist_ok=True)
    dataloader = torch.utils.data.DataLoader(
        datasets.MNIST(
            location,
            train=True,
            download=True,
            transform=transforms.Compose(
                [transforms.ToTensor()]
            ),
        ),
        batch_size=64,
        shuffle=True,
    )
    return dataloader

# Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
def VAE_LOSS( recon_x, x, mu, logvar ):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), size_average=False)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

def main():
    epoch_size = 50
    vae = VAE()
    Tensor = torch.FloatTensor
    
    USE_GPU=True
    if USE_GPU:
        vae.cuda()
        Tensor = torch.cuda.FloatTensor
    dataloader=get_dataloader()
    optimizer = torch.optim.Adam( vae.parameters(), lr=1e-3 )
    
    for epoch in range( epoch_size ):
        for i, ( imgs, _ ) in enumerate(dataloader):
            optimizer.zero_grad()
            real_images          = Variable( imgs.type( Tensor ) )
            gen_imgs, mu, logvar = vae( real_images )
            loss                 = VAE_LOSS( gen_imgs, real_images, mu, logvar )
            if USE_GPU:
                loss = loss.cuda()
            loss.backward()
            optimizer.step()
            print(
                "[Epoch %d/%d] [Batch %d/%d] [Loss: %f]"
                % (epoch, epoch_size, i, len(dataloader), loss.item())
                )
            if i % 50 == 0:
                save_image( gen_imgs.view(64, 1, 28, 28), "images/%d.png" % (i * epoch), normalize=True)

if __name__ == '__main__':
    main()
