import numpy as np

def check( m ):
    val = True
    for i in range( 4 ):
        for j in range( 1, 4 ):
            for k in range( 4 ):
                if k < j:
                    if m[ k ][ i ] == m[ j ][ i ]:
                        val = False
    return val

def main():
    res1=np.array( [[1,2,3,4],[1,2,3,4],[1,2,3,4],[1,2,3,4]] )
    print( check(res1) )
    l=[1,2,3,4]
    import itertools
    ret = []
    for v in itertools.permutations(l, 4):
        ret.append(v)
    count = 0
    ans=[]
    for i in ret:
        for j in ret:
            for k in ret:
                for l in ret:

                    if check(np.array( [i, j, k, l] )):

                        ans.append( np.array( [i, j, k, l] ) )
                        count += 1
    print(count)
    for x in ans:
        print( x )


    # print(res1[0][0])
    # res1[ 0 ]=np.roll( res1[ 0 ], 1 )
    # print( res1 )
    # pass
def sub():
    # Configure data loader
    import os
    import torch
    from torchvision import datasets
    import torchvision.transforms as transforms
    import numpy

    os.makedirs("data/mnist", exist_ok=True)
    dataloader = torch.utils.data.DataLoader(
        datasets.MNIST(
            "data/mnist",
            train=True,
            download=True,
            transform=transforms.Compose(
                [transforms.Resize( 28 ), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
            ),
        ),
        batch_size=64,
        shuffle=True,
    )
    Tensor = torch.cuda.FloatTensor#  if cuda else torch.FloatTensor

    from torch.autograd import Variable
    for i, ( imgs, some ) in enumerate( dataloader ):
        # valid = Variable(Tensor(imgs.size(0), 1).fill_(1.0), requires_grad=False)
        # print( valid )
        # print( imgs.shape )
        # print(imgs)
        print( imgs.type(Tensor) )
        exit( 0 )
        print( imgs.size(), some[0], int( some[ 0 ] ) )


import os
import torch
from torch.autograd import Variable
from torchvision import datasets
import torchvision.transforms as transforms
import torch.nn as nn

class Generator( nn.Module ):
    def __init__( self, z_dim = 100, channel = 1, w = 28, h = 28 ):
        super().__init__()
        self.latent_dim = z_dim
        self.img_channels = channel
        self.img_width = w
        self.img_height = h
        self.img_shape = ( self.img_channels, self.img_width, self.img_height )

        def _block( in_feat, out_feat, normalize=True ):
            layers = [nn.Linear(in_feat, out_feat) ]
            if normalize:
                layers.append( nn.BatchNorm1d( out_feat ) )
            layers.append( nn.LeakyReLU( 0.2 ) )
            return layers

        self.model = nn.Sequential(
            *_block( self.latent_dim, 128, normalize=False ),
            *_block( 128, 256 ),
            *_block( 256, 512 ),
            *_block( 512, 1024 ),
            nn.Linear( 1024, int( np.prod( self.img_shape ) ) ),
            nn.Tanh()
        )
    def forward( self, z ):
        img = self.model( z )
        img = img.view( img.size( 0 ), self.img_channels, self.img_width, self.img_height )
        return img
 
class Discriminator( nn.Module ):
    def __init__(self, channel = 1, w = 28, h = 28):
        super().__init__()
        
        self.img_channels = channel
        self.img_width = w
        self.img_height = h

        self.img_shape = ( self.img_channels, self.img_width, self.img_height )        
        
        self.model = nn.Sequential(
            nn.Linear( int( np.prod( self.img_shape ) ), 512),
            nn.LeakyReLU( 0.2 ),
            nn.Linear( 512, 256 ),
            nn.LeakyReLU( 0.2 ),
            nn.Linear( 256, 1 ),
            nn.Sigmoid(),
        )
        
    def forward( self, img ):
        img_flat = img.view( img.size( 0 ), -1 )
        validity = self.model( img_flat )
        return validity

def get_dataloader():
    location = "data/mnist"
    os.makedirs(location, exist_ok=True)
    dataloader = torch.utils.data.DataLoader(
        datasets.MNIST(
            location,
            train=True,
            download=True,
            transform=transforms.Compose(
                [transforms.Resize( 28 ), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
            ),
        ),
        batch_size=64,
        shuffle=True,
    )
    return dataloader

from torchvision.utils import save_image

if __name__ == "__main__":
    batch_size = 64
    # 色々と初期化
    # Tensor = torch.FloatTensor ## CPU Version
    Tensor = torch.cuda.FloatTensor
    generator     = Generator().cuda()
    optimizer_G   = torch.optim.Adam( generator.parameters(), lr=0.0002, betas=( 0.5, 0.999 ) )
    discriminator = Discriminator().cuda()
    optimizer_D   = torch.optim.Adam( discriminator.parameters(), lr=0.0002, betas=( 0.5, 0.999 ) )
    # ロス関数の初期化
    adversarial_loss = torch.nn.BCELoss().cuda()
    
    epoch_size = 200 # 普通は100-200くらい。
    for epoch in range( epoch_size ):
        dataloader = get_dataloader()
        for i, ( real_images, some ) in enumerate( dataloader ):
            batch_size = real_images.size( 0 )
            # 正解と不正解のラベルを作る
            valid = torch.ones( (batch_size,1), requires_grad=False ).cuda()
            fake = torch.zeros( (batch_size,1), requires_grad=False ).cuda()
            # ---------------------
            #  Dの学習
            # ---------------------
            # DはGより２０回多く学習をさせる。( オリジナルの論文より）
            for j in range( 20 ):
                # まず初期化
                optimizer_D.zero_grad()
                # 偽画像の作成
                # ランダムな潜在変数を作成
                z = torch.empty( real_images.shape[0], 100,requires_grad=False ).normal_( mean = 0, std = 1 ).cuda()
                # fake imageを取得
                fake_images = generator( z )
                # ロスの計算. 
                real_loss = adversarial_loss( discriminator( real_images.type( Tensor ) ), valid )
                fake_loss = adversarial_loss( discriminator( fake_images.detach() ), fake )
                d_loss = (real_loss + fake_loss) / 2
                # 勾配を計算
                d_loss.backward()
                # 伝搬処理。Dにだけ誤差伝搬される
                optimizer_D.step()
            # ---------------------
            #  Gの学習
            # ---------------------            
            # まず初期化
            optimizer_G.zero_grad()
            # ランダムな潜在変数を作成
            z = torch.empty( real_images.shape[0], 100,requires_grad=False ).normal_( mean = 0, std = 1 ).cuda()
            # fake imageを取得
            fake_images = generator( z )
            # discriminatorを利用して結果を取得する
            g_loss = adversarial_loss(discriminator( fake_images ), valid )
            # 勾配を計算
            g_loss.backward()
            # 重みを更新する。Gのみにだけ勾配伝搬処理がされる
            optimizer_G.step()
            
            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, epoch_size, i, len(dataloader), d_loss.item(), g_loss.item())
                )

            batches_done = epoch * len(dataloader) + i
            if batches_done % 400 == 0:
                save_image(fake_images.data[:25], "images/%08d.png" % batches_done, nrow=5, normalize=True)
