from typing import (Union,
                    Callable)

from typing import (Sequence,
                    Generator,
                    List)

import torch
from torch import nn

import torch
from torch import nn
from torch import distributions
from SegNet import SegNet
import unet_github


def ce_loss(labels,logits,n_classes,one_hot_labels ):
    batch_size = labels.shape[0]

    if not one_hot_labels: #REMOVE NOT!!!!!!!!!!!!!!!!!!!!!!!!!!! заглушка
        flat_labels = labels.view(-1, n_classes)
    else:
        flat_labels = labels.view(-1)
        flat_labels = tf.one_hot(indices=flat_labels, depth=n_classes, axis=-1) ## HOW TO CHANGE IT?

    flat_logits = logits.view(-1, n_classes)
    sm = nn.Softmax()
    bce = nn.BCELoss()
    ce_per_pixel = bce(sm(logits),labels)#tf.nn.softmax_cross_entropy_with_logits_v2(labels=flat_labels, logits=flat_logits)
    print(ce_per_pixel)

    ce_sum = torch.sum(ce_per_pixel) / batch_size
    ce_mean = torch.mean(ce_per_pixel)

    return {'sum': ce_sum, 'mean': ce_mean}


from typing import (Sequence,
                    Generator,
                    List)
def yield_pairs(in_sequence: Sequence,
                reverse: bool = False) -> Generator[Sequence, None, None]:

    """Yields pairs of elements from in_sequence"""

    if reverse:
        in_sequence = list(reversed(in_sequence))
    else:
        in_sequence = list(in_sequence)
    for i in range(len(in_sequence) - 1):
        yield in_sequence[i: i + 2]


class EncoderBlock(nn.Module):

    """Docstring"""

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 n_convs: int = 3,
                 activation: nn.Module = nn.ReLU()) -> None:
        super(EncoderBlock, self).__init__()

        self.pool = nn.MaxPool2d(kernel_size=2, 
                                 return_indices=True)

        channels = []  # type: List
        channels.append([in_channels, out_channels])
        for _ in range(n_convs - 1):
            channels.append([out_channels, out_channels])

        modules = []  # type: List
        for pair in channels:
            modules += [nn.Conv2d(*pair, kernel_size=3)]
            modules += [activation]

        self.sequence = nn.Sequential(*modules)

    def forward(self, in_batch: torch.Tensor) -> Sequence:
        print(in_batch.shape)
        in_batch, indices = self.pool(in_batch)
        return self.sequence(in_batch), indices
    

class DecoderBlock(nn.Module):

    """Docstring"""

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 n_convs: int = 3,
                 activation: nn.Module = nn.ReLU()) -> None:
        super(DecoderBlock, self).__init__()

        self.unpool = nn.MaxUnpool2d(kernel_size=2)
        channels = []  # type: List
        channels.append([in_channels, out_channels])
        for _ in range(n_convs - 1):
            channels.append([out_channels, out_channels])

        modules = []  # type: List
        for pair in channels:
            modules += [nn.Conv2d(*pair, kernel_size=3)]
            modules += [activation]

        self.sequence = nn.Sequential(*modules)

    def forward(self,
                up_features: torch.Tensor,
                cat_features: torch.Tensor,
                in_indices: torch.Tensor,
                in_shape: torch.Size) -> torch.Tensor:
        print(up_features.shape)
        up_features = self.unpool(up_features, in_indices, in_shape)
        return self.sequence(torch.cat([cat_features, up_features], dim=1))


class VGGEncoder(nn.Module):

    """Docstring"""

    def __init__(self,
                 in_channels: int,
                 start_out_channels: int = 64,
                 n_blocks: int = 5,
                 n_convs_per_layer: int = 3) -> None:
        super(VGGEncoder, self).__init__()

        self.n_blocks = n_blocks

        _sizes = [start_out_channels * 2 ** i for i in range(n_blocks)]
        self.pairs = list(yield_pairs([in_channels] + _sizes))
        modules = []  # type: List
        for i, pair in enumerate(self.pairs):
            if i == 0:
                # applying just convolution
                modules += [nn.Conv2d(*pair, kernel_size=3)]
            else:
                # applying EncodeBlock
                modules += [EncoderBlock(*pair, n_convs_per_layer)]
        self.sequence = nn.Sequential(*modules)

    def forward(self, in_batch: torch.Tensor) -> List[torch.Tensor]:
        print('Calculating in encoder')
        print(f'Is cuda: {in_batch.is_cuda}')
        _features = []  # type: List
        _shapes = []  # type: List
        _indices = []  # type: List
        for i in range(self.n_blocks):
            if i == 0:
                in_batch = self.sequence[i](in_batch)
            else:
                in_batch, indices = self.sequence[i](in_batch)
                _indices += [indices]
            _features += [in_batch]
            _shapes += [in_batch.shape]
        return _features, _indices, _shapes


class Gaussian(nn.Module):

    """Performs axis-aligned Gaussian"""

    def __init__(self,
                 latent_dim: int,
                 num_channels: int ,
                 nonlinearity: nn.Module = nn.ReLU(),
                 num_convs_per_block: int = 3,
                 down_sampling_op: Union[None, Callable]=None) -> None:
        super(Gaussian, self).__init__()
        
        if down_sampling_op is None:
            self.down_sampling_op = nn.AvgPool2d(kernel_size=2, stride=(2,2)) #padding='SAME' - ?
    
        # batch dim := 0
        # channel dim := 1
        # spatial dims := [2, 3]

        self._latent_dim = latent_dim
        self._encoder = VGGEncoder(in_channels = 1, 
                                    start_out_channels = 64, 
                                    n_blocks = 5, 
                                    n_convs_per_layer = 3)
        
    def forward(self,
                img: torch.Tensor, 
                seg: Union[torch.Tensor, None]=None) -> torch.Tensor:
        if seg is not None:
            seg = seg.float()
            img = torch.cat([img, seg], dim=1)
            self._encoder = VGGEncoder(in_channels = 2, 
                                       start_out_channels = 64, 
                                       n_blocks = 5, 
                                       n_convs_per_layer = 3)

        encoding = self._encoder(img)[0][-1]
        # encoding = torch.mean(encoding, dim=self._spatial_axes, keepdims=True)
        # print(encoding)
        # enc = encoding[0]
        # print('ENCODING[-1]:',enc)
        encoding = torch.mean(encoding, dim=3, keepdim=True)
        encoding = torch.mean(encoding, dim=2, keepdim=True)

        mu_log_sigma = nn.Conv2d(in_channels = encoding.shape[1], out_channels=2 * self._latent_dim, 
                                 kernel_size=1, 
                                 stride=1, 
                                 dilation=1)(encoding)

        
        # mu_log_sigma = torch.squeeze(mu_log_sigma, axis=self._spatial_axes)
        mu_log_sigma = torch.squeeze(mu_log_sigma, dim=3)
        mu_log_sigma = torch.squeeze(mu_log_sigma, dim=2)
        mu = mu_log_sigma[:, :self._latent_dim]
        log_sigma = mu_log_sigma[:, self._latent_dim:]
        
        _fn = distributions.multivariate_normal.MultivariateNormal
        s = torch.exp(log_sigma)
        b = torch.eye(s.size(1))
        c = s.unsqueeze(2).expand(*s.size(), s.size(1))
        d = c * b
        return _fn(mu, d)


class Conv1x1Decoder(nn.Module):

    """Performs 1x1 convolutions"""

    def __init__(self,num_classes,num_channels,num_1x1_convs,nonlinearity=nn.ReLU()):
        super(Conv1x1Decoder, self).__init__()

        self._num_classes = num_classes
        self._num_channels = num_channels
        self._num_1x1_convs = num_1x1_convs
        self._nonlinearity = nonlinearity
        self._spatial_axes = [2, 3]

        # batch dim := 0
        # channel dim := 1
        # spatial dims := [2, 3]

    def forward(self, 
                features: torch.Tensor, 
                z: torch.Tensor) -> torch.Tensor:
        
        shp = features.shape
        spatial_shape = [shp[axis] for axis in self._spatial_axes]
        multiples = [1] + spatial_shape
        multiples.insert(1, 1)

        if len(z.shape) == 2:
            z = torch.unsqueeze(z, dim=2)
            z = torch.unsqueeze(z, dim=2)

        # broadcast latent vector to spatial dimensions of the image/feature tensor
        broadcast_z = z.repeat(multiples) #tf.tile(z, multiples) 
        
        features = torch.cat([features, broadcast_z], dim=1)
        for _ in range(self._num_1x1_convs):
            features = nn.Conv2d(in_channels = features.shape[1], out_channels = 1,#in_channels=self._num_channels, 
                                 kernel_size=1,
                                 stride=1, 
                                 dilation=1)(features) # rate equivalent to dilation?
            features = self._nonlinearity(features)

        logits = nn.Conv2d(in_channels=features.shape[1], out_channels=self._num_classes, kernel_size=1, stride=1, dilation=1)        
        return logits(features)
    
    
    
class ProbUNet(nn.Module):

    """Performs Probabilistic Unet"""

    def __init__(self,
                 num_channels: int,
                 num_classes: int,
                 latent_dim: int = 6,
                 nonlinearity: nn.Module = nn.ReLU(),
                 num_1x1_convs: int = 3,
                 num_convs_per_block: int = 3) -> None:
        super(ProbUNet, self).__init__()

        self.num_channels = num_channels
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.uses_cuda = False
        
        print('num channels and num classes: ', num_channels, num_classes)

        #self.unet = SegNet(in_channels=1, out_classes=1)
        self.unet = unet_github.UNet(num_classes = num_classes, in_channels = 1)
        
        params = dict(num_channels=num_channels,
                      latent_dim = latent_dim, 
                      nonlinearity=nonlinearity,
                      num_convs_per_block=num_convs_per_block)
        self.prior_net = Gaussian(**params)
        self.posterior_net = Gaussian(**params)
        
        del params['num_convs_per_block']
        del params['latent_dim']
        params['num_1x1_convs'] = num_1x1_convs
        params['num_classes'] = num_classes
        self.f_comb = Conv1x1Decoder(**params)

        # print('Unet ', list(self.unet.parameters()))

    def forward(self,
                img: torch.Tensor,
                seg: Union[None, torch.Tensor]=None) -> None:

        """Docstring"""
        print('IMG and SEG SHAPES: ',img.shape, seg.shape)
        if self.training:
            if seg is not None:
                pass
            print('Started calculating posterior')
            print(f'seg.is_cuda: {seg.is_cuda}, img.is_cuda: {img.is_cuda}')
            print(f'posterion_net.is_cuda: {self.posterior_net.is_cuda}')
            self.q = self.posterior_net(img, seg)
            print('Done!')
        print('Started calculation prior')
        self.p = self.prior_net(img)
        print('Done!')
        self.unet_features = self.unet(img)

    def reconstruct(self,
                    use_posterior_mean: bool = True,
                    z_q: Union[None, torch.Tensor]=None) -> torch.Tensor:

        """Docstring"""

        if use_posterior_mean:
            z_q = self.q.loc
        else:
            if z_q is None:
                z_q = self.q.sample()
        return self.f_comb(self.unet_features, z_q)

    def sample(self) -> torch.Tensor:

        """Docstring"""

        z_p = self.p.sample()
        return self.f_comb(self.unet_features, z_p)

    def kl(self,
           analytic: bool = True,
           z_q: Union[None, torch.Tensor]=None) -> torch.Tensor:

        """Docstring"""

        if analytic:
            _fn = distributions.kl.kl_divergence
            kl = _fn(self.q, self.p)
        else:
            if z_q is None:
                z_q = self.q.sample()
            log_q = self.q.log_prob(z_q)
            log_p = self.p.log_prob(z_q)
            kl = log_q - log_p
        return kl

    def elbo(self, 
             seg: torch.Tensor,
             beta: float = 1.0,
             analytic_kl: bool = True,
             reconstruct_posterior_mean: bool=False,
             z_q: Union[None, torch.Tensor]=None,
             one_hot_labels: bool = True,
             loss_mask: Union[None, torch.Tensor]=None) -> torch.Tensor:

        """Docstring"""

        if z_q is None:
            z_q = self.q.sample()

        self.kl = torch.mean(self.kl(analytic_kl, z_q))

        params = dict(use_posterior_mean=reconstruct_posterior_mean, z_q=z_q)
        self.rec_logits = self.reconstruct(**params)

        params = dict(labels=seg,
                      logits=self.rec_logits,
                      n_classes=self.num_classes,
                     one_hot_labels = False)
        rec_loss = ce_loss(**params)
        self.rec_loss = rec_loss['sum']
        self.rec_liss_mean = rec_loss['mean']

        return -(self.rec_loss + beta * self.kl)