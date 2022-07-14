import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable


class MaskedAvePooling(nn.Module):
    def __init__(self, size=1):
        super(MaskedAvePooling, self).__init__()
        self.pooling = nn.AdaptiveAvgPool2d(size)

    def forward(self, x, mask):
        if mask is None:
            return self.pooling(x)
        pooled_feat = self.pooling(x * mask.expand_as(x))
        total_pixel_num = mask.shape[-1] * mask.shape[-2]
        active_pixel_num = mask.view(x.shape[0], -1).sum(dim=1)
        active_mask = active_pixel_num.unsqueeze(dim=1).unsqueeze(dim=1).unsqueeze(dim=1).expand_as(pooled_feat) + 1e-4
        return (pooled_feat * total_pixel_num)/active_mask


class GumbelSoftmax(nn.Module):
    '''
        gumbel softmax gate.
    '''
    def __init__(self, eps=1):
        super(GumbelSoftmax, self).__init__()
        self.eps = eps
        self.sigmoid = nn.Sigmoid()
    
    def gumbel_sample(self, template_tensor, eps=1e-8):
        uniform_samples_tensor = template_tensor.clone().uniform_()
        gumble_samples_tensor = torch.log(uniform_samples_tensor+eps)-torch.log(
                                          1-uniform_samples_tensor+eps)
        return gumble_samples_tensor
    
    def gumbel_softmax(self, logits):
        """ Draw a sample from the Gumbel-Softmax distribution"""
        gsamples = self.gumbel_sample(logits.data)
        logits = logits + Variable(gsamples)
        soft_samples = self.sigmoid(logits / self.eps)
        return soft_samples, logits
    
    def forward(self, logits):
        if not self.training:
            out_hard = (logits>=0).float()
            return out_hard
        out_soft, prob_soft = self.gumbel_softmax(logits)
        out_hard = ((out_soft >= 0.5).float() - out_soft).detach() + out_soft
        return out_hard


class Mask_s(nn.Module):
    '''
        Attention Mask spatial.
    '''
    def __init__(self, h, w, planes, block_w, block_h, eps=0.66667, bias=-1, DPACS=False, **kwargs):
        super(Mask_s, self).__init__()
        # Parameter
        self.width, self.height, self.channel = w, h, planes
        self.mask_h, self.mask_w = int(np.ceil(h / block_h)), int(np.ceil(w / block_w))
        self.eleNum_s = torch.Tensor([self.mask_h*self.mask_w])
        # spatial attention
        if DPACS:
            self.atten_s = nn.Conv2d(planes, 1, kernel_size=1, stride=1, bias=bias>=0, padding=0)
        else:
            self.atten_s = nn.Conv2d(planes, 1, kernel_size=3, stride=1, bias=bias>=0, padding=1)

        if bias>=0:
            nn.init.constant_(self.atten_s.bias, bias)
        # Gate
        self.gate_s = GumbelSoftmax(eps=eps)
        # Norm
        self.norm = lambda x: torch.norm(x, p=1, dim=(1,2,3))
    
    def forward(self, x):
        batch, channel, height, width = x.size()
        # Pooling
        input_ds = F.adaptive_avg_pool2d(input=x, output_size=(self.mask_h, self.mask_w))
        # spatial attention
        s_in = self.atten_s(input_ds) # [N, 1, h, w]
        # spatial gate
        mask_s = self.gate_s(s_in) # [N, 1, h, w]
        # norm
        norm = self.norm(mask_s)
        norm_t = self.eleNum_s.to(x.device)
        return mask_s, norm, norm_t
    
    def get_flops(self):
        flops = self.mask_h * self.mask_w * self.channel * 9
        return flops


class Mask_c(nn.Module):
    '''
        Attention Mask.
    '''

    def __init__(self, inplanes, outplanes, fc_reduction=4, eps=0.66667, bias=-1, DPACS=False, **kwargs):
        super(Mask_c, self).__init__()
        # Parameter
        self.bottleneck = inplanes // fc_reduction
        self.inplanes, self.outplanes = inplanes, outplanes
        self.eleNum_c = torch.Tensor([outplanes])
        self.DPACS = DPACS
        # channel attention
        if not DPACS:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
            self.atten_c = nn.Sequential(
                nn.Conv2d(inplanes, self.bottleneck, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(self.bottleneck),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.bottleneck, outplanes, kernel_size=1, stride=1, bias=bias >= 0),
            )
            if bias >= 0:
                nn.init.constant_(self.atten_c[3].bias, bias)
        else:
            self.group_size = 64
            self.avg_pool = MaskedAvePooling()
            self.atten_c = nn.Sequential(
                nn.Linear(inplanes, outplanes // self.group_size)
            )

        # Gate
        self.gate_c = GumbelSoftmax(eps=eps)
        # Norm
        self.norm = lambda x: torch.norm(x, p=1, dim=(1, 2, 3))

    def forward(self, x, meta=None):
        batch, channel, _, _ = x.size()
        if self.DPACS:
            context = self.avg_pool(x, meta["saliency_mask"])
            c_in = self.atten_c(context.view(context.shape[0], -1))
            c_in = self.expand(c_in).view(context.shape[0], -1, 1, 1)
        else:
            context = self.avg_pool(x)  # [N, C, 1, 1]
            c_in = self.atten_c(context)
        # transform
        mask_c = self.gate_c(c_in)  # [N, C_out, 1, 1]
        # norm
        norm = self.norm(mask_c)
        norm_t = self.eleNum_c.to(x.device)
        return mask_c, norm, norm_t

    def get_flops(self):
        flops = self.inplanes * self.bottleneck + self.bottleneck * self.outplanes
        return flops

    def expand(self, x):
        bs, vec_size = x.shape
        return x.unsqueeze(dim=-1).expand(bs, vec_size, self.group_size).reshape(bs, vec_size*self.group_size)

