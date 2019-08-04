import math
import torch
import torch.nn as nn
import torchvision
from . import block as B
from . import spectral_norm as SN
from . import architecture

class DualSR_RRDB(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb_e,nb_l_1,nb_l_2,nb_h_1,nb_h_2,nb_m,gc=32, upscale=4, norm_type=None, \
            act_type='leakyrelu', mode='CNA', upsample_mode='upconv'):
        super(DualSR_RRDB, self).__init__()
        n_upscale = int(math.log(upscale, 2))
        if upscale == 3:
            n_upscale = 1

        fea_conv = B.conv_block(in_nc, nf, kernel_size=3, norm_type=None, act_type=None)
        rb_blocks_e = [B.RRDB(nf, kernel_size=3, gc=32, stride=1, bias=True, pad_type='zero', \
            norm_type=norm_type, act_type=act_type, mode='CNA') for _ in range(nb_e)]

        rb_blocks_l1 = [B.RRDB(nf, kernel_size=3, gc=32, stride=1, bias=True, pad_type='zero', \
                              norm_type=norm_type, act_type=act_type, mode='CNA') for _ in range(nb_l_1)]
        rb_blocks_l2 = [B.RRDB(nf, kernel_size=3, gc=32, stride=1, bias=True, pad_type='zero', \
                              norm_type=norm_type, act_type=act_type, mode='CNA') for _ in range(nb_l_2)]
        rb_blocks_h1 = [B.RRDB(nf, kernel_size=3, gc=32, stride=1, bias=True, pad_type='zero', \
                              norm_type=norm_type, act_type=act_type, mode='CNA') for _ in range(nb_h_1)]

        rb_blocks_concat = B.conv_block(nf*2, nf, kernel_size=3, norm_type=None, act_type=None)


        rb_blocks_h2= [B.RRDB(nf, kernel_size=3, gc=32, stride=1, bias=True, pad_type='zero', \
                              norm_type=norm_type, act_type=act_type, mode='CNA') for _ in range(nb_h_2)]

        rb_blocks_M = [B.RRDB(nf, kernel_size=3, gc=32, stride=1, bias=True, pad_type='zero', \
                              norm_type=norm_type, act_type=act_type, mode='CNA') for _ in range(nb_m)]

        self.body_ex = B.sequential(fea_conv, B.ShortcutBlock(B.sequential(*rb_blocks_e)))

        self.body_l1 = B.sequential(B.ShortcutBlock(B.sequential(*rb_blocks_l1)))
        self.body_l2 = B.sequential(B.ShortcutBlock(B.sequential(*rb_blocks_l2)))
        self.body_h1 = B.sequential(B.ShortcutBlock(B.sequential(*rb_blocks_h1)))
        self.concat = rb_blocks_concat
        self.body_h2 = B.sequential(B.ShortcutBlock(B.sequential(*rb_blocks_h2)))
        self.body_m = B.sequential(B.ShortcutBlock(B.sequential(*rb_blocks_M)))




        self.LR_conv_l = B.conv_block(nf, nf, kernel_size=3, norm_type=norm_type, act_type=None, mode=mode)

        self.LR_conv_h = B.conv_block(nf, nf, kernel_size=3, norm_type=norm_type, act_type=None, mode=mode)

        self.LR_conv_M = B.conv_block(nf, nf, kernel_size=3, norm_type=norm_type, act_type=None, mode=mode)


        if upsample_mode == 'upconv':
            upsample_block = B.upconv_blcok
        elif upsample_mode == 'pixelshuffle':
            upsample_block = B.pixelshuffle_block
        else:
            raise NotImplementedError('upsample mode [{:s}] is not found'.format(upsample_mode))
        if upscale == 3:
            self.upsampler_l = B.sequential(*upsample_block(nf, nf, 3, act_type=act_type))
            self.upsampler_h = B.sequential(*upsample_block(nf, nf, 3, act_type=act_type))
            self.upsampler_m = B.sequential(*upsample_block(nf, nf, 3, act_type=act_type))
        else:
            self.upsampler_l = B.sequential(*[upsample_block(nf, nf, act_type=act_type) for _ in range(n_upscale)])
            self.upsampler_h = B.sequential(*[upsample_block(nf, nf, act_type=act_type) for _ in range(n_upscale)])
            self.upsampler_m = B.sequential(*[upsample_block(nf, nf, act_type=act_type) for _ in range(n_upscale)])

        self.HR_conv0_l = B.conv_block(nf, nf, kernel_size=3, norm_type=None, act_type=act_type)
        self.HR_conv0_h = B.conv_block(nf, nf, kernel_size=3, norm_type=None, act_type=act_type)
        self.HR_conv0_m = B.conv_block(nf, nf, kernel_size=3, norm_type=None, act_type=act_type)

        self.HR_conv1_l = B.conv_block(nf, out_nc, kernel_size=3, norm_type=None, act_type=None)
        self.HR_conv1_h = B.conv_block(nf, out_nc, kernel_size=3, norm_type=None, act_type=None)
        self.HR_conv1_m = B.conv_block(nf, out_nc, kernel_size=3, norm_type=None, act_type=None)

        # self.model = B.sequential(fea_conv, B.ShortcutBlock(B.sequential(*rb_blocks, LR_conv)),\
        #     *upsampler, HR_conv0, HR_conv1)

    def forward(self, x):
        # head
        x = self.body_ex(x)
        #low

        x_guide = self.body_l1(x)
        x_l = self.body_l2(x_guide)
        x_l = self.LR_conv_l(x_l)
        x_l = self.upsampler_l(x_l)
        x_fea_l =self.HR_conv0_l(x_l)
        x_l = self.HR_conv1_l(x_fea_l)

        #high

        x_h = self.body_h1(x)

        x_h = self.concat(torch.cat((x_guide, x_h), 1))
        x_h = self.body_h2(x_h)
        x_h = self.LR_conv_h(x_h)
        x_h = self.upsampler_h(x_h)
        x_fea_h = self.HR_conv0_h(x_h)
        x_h = self.HR_conv1_h(x_fea_h)

        # mask
        m = self.body_m(x)
        m = self.LR_conv_M(m)
        m = self.upsampler_m(m)
        m = self.HR_conv0_m(m)
        M_sigmoid = torch.sigmoid(m)

        combine = M_sigmoid.mul(x_fea_h)+(1-M_sigmoid).mul(x_fea_l)

        combine = self.HR_conv1_m(combine)



        return x_l,x_h,combine

    def load_state_dict(self, state_dict, strict=False):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') >= 0:
                        print('Replace pre-trained upsampler to new one...')
                    else:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if len(missing) > 0:
                raise KeyError('missing keys in state_dict: "{}"'.format(missing))


