## ECCV-2018-Image Super-Resolution Using Very Deep Residual Channel Attention Networks
## https://arxiv.org/abs/1807.02758
from models.modules import common

import torch.nn as nn

# def make_model(args, parent=False):
#     return RCAN_E(args)

## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

## Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(
        self, conv, n_feat, kernel_size, reduction,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        #res = self.body(x).mul(self.res_scale)
        res += x
        return res

## Residual Group (RG)
class ResidualGroup(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, act, res_scale, n_resblocks):
        super(ResidualGroup, self).__init__()
        modules_body = []
        modules_body = [
            RCAB(
                conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(True), res_scale=1) \
            for _ in range(n_resblocks)]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res

## Residual Channel Attention Network (RCAN)
class DualSR_SR(nn.Module):
    def __init__(self,n_resblocks,n_resgroups_mask,n_resgroups_share,n_resgroups_high_1,n_resgroups_high_2,n_resgroups_low_1,n_resgroups_low_2,n_feats,reduction,scale,n_colors,rgb_range,res_scale, conv=common.default_conv):
        super(DualSR_SR, self).__init__()

        # n_resgroups = n_resgroups
        n_resblocks = n_resblocks
        n_feats = n_feats
        kernel_size = 3
        reduction = reduction
        scale = 4
        act = nn.ReLU(True)
        
        # RGB mean for DIV2K
        # self.sub_mean = common.MeanShift(args.rgb_range)
        
        # define head module
        modules_head = [conv(n_colors, n_feats, kernel_size)]

        # define body module
        modules_share = [
            ResidualGroup(
                conv, n_feats, kernel_size, reduction, act=act, res_scale=res_scale, n_resblocks=n_resblocks) \
            for _ in range(n_resgroups_share)]

        # modules_body_share.append(conv(n_feats, n_feats, kernel_size))

        # modules_high_1=[ResidualGroup(
        #         conv, n_feats, kernel_size, reduction, act=act, res_scale=res_scale, n_resblocks=n_resblocks) \
        #     for _ in range(n_resgroups_high_1)]
        #
        # modules_high_2 = [ResidualGroup(
        #     conv, n_feats, kernel_size, reduction, act=act, res_scale=res_scale, n_resblocks=n_resblocks) \
        #     for _ in range(n_resgroups_high_2)]
        # modules_high_2.append(conv(n_feats, n_feats, kernel_size))
        #
        # modules_tail_high = [
        #     common.Upsampler(conv, scale, n_feats, act=False),
        #     conv(n_feats, n_colors, kernel_size)]

        modules_low_1 = [ResidualGroup(
            conv, n_feats, kernel_size, reduction, act=act, res_scale=res_scale, n_resblocks=n_resblocks) \
            for _ in range(n_resgroups_low_1)]

        modules_low_2 = [ResidualGroup(
            conv, n_feats, kernel_size, reduction, act=act, res_scale=res_scale, n_resblocks=n_resblocks) \
            for _ in range(n_resgroups_low_2)]
        modules_low_2.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        modules_tail_low = [
            common.Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, n_colors, kernel_size)]

        # modules_Mask= [ResidualGroup(
        #     conv, n_feats, kernel_size, reduction, act=act, res_scale=res_scale, n_resblocks=n_resblocks) \
        #     for _ in range(n_resgroups_mask)]
        # modules_Mask.append(conv(n_feats, n_feats, kernel_size))
        #
        # modules_tail_Mask = [
        #     common.Upsampler(conv, scale, n_feats, act=False),
        #     conv(n_feats, n_colors, kernel_size)]

        # self.add_mean = common.MeanShift(args.rgb_range, sign=1)
        #share part
        self.head = nn.Sequential(*modules_head)
        self.share = nn.Sequential(*modules_share)
        #high frequence. divide two parts for letting the guide feature in
        # self.high_1 = nn.Sequential(*modules_high_1)
        # self.high_2 =nn.Sequential(*modules_high_2)
        # self.tail_high=nn.Sequential(*modules_tail_high)
        #low frequence. divide two parts for getting the guide feature
        self.low1=nn.Sequential(*modules_low_1)
        self.low2=nn.Sequential(*modules_low_2)
        self.tail_low = nn.Sequential(*modules_tail_low)
        # mask part. need to up scaling for same spatial size of the HR image. Mask attention
        # self.Mask = nn.Sequential(*modules_Mask)
        # self.tail_mask = nn.Sequential(*modules_tail_Mask)

    def forward(self, x):
        # x = self.sub_mean(x)
        x = self.head(x)
        res = self.share(x)
        ##low frequence(PSNR loss part)
        x_guide = self.low1(res) #the guide feature
        #skipping connection in low frequence block
        x_low = self.low2(x_guide)+res
        #global skipping connection
        x_low += x
        low = self.tail_low(x_low)

        #high frequence(GAN and perceptual loss part)
        x_guide_in = self.high_1(res) #the position of the guide features in
        # skipping connection in high frequence block
        x_high = self.high_2(x_guide_in+x_guide)+res
        # x_high = self.high_2(x_guide_in)+res

        # global skipping connection
        x_high += x
        high = self.tail_high(x_high)

        #Mask
        # mask = self.Mask(res)+res
        # mask += x
        # mask = self.tail_mask(mask)


        # # res = self.body(x)
        # res += x

        # x = self.tail(res)
        # x = self.add_mean(x)

        return high

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
