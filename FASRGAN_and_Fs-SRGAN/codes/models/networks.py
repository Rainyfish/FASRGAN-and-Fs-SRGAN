import functools
import logging
import torch
import torch.nn as nn
from torch.nn import init

import models.modules.architecture as arch
import models.modules.sft_arch as sft_arch
import models.modules.rcan_e as rcan_e
import models.modules.rcan_g as rcan_g
import models.modules.DualSR as DualSR
import models.modules.DualSR_SR as DualSR_SR
import models.modules.DualSR_RRDB as DualSR_RRDB
logger = logging.getLogger('base')
####################
# initialize
####################


def weights_init_normal(m, std=0.02):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.normal_(m.weight.data, 0.0, std)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0.0, std)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, std)  # BN also uses norm
        init.constant_(m.bias.data, 0.0)


def weights_init_kaiming(m, scale=1):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        m.weight.data *= scale
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        m.weight.data *= scale
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.constant_(m.weight.data, 1.0)
        init.constant_(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.orthogonal_(m.weight.data, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.orthogonal_(m.weight.data, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.constant_(m.weight.data, 1.0)
        init.constant_(m.bias.data, 0.0)


def init_weights(net, init_type='kaiming', scale=1, std=0.02):
    # scale for 'kaiming', std for 'normal'.
    logger.info('Initialization method [{:s}]'.format(init_type))
    if init_type == 'normal':
        weights_init_normal_ = functools.partial(weights_init_normal, std=std)
        net.apply(weights_init_normal_)
    elif init_type == 'kaiming':
        weights_init_kaiming_ = functools.partial(weights_init_kaiming, scale=scale)
        net.apply(weights_init_kaiming_)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError('initialization method [{:s}] not implemented'.format(init_type))


####################
# define network
####################


# Generator
def define_G(opt):
    gpu_ids = opt['gpu_ids']
    opt_net = opt['network_G']
    which_model = opt_net['which_model_G']

    if which_model == 'sr_resnet':  # SRResNet
        netG = arch.SRResNet(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'], nf=opt_net['nf'], \
            nb=opt_net['nb'], upscale=opt_net['scale'], norm_type=opt_net['norm_type'], \
            act_type='relu', mode=opt_net['mode'], upsample_mode='pixelshuffle')

    elif which_model == 'sft_arch':  # SFT-GAN
        netG = sft_arch.SFT_Net()

    elif which_model == 'RRDB_net':  # RRDB
        netG = arch.RRDBNet(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'], nf=opt_net['nf'],
            nb=opt_net['nb'], gc=opt_net['gc'], upscale=opt_net['scale'], norm_type=opt_net['norm_type'],
            act_type='leakyrelu', mode=opt_net['mode'], upsample_mode='upconv')
    #define the ex_G
    elif which_model == 'RRDBNet_G':
        netG = arch.RRDBNet(in_nc=opt_net['nf'], out_nc=opt_net['out_nc'], nf=opt_net['nf'],
                            nb=opt_net['nb'], gc=opt_net['gc'], upscale=opt_net['scale'],
                            norm_type=opt_net['norm_type'],
                            act_type='leakyrelu', mode=opt_net['mode'], upsample_mode='upconv')
    elif which_model =='RCAN_G':
        netG =rcan_g.RCAN_G(n_resblocks=opt_net['n_resblocks'], n_resgroups=opt_net['n_resgroups'],
                            n_feats=opt_net['n_feats'], reduction=16, scale=4, n_colors=3, rgb_range=255, res_scale=1)
    elif which_model == 'DualSR_RCAN':
        #def __init__(self, n_resblocks, n_resgroups_mask, n_resgroups_share, n_resgroups_high_1, n_resgroups_high_2,
        #            n_resgroups_low1, n_resgroups_low2, n_feats, reduction, scale, n_colors, rgb_range, res_scale,
        #            conv=common.default_conv):

        netG =DualSR.DualSR(n_resblocks=opt_net['n_resblocks'], n_resgroups_mask=opt_net['n_resgroups_mask'],\
                                n_resgroups_share=opt_net['n_resgroups_share'], n_resgroups_high_1=opt_net['n_resgroups_high_1'], \
                                n_resgroups_high_2=opt_net['n_resgroups_high_2'], n_resgroups_low_1=opt_net['n_resgroups_low_1'], \
                                n_resgroups_low_2=opt_net['n_resgroups_low_2'],\
                                n_feats=opt_net['n_feats'], reduction=16, scale=4, n_colors=3, rgb_range=255, res_scale=1)
    elif which_model == 'DualSR_SR':
        # def __init__(self, n_resblocks, n_resgroups_mask, n_resgroups_share, n_resgroups_high_1, n_resgroups_high_2,
        #            n_resgroups_low1, n_resgroups_low2, n_feats, reduction, scale, n_colors, rgb_range, res_scale,
        #            conv=common.default_conv):

        netG = DualSR_SR.DualSR_SR(n_resblocks=opt_net['n_resblocks'], n_resgroups_mask=opt_net['n_resgroups_mask'], \
                             n_resgroups_share=opt_net['n_resgroups_share'],
                             n_resgroups_high_1=opt_net['n_resgroups_high_1'], \
                             n_resgroups_high_2=opt_net['n_resgroups_high_2'],
                             n_resgroups_low_1=opt_net['n_resgroups_low_1'], \
                             n_resgroups_low_2=opt_net['n_resgroups_low_2'], \
                             n_feats=opt_net['n_feats'], reduction=16, scale=4, n_colors=3, rgb_range=255, res_scale=1)
    elif which_model=='DualSR_RRDB':
        netG = DualSR_RRDB.DualSR_RRDB(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'], nf=opt_net['nf'],
                            nb_l_1=opt_net['nb_l_1'],nb_l_2=opt_net['nb_l_2'],nb_h_1=opt_net['nb_h_1'],nb_e=opt_net['nb_e'],
                            nb_h_2=opt_net['nb_h_2'],nb_m=opt_net['nb_m'], gc=opt_net['gc'], upscale=opt_net['scale'],
                            norm_type=opt_net['norm_type'],
                            act_type='leakyrelu', mode=opt_net['mode'], upsample_mode='upconv')

    else:
        raise NotImplementedError('Generator model [{:s}] not recognized'.format(which_model))

    if opt['is_train']:
        init_weights(netG, init_type='kaiming', scale=0.1)
    if gpu_ids:
        assert torch.cuda.is_available()
        netG = nn.DataParallel(netG)
    return netG

#new added
def define_E(opt):
    gpu_ids = opt['gpu_ids']
    opt_net = opt['network_E']
    which_model = opt_net['which_model_ex']

    if which_model == 'model_ex':
        netE = arch.model_ex(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'], nf=opt_net['nf'],
            nb=opt_net['nb'], gc=opt_net['gc'], upscale=opt_net['scale'], norm_type=opt_net['norm_type'],
            act_type='leakyrelu', mode=opt_net['mode'], upsample_mode='upconv')
    elif which_model =='RCAN_E':
        netE =rcan_e.RCAN_E(n_resblocks=opt_net['n_resblocks'], n_resgroups=opt_net['n_resgroups'], n_feats=opt_net['n_feats'], reduction=16, scale=4, n_colors=3, rgb_range=255, res_scale=1)
    else:
        raise NotImplementedError('Discriminator model [{:s}] not recognized'.format(which_model))
    init_weights(netE, init_type='kaiming', scale=1)
    if gpu_ids:
        netE = nn.DataParallel(netE)
    return netE



# Discriminator
def define_D(opt):
    gpu_ids = opt['gpu_ids']
    opt_net = opt['network_D']
    which_model = opt_net['which_model_D']

    if which_model == 'discriminator_vgg_128':
        netD = arch.Discriminator_VGG_128(in_nc=opt_net['in_nc'], base_nf=opt_net['nf'], \
            norm_type=opt_net['norm_type'], mode=opt_net['mode'], act_type=opt_net['act_type'])

    elif which_model == 'dis_acd':  # sft-gan, Auxiliary Classifier Discriminator
        netD = sft_arch.ACD_VGG_BN_96()

    elif which_model == 'discriminator_vgg_96':
        netD = arch.Discriminator_VGG_96(in_nc=opt_net['in_nc'], base_nf=opt_net['nf'], \
            norm_type=opt_net['norm_type'], mode=opt_net['mode'], act_type=opt_net['act_type'])
    elif which_model == 'discriminator_vgg_192':
        netD = arch.Discriminator_VGG_192(in_nc=opt_net['in_nc'], base_nf=opt_net['nf'], \
            norm_type=opt_net['norm_type'], mode=opt_net['mode'], act_type=opt_net['act_type'])
    elif which_model == 'discriminator_vgg_128_SN':
        netD = arch.Discriminator_VGG_128_SN()
    elif which_model=='Unet':
        netD = arch.UNet()
    #define the ex_d
    elif which_model=='discriminator_vgg_192_ex':
        netD = arch.Discriminator_VGG_192(in_nc=opt_net['nf'], base_nf=opt_net['nf'], \
            norm_type=opt_net['norm_type'], mode=opt_net['mode'],act_type=opt_net['act_type'])
    elif which_model == 'discriminator_vgg_128_ex':
        netD = arch.Discriminator_VGG_128(in_nc=opt_net['nf'], base_nf=opt_net['nf'], \
                                          norm_type=opt_net['norm_type'], mode=opt_net['mode'],
                                          act_type=opt_net['act_type'])
    else:
        raise NotImplementedError('Discriminator model [{:s}] not recognized'.format(which_model))

    init_weights(netD, init_type='kaiming', scale=1)
    if gpu_ids:
        netD = nn.DataParallel(netD)
    return netD


def define_F(opt, use_bn=False):
    gpu_ids = opt['gpu_ids']
    device = torch.device('cuda' if gpu_ids else 'cpu')
    # pytorch pretrained VGG19-54, before ReLU.
    if use_bn:
        feature_layer = 49
    else:
        feature_layer = 34
    netF = arch.VGGFeatureExtractor(feature_layer=feature_layer, use_bn=use_bn, \
        use_input_norm=True, device=device)
    # netF = arch.ResNet101FeatureExtractor(use_input_norm=True, device=device)
    if gpu_ids:
        netF = nn.DataParallel(netF)
    netF.eval()  # No need to train
    return netF
