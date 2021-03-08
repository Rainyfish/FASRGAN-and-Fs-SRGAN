import os
import logging
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.optim import lr_scheduler

import models.networks as networks
from .base_model import BaseModel

logger = logging.getLogger('base')


class FsSRModel(BaseModel):
    def __init__(self, opt):
        super(FsSRModel, self).__init__(opt)
        train_opt = opt['train']

        # define network and load pretrained models
        self.netE = networks.define_E(opt).to(self.device)
        self.netG = networks.define_G(opt).to(self.device)
        self.load()

        if self.is_train:
            self.netE.train()
            self.netG.train()

            # loss
            loss_type = train_opt['pixel_criterion']
            if loss_type == 'l1':
                self.cri_pix = nn.L1Loss().to(self.device)
            elif loss_type == 'l2':
                self.cri_pix = nn.MSELoss().to(self.device)
            else:
                raise NotImplementedError('Loss type [{:s}] is not recognized.'.format(loss_type))
            self.l_pix_w = train_opt['pixel_weight']

            # optimizers
            wd_G = train_opt['weight_decay_G'] if train_opt['weight_decay_G'] else 0
            optim_params = []

            for k, v in self.netE.named_parameters():  # can optimize for a part of the model
                if v.requires_grad:
                    optim_params.append(v)
                else:
                    logger.warning('Params [{:s}] will not optimize.'.format(k))

            for k, v in self.netG.named_parameters():  # can optimize for a part of the model
                if v.requires_grad:
                    optim_params.append(v)
                else:
                    logger.warning('Params [{:s}] will not optimize.'.format(k))
            self.optimizer_G = torch.optim.Adam(
                optim_params, lr=train_opt['lr_G'], weight_decay=wd_G)
            self.optimizers.append(self.optimizer_G)

            # schedulers
            if train_opt['lr_scheme'] == 'MultiStepLR':
                for optimizer in self.optimizers:
                    self.schedulers.append(lr_scheduler.MultiStepLR(optimizer, \
                        train_opt['lr_steps'], train_opt['lr_gamma']))
            else:
                raise NotImplementedError('MultiStepLR learning rate scheme is enough.')

            self.log_dict = OrderedDict()
        # print network
        self.print_network()

    def feed_data(self, data, need_HR=True):
        self.var_L = data['LR'].to(self.device)  # LR
        if need_HR:
            self.real_H = data['HR'].to(self.device)  # HR

    def feed_data2(self, data, need_HR=True):
        self.var_L = data['LR']  # LR
        if need_HR:
            self.real_H = data['HR']  # HR

    def optimize_parameters(self, step):
        self.optimizer_G.zero_grad()
        self.fake_feature = self.netE(self.var_L)
        self.fake_H = self.netG(self.fake_feature)
        l_pix = self.l_pix_w * self.cri_pix(self.fake_H, self.real_H)
        l_pix.backward()
        self.optimizer_G.step()

        # set log
        self.log_dict['l_pix'] = l_pix.item()

    def test(self):
        self.netE.eval()
        self.netG.eval()
        with torch.no_grad():
            self.fake_feature = self.netE(self.var_L)
            self.fake_H = self.netG(self.fake_feature)
        self.netE.train()
        self.netG.train()


    # def test_chop(self, shave=10, min_size=160000):
    #     self.netE.eval()
    #     self.netG.eval()
    #     for k, v in self.netE.named_parameters():
    #         v.requires_grad = False
    #     for k, v in self.netG.named_parameters():
    #         v.requires_grad = False
    #
    #     scale = 4
    #     n_GPUs = min(self.n_GPUs, 4)
    #     # height, width
    #     h, w = self.var_L.size()[-2:]
    #
    #     top = slice(0, h//2 + shave)
    #     bottom = slice(h - h//2 - shave, h)
    #     left = slice(0, w//2 + shave)
    #     right = slice(w - w//2 - shave, w)
    #     x_chops = [torch.cat([
    #         a[..., top, left],
    #         a[..., top, right],
    #         a[..., bottom, left],
    #         a[..., bottom, right]
    #     ]) for a in args]
    #
    #     y_chops = []
    #     if h * w < 4 * min_size:
    #         for i in range(0, 4, n_GPUs):
    #             x = [x_chop[i:(i + n_GPUs)] for x_chop in x_chops]
    #             y = P.data_parallel(self.model, *x, range(n_GPUs))
    #             if not isinstance(y, list): y = [y]
    #             if not y_chops:
    #                 y_chops = [[c for c in _y.chunk(n_GPUs, dim=0)] for _y in y]
    #             else:
    #                 for y_chop, _y in zip(y_chops, y):
    #                     y_chop.extend(_y.chunk(n_GPUs, dim=0))
    #     else:
    #         for p in zip(*x_chops):
    #             y = self.forward_chop(*p, shave=shave, min_size=min_size)
    #             if not isinstance(y, list): y = [y]
    #             if not y_chops:
    #                 y_chops = [[_y] for _y in y]
    #             else:
    #                 for y_chop, _y in zip(y_chops, y): y_chop.append(_y)
    #
    #     h *= scale
    #     w *= scale
    #     top = slice(0, h//2)
    #     bottom = slice(h - h//2, h)
    #     bottom_r = slice(h//2 - h, None)
    #     left = slice(0, w//2)
    #     right = slice(w - w//2, w)
    #     right_r = slice(w//2 - w, None)
    #
    #     # batch size, number of color channels
    #     b, c = y_chops[0][0].size()[:-2]
    #     y = [y_chop[0].new(b, c, h, w) for y_chop in y_chops]
    #     for y_chop, _y in zip(y_chops, y):
    #         _y[..., top, left] = y_chop[0][..., top, left]
    #         _y[..., top, right] = y_chop[1][..., top, right_r]
    #         _y[..., bottom, left] = y_chop[2][..., bottom_r, left]
    #         _y[..., bottom, right] = y_chop[3][..., bottom_r, right_r]
    #
    #     if len(y) == 1: y = y[0]
    #
    #     return y

    def test_x8(self):
        # from https://github.com/thstkdgus35/EDSR-PyTorch
        self.netG.eval()
        for k, v in self.netG.named_parameters():
            v.requires_grad = False

        def _transform(v, op):
            # if self.precision != 'single': v = v.float()
            v2np = v.data.cpu().numpy()
            if op == 'v':
                tfnp = v2np[:, :, :, ::-1].copy()
            elif op == 'h':
                tfnp = v2np[:, :, ::-1, :].copy()
            elif op == 't':
                tfnp = v2np.transpose((0, 1, 3, 2)).copy()

            ret = torch.Tensor(tfnp).to(self.device)
            # if self.precision == 'half': ret = ret.half()

            return ret

        lr_list = [self.var_L]
        for tf in 'v', 'h', 't':
            lr_list.extend([_transform(t, tf) for t in lr_list])
        sr_list = [self.netG(aug) for aug in lr_list]
        for i in range(len(sr_list)):
            if i > 3:
                sr_list[i] = _transform(sr_list[i], 't')
            if i % 4 > 1:
                sr_list[i] = _transform(sr_list[i], 'h')
            if (i % 4) % 2 == 1:
                sr_list[i] = _transform(sr_list[i], 'v')

        output_cat = torch.cat(sr_list, dim=0)
        self.fake_H = output_cat.mean(dim=0, keepdim=True)

        for k, v in self.netG.named_parameters():
            v.requires_grad = True
        self.netG.train()

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_HR=True):
        out_dict = OrderedDict()
        out_dict['LR'] = self.var_L.detach()[0].float().cpu()
        out_dict['SR'] = self.fake_H.detach()[0].float().cpu()
        if need_HR:
            out_dict['HR'] = self.real_H.detach()[0].float().cpu()
        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.netE)
        if isinstance(self.netE, nn.DataParallel):
            net_struc_str = '{} - {}'.format(self.netE.__class__.__name__,
                                             self.netE.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netE.__class__.__name__)

        logger.info('Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
        logger.info(s)

        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
                                             self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)

        logger.info('Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
        logger.info(s)

    def load(self):
        load_path_E = self.opt['path']['pretrain_model_E']
        if load_path_E is not None:
            logger.info('Loading pretrained model for G [{:s}] ...'.format(load_path_E))
            self.load_network(load_path_E, self.netE)

        load_path_G = self.opt['path']['pretrain_model_G']
        if load_path_G is not None:
            logger.info('Loading pretrained model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG)

    def save(self, iter_step):
        self.save_network(self.netE, 'E', iter_step)
        self.save_network(self.netG, 'G', iter_step)
