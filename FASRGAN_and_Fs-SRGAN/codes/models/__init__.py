import logging
logger = logging.getLogger('base')


def create_model(opt):
    model = opt['model']

    if model == 'sr':
        from .SR_model import SRModel as M
    elif model == 'FsSRModel':
        from  .FsSR_model import FsSRModel as M
    elif model == 'srgan':
        from .SRGAN_model import SRGANModel as M
    elif model == 'srragan':
        from .SRRaGAN_model import SRRaGANModel as M
    elif model == 'sftgan':
        from .SFTGAN_ACD_model import SFTGAN_ACD_Model as M
    elif model =='FASRGAN':  #Fine-grained attention SRGAN
        from .FASRGAN_model import FASRGANModel as M
    elif model =='FsSRGAN': #feature-sharing SRGAN
        from .FsSRGAN_model import FsSRGANModel as M
    elif model == 'FAFS_SRGAN':
        from .FAFS_SRGAN_model import FAFS_SRGANModel as M
    else:
        raise NotImplementedError('Model [{:s}] not recognized.'.format(model))
    m = M(opt)
    logger.info('Model [{:s}] is created.'.format(m.__class__.__name__))
    return m
