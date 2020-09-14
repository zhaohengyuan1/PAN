import torch
import models.archs.PAN_arch as PAN_arch
import models.archs.SRResNet_arch as SRResNet_arch
import models.archs.RCAN_arch as RCAN_arch


# Generator
def define_G(opt):
    opt_net = opt['network_G']
    which_model = opt_net['which_model_G']

    # image restoration
    if which_model == 'PAN':
        netG = PAN_arch.PAN(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'],
                                       nf=opt_net['nf'], unf=opt_net['unf'], nb=opt_net['nb'], scale=opt_net['scale'])
    elif which_model == 'SRResNet_PA':
        netG = SRResNet_arch.SRResNet_PA(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'],
                                       nf=opt_net['nf'], nb=opt_net['nb'], upscale=opt_net['scale'])
    else:
        raise NotImplementedError('Generator model [{:s}] not recognized'.format(which_model))

    return netG
