from .dualenc import DualEncoderEpsNetwork
from .diffusion import *

def get_model(config):
    if config.network == 'dualenc':
        return DualEncoderEpsNetwork(config)
    else:
        raise NotImplementedError('Unknown network: %s' % config.network)
