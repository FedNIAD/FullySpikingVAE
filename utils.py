import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import pycuda.driver as cuda
import pycuda.autoinit  # Necessary for using its functions
import fsvae_models.snn_layers as snn_layers
from typing import Any, Dict, List
import argparse
import copy


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class aboutCudaDevices():
    def __init__(self):
        pass

    def num_devices(self):
        """Return number of devices connected."""
        return cuda.Device.count()

    def devices(self):
        """Get info on all devices connected."""
        num = cuda.Device.count()
        print("%d device(s) found:" % num)
        for i in range(num):
            print(cuda.Device(i).name(), "(Id: %d)" % i)

    def mem_info(self):
        """Get available and total memory of all devices."""
        available, total = cuda.mem_get_info()
        print("Available: %.2f GB\nTotal:     %.2f GB" % (available / 1e9, total / 1e9))

    def attributes(self, device_id=0):
        """Get attributes of device with device Id = device_id"""
        return cuda.Device(device_id).get_attributes()

    def info(self):
        """Class representation as number of devices connected and about them."""
        num = cuda.Device.count()
        string = ""
        string += ("%d device(s) found:\n" % num)
        for i in range(num):
            string += ("    %d) %s (Id: %d)\n" % ((i + 1), cuda.Device(i).name(), i))
            string += ("          Memory: %.2f GB\n" % (cuda.Device(i).total_memory() / 1e9))
        return string

class CountMulAddANN:
    def __init__(self) -> None:
        self.mul_sum = 0
        self.add_sum = 0
    def __call__(self, module, module_in, module_out):
        
        if isinstance(module_in, tuple):
            module_in = module_in[0]
        if isinstance(module_out, tuple):
            module_out = module_out[0]

        if not module.training:
            with torch.no_grad():
                if isinstance(module, torch.nn.Conv2d):
                    s_in = module_in.shape
                    s_out = module_in.shape
                    mul = s_in[0]*s_in[1]*s_in[2]*s_in[3] * module.kernel_size[0] * module.kernel_size[1] * module.out_channels / (module.stride[0]*module.stride[1])
                    add = mul + s_out[0]*s_out[1]*s_out[2]*s_out[3] # 掛け合わせた分だけ足す必要がある + bias
                elif isinstance(module, torch.nn.Linear):
                    s_in = module_in.shape
                    s_out = module_in.shape
                    mul = s_in[0]*s_in[1]*s_out[1]
                    add = mul + s_out[0]*s_out[1]
                elif isinstance(module, torch.nn.ConvTranspose2d):
                    s_in = module_in.shape
                    s_out = module_in.shape
                    mul = s_in[0]*s_in[1]*s_in[2]*s_in[3] * module.kernel_size[0] * module.kernel_size[1] * module.out_channels * (module.stride[0]*module.stride[1])
                    add = mul + s_out[0]*s_out[1]*s_out[2]*s_out[3]
                else:
                    add = 0
                    mul = 0
                
                self.mul_sum = self.mul_sum + mul
                self.add_sum = self.add_sum + add

    def clear(self):
        self.mul_sum = 0
        self.add_sum = 0

class CountMulAddSNN:
    def __init__(self) -> None:
        self.mul_sum = 0
        self.add_sum = 0
    def __call__(self, module, module_in, module_out):
        
        if isinstance(module_in, tuple):
            module_in = module_in[0]
        if isinstance(module_out, tuple):
            module_out = module_out[0]

        if not module.training:
            with torch.no_grad():
                if isinstance(module, torch.nn.Conv3d):
                    if module.is_first_conv:
                        # real-value images are input to the first conv layer.
                        s_in = module_in.shape
                        s_out = module_in.shape
                        mul = s_in[0]*s_in[1]*s_in[2]*s_in[3]*s_in[4] * module.kernel_size[0] * module.kernel_size[1] * module.out_channels / (module.stride[0]*module.stride[1])
                        add = mul + s_out[0]*s_out[1]*s_out[2]*s_out[3]*s_out[4] # calc of bias
                    else:
                        add = module_in.sum() * module.kernel_size[0] * module.kernel_size[1] * module.out_channels / (module.stride[0]*module.stride[1])
                        s = module_out.shape # (N,C,H,W,T)
                        add += s[0] * s[1] * s[2] * s[3] * s[4] # calc of bias
                        mul = 0
                elif isinstance(module, torch.nn.Linear):
                    add = module_in.sum() * module.out_features
                    s = module_out.shape # (N,C,T)
                    add += s[0] * s[1] * s[2]
                    mul = 0
                elif isinstance(module, torch.nn.ConvTranspose3d):
                    add = module_in.sum() * module.kernel_size[0] * module.kernel_size[1] * module.out_channels * module.stride[0]*module.stride[1]
                    s = module_out.shape # (N,C,H,W,T)
                    add += s[0] * s[1] * s[2] * s[3] * s[4]
                    mul = 0
                elif isinstance(module, snn_layers.LIFSpike):
                    s_in = module_in.shape
                    if len(s_in) == 5: # conv layer
                        add = s_in[0] * s_in[1] * s_in[2] * s_in[3] * s_in[4]
                    elif len(s_in) == 3: # linear layer
                        add = s_in[0] * s_in[1] * s_in[2]
                    else:
                        raise ValueError()
                    mul = (1-module_out).sum() # event-based activation
                else:
                    add = 0
                    mul = 0
                
                self.mul_sum = self.mul_sum + mul
                self.add_sum = self.add_sum + add

    def clear(self):
        self.mul_sum = 0
        self.add_sum = 0

def average_weights(weights: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    weights_avg = copy.deepcopy(weights[0])

    for key in weights_avg.keys():
        for i in range(1, len(weights)):
            weights_avg[key] += weights[i][key]
        weights_avg[key] = torch.div(weights_avg[key], len(weights))

    return weights_avg

def arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_root", type=str, default="../datasets/")
    parser.add_argument("--model_name", type=str, default="cnn")

    parser.add_argument("--non_iid", type=int, default=0)  # 0: IID, 1: Non-IID
    parser.add_argument("--n_clients", type=int, default=4)
    parser.add_argument("--n_shards", type=int, default=200)
    parser.add_argument("--frac", type=float, default=0.1)

    parser.add_argument("--n_epochs", type=int, default=150)
    parser.add_argument("--n_client_epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--optim", type=str, default="sgd")
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--log_every", type=int, default=1)
    parser.add_argument("--early_stopping", type=int, default=1)

    parser.add_argument("--device", type=int, default=0)

    parser.add_argument("--wandb", type=bool, default=False)
    parser.add_argument("--wandb_project", type=str, default="FedAvg")
    parser.add_argument("--exp_name", type=str, default="exp")
    parser.add_argument('-config', action='store', dest='config', help='The path of config file',
                        default='NetworkConfigs/MVTEC.yaml')
    parser.add_argument('-name', type=str, default='FEDAVG')

    return parser.parse_args()