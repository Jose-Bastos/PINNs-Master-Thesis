import numpy as np
import torch
from torch.autograd import grad
from itertools import combinations
from functools import reduce
from ..model.deepmod_lightning import Library
from typing import Tuple
from ..utils.types import TensorList

class Library_nonlinear(Library):
    """[summary]

    Args:
        Library ([type]): [description]
    """
    def __init__(self, period_list) -> None:
        super().__init__()
        self.period_list = period_list

        self.cal_count = 0

        self.all_ele_name = []

    def library(self, input):
        prediction, data = input
        samples = prediction.shape[0]

        n_order = int((data.shape[1] - 1)/2.0)

        # Construct the theta matrix
        C = torch.ones_like(prediction[:, 0]).view(samples, -1)

        x = data[:, 1:2]
        y = data[:, 2:3]

        xy_list = [x, y, torch.pow(x, 2), torch.pow(y, 2), torch.mul(x, y)]

        xy_name = ['x', 'y', 'x^2', 'y^2', 'xy']

        time_array = data[:, 0:1]
        sincos_list = []
        sincos_name = []
        for i in range(len(self.period_list)):
            omega = 2.0*np.pi/self.period_list[i]
            sincos_list.append(torch.sin(time_array*omega))
            sincos_list.append(torch.cos(time_array*omega))

            sincos_name.append(f'sw{i}')
            sincos_name.append(f'cw{i}')

        # Construct time_derivatives
        # Only consider one output ATM

        dT = grad(prediction, data, grad_outputs=torch.ones_like(prediction),
                  create_graph=True, allow_unused=True)[0]

        time_deriv = dT[:, 0:1]
        time_deriv_list = [time_deriv]

        dT_dx = dT[:, 1:2]
        dT_dy = dT[:, 2:3]

        dT2 = grad(dT_dx, data, grad_outputs=torch.ones_like(prediction),
                   create_graph=True, allow_unused=True)[0]

        dT_dxx = dT2[:, 1:2]
        dT_dxy = dT2[:, 2:3]

        dT_dyy = grad(dT_dy, data, grad_outputs=torch.ones_like(prediction),
                      create_graph=True, allow_unused=True)[0][:, 2:3]

        dT_list = [dT_dx, dT_dy, dT_dxx, dT_dyy, dT_dxy]
        dT_name = ['dTdx', 'dTdy', 'd2Td2x', 'd2Td2y', 'd2Tdxdy']

        coupling_list = []
        coupling_name = []
        for xy_term in xy_list + dT_list + [time_array]:
            for sc_term in sincos_list:
                coupling_list. append(torch.mul(xy_term, sc_term))

        for ele in xy_name + dT_name + ['t']:
            for sc in sincos_name:
                coupling_name.append(f'{ele}*{sc}')

        self.all_ele_name = ['1'] + xy_name + dT_name + sincos_name + \
            coupling_name

        theta = torch.cat([C] + xy_list + dT_list + sincos_list +
                          coupling_list, dim=1)


        return time_deriv_list, [theta]