import numpy as np
import torch
from torch.autograd import grad
from itertools import combinations
from functools import reduce
from ..model.deepmod_lightning import Library
from typing import Tuple
from ..utils.types import TensorList
import torch.nn as nn


class CustomTemperatureLibrary(Library):

    def __init__(self):
        super().__init__()
        self.w_array = None
        self.w_x = torch.tensor((-3.2097), device=self.device, dtype=torch.float32)
        self.w_y = torch.tensor((-5.2696), device=self.device, dtype=torch.float32)

    def library(self, input):
        T, data = input

        dT = grad(T, data, grad_outputs=torch.ones_like(T), create_graph=True)[0]
        dT_dt = dT[:, 0:1]
        dT_dx = dT[:, 1:2]
        dT_dy = dT[:, 2:3]

        dT2 = grad(dT_dx, data, grad_outputs=torch.ones_like(T), create_graph=True)[0]
        dT_dxx = dT2[:, 1:2]
        dT_dyy = grad(dT_dy, data, grad_outputs=torch.ones_like(T), create_graph=True)[0][:, 2:3]

        first_term = (dT_dxx + dT_dyy)
        # with omega being the average
        second_term = dT_dx * self.w_x + dT_dy * self.w_y

        # with omega being time dependent
        # second_term = dT_dx * self.w_array[:, 0].reshape(-1, 1) + dT_dy * self.w_array[:, 1].reshape(-1, 1)

        theta = torch.cat((first_term, second_term), dim=1)

        return [dT_dt], [theta]


class Library2D(Library):
    def __init__(self, poly_order: int) -> None:
        """Create a 2D library up to given polynomial order with second order derivatives
         i.e. for poly_order=1: [$1, u_x, u_y, u_{xx}, u_{yy}, u_{xy}$]
        Args:
            poly_order (int): maximum order of the polynomial in the library
        """
        super().__init__()
        self.poly_order = poly_order

    def library(
            self, input: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[TensorList, TensorList]:
        """Compute the library for the given a prediction and data

        Args:
            input (Tuple[torch.Tensor, torch.Tensor]): A prediction and its data

        Returns:
            Tuple[TensorList, TensorList]: The time derivatives and the thetas
            computed from the library and data.
        """

        prediction, data = input
        # Polynomial

        u = torch.ones_like(prediction, device=self.device)
        for order in np.arange(1, self.poly_order + 1):
            u = torch.cat((u, u[:, order - 1: order] * prediction), dim=1)

        # Gradients
        du = grad(
            prediction,
            data,
            grad_outputs=torch.ones_like(prediction),
            create_graph=True,
        )[0]
        u_t = du[:, 0:1]
        u_x = du[:, 1:2]
        u_y = du[:, 2:3]
        du2 = grad(
            u_x, data, grad_outputs=torch.ones_like(prediction), create_graph=True
        )[0]
        u_xx = du2[:, 1:2]
        u_xy = du2[:, 2:3]
        u_yy = grad(
            u_y, data, grad_outputs=torch.ones_like(prediction), create_graph=True
        )[0][:, 2:3]

        du = torch.cat((torch.ones_like(u_x), u_x, u_y, u_xx, u_yy, u_xy), dim=1)

        samples = du.shape[0]
        # Bringing it together
        theta = torch.matmul(u[:, :, None], du[:, None, :]).view(samples, -1)

        return [u_t], [theta]


class StringPDE(Library):
    """Compute the library of terms that come from the wave equation and particularly the wave equation with an
    horizontal speed(v).
    """

    def __init__(self, c, v,full_lib):
        super().__init__()
        self.c = c
        self.v = v
        self.full_lib = full_lib

    def library(self, input):

        prediction, coords = input  # data is the (temporal location,spatial location) (t,x)
        # prediction is (n_samples,n_outputs), which in this case is (batch_size,1) because we are only prediction 1
        # variable
        print(prediction)
        print(coords)

        # since we only have 1 output and 2 inputs the Jacobian will be by Nx2
        du = \
            grad(outputs=prediction, inputs=coords, grad_outputs=torch.ones_like(prediction), create_graph=True)[0]

        time_deriv = du[:, 0:1]  # this syntax is just to make it 1 column. if du[:,0] this would have 0 column
        du_dx = du[:, 1:2]       # since the input is (t,x) 2nd colum is partial derivative in respect to x

        du2 = grad(outputs=du_dx, inputs=coords, grad_outputs=torch.ones_like(prediction), create_graph=True)[0]

        du2_dt_dx = du2[:, 0:1]
        du2_d2x = du2[:, 1:2]

        du2_dt2 = \
            grad(outputs=time_deriv, inputs=coords, grad_outputs=torch.ones_like(prediction), create_graph=True)[0]

        second_time_deriv = du2_dt2[:, 0:1]
        du2_dx_dt = du2_dt2[:, 1:2]
        eps = 1e-4
        # if torch.sum(du2_dx_dt - du2_dt_dx) < eps:
        #     print("Derivatives seem fine")
        # else:
        #     print("Derivatives maybe are wrong")
        #
        # polynomial terms
        # u = library_poly(prediction[:], self.poly_order)

        # now computing theta, which is a mix of polynomial and derivative terms
        # but in this case I know which terms will appear in the PDE, so I will just place those 1st and then add the
        # mix terms with derivatives, 1st by hand and then I build them with cycles

        mix_term1   = torch.mul(time_deriv, du_dx)
        mix_term2   = torch.mul(time_deriv, du2_d2x)
        mix_term3   = torch.mul(second_time_deriv, du_dx)
        mix_term4   = torch.mul(second_time_deriv, du2_d2x)
        square_term = prediction**2
        cube_term   = prediction**3

        # pde terms
        term1 = (self.c ** 2 - self.v ** 2) * du2_d2x
        term2 = (2 * self.v * du2_dt_dx)

        pde_terms = torch.cat((term1, term2), dim=1).view(du_dx.shape[0], -1)


        theta = torch.cat((pde_terms, mix_term1, mix_term2, mix_term3, mix_term4, square_term, cube_term), dim=1).view(
            prediction.shape[0], -1)

        if self.full_lib:
            return [second_time_deriv], [theta]
        else:
            return [second_time_deriv], [pde_terms]
