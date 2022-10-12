import torch.nn as nn
import torch
from typing import Tuple
from ..utils.types import TensorList
from abc import ABCMeta, abstractmethod
import numpy as np
from pytorch_lightning import LightningModule


class Constraint(LightningModule, metaclass=ABCMeta):
    def __init__(self) -> None:
        """Abstract baseclass for the constraint module."""
        super().__init__()
        self.sparsity_masks = None

    def forward(self, input: Tuple[TensorList, TensorList]):
        """The forward pass of the constraint module applies the sparsity mask to the
        feature matrix theta, and then calculates the coefficients according to the
        method in the child.

        Args:
            input (Tuple[TensorList, TensorList]): (time_derivs, library) tuple of size
                    ([(n_samples, 1) X n_outputs], [(n_samples, n_features) x n_outputs]).
        Returns:
            coeff_vectors (TensorList): List with coefficient vectors of size ([(n_features, 1) x n_outputs])
        """

        time_derivs, thetas = input

        if self.sparsity_masks is None:
            self.sparsity_masks = [torch.ones(theta.shape[1], dtype=torch.bool, device=self.device) for theta in thetas]

        sparse_thetas = self.apply_mask(thetas, self.sparsity_masks)

        # Constraint grad. desc style doesn't allow to change shape, so we return full coeff
        # and multiply by mask to set zeros. For least squares-style, we need to put in
        # zeros in the right spot to get correct shape.
        coeff_vectors = self.fit(sparse_thetas, time_derivs)
        self.coeff_vectors = [
            self.map_coeffs(mask, coeff)
            if mask.shape[0] != coeff.shape[0]
            else coeff * mask[:, None]
            for mask, coeff in zip(self.sparsity_masks, coeff_vectors)
        ]

        return self.coeff_vectors

    @staticmethod
    def apply_mask(thetas: TensorList, masks: TensorList):
        """Applies the sparsity mask to the feature (library) matrix.

        Args:
            thetas (TensorList): List of all library matrices of size [(n_samples, n_features) x n_outputs].

        Returns:
            TensorList: The sparse version of the library matrices of size [(n_samples, n_active_features) x n_outputs].
        """
        sparse_thetas = [theta[:, mask] for theta, mask in zip(thetas, masks)]
        return sparse_thetas

    def map_coeffs(self, mask: torch.Tensor, coeff_vector: torch.Tensor) -> torch.Tensor:
        """Places the coeff_vector components in the true positions of the mask.
        I.e. maps ((0, 1, 1, 0), (0.5, 1.5)) -> (0, 0.5, 1.5, 0).

        Args:
            mask (torch.Tensor): Boolean mask describing active components.
            coeff_vector (torch.Tensor): Vector with active-components.

        Returns:
            mapped_coeffs (torch.Tensor): mapped coefficients.
        """
        mapped_coeffs = (
            torch.zeros((mask.shape[0], 1), device=self.device).masked_scatter_(mask[:, None], coeff_vector)
        )
        return mapped_coeffs

    @abstractmethod
    def fit(self, sparse_thetas: TensorList, time_derivs: TensorList) -> TensorList:
        """Abstract method. Specific method should return the coefficients as calculated from the sparse feature
        matrices and temporal derivatives.

        Args:
            sparse_thetas (TensorList): List containing the sparse feature tensors of size (n_samples,
            n_active_features).
            time_derivs (TensorList): List containing the time derivatives of size (n_samples, n_outputs).
self.log("REG_TRAIN", reg_loss, prog_bar=True, logger=True)
        Returns:
            (TensorList): Calculated coefficients of size (n_active_features, n_outputs).
        """
        raise NotImplementedError


class Estimator(LightningModule, metaclass=ABCMeta):
    def __init__(self) -> None:
        """Abstract baseclass for the sparse estimator module."""
        super().__init__()
        self.coeff_vectors = None

    def forward(self, thetas: TensorList, time_derivs: TensorList):
        """The forward pass of the sparse estimator module first normalizes the library matrices
        and time derivatives by dividing each column (i.e. feature) by their l2 norm, than calculate the coefficient
        vectors
        according to the sparse estimation algorithm supplied by the child and finally returns the sparsity
        mask (i.e. which terms are active) based on these coefficients.

        Args:
            thetas (TensorList): List containing the sparse feature tensors of size  [(n_samples, n_active_features)
            x n_outputs].
            time_derivs (TensorList): List containing the time derivatives of size  [(n_samples, 1) x n_outputs].

        Returns:
            (TensorList): List containting the sparsity masks of a boolean type and size  [(n_samples, n_features) x
            n_outputs].
        """

        # we first normalize theta and the time deriv
        with torch.no_grad():
            normed_time_derivs = [
                (time_deriv / torch.norm(time_deriv)).detach().cpu().numpy()
                for time_deriv in time_derivs
            ]
            normed_thetas = [
                (theta / torch.norm(theta, dim=0, keepdim=True)).detach().cpu().numpy()
                for theta in thetas
            ]
        self.coeff_vectors = [
            self.fit(theta, time_deriv.squeeze())[:, None]
            for theta, time_deriv in zip(normed_thetas, normed_time_derivs)
        ]

        sparsity_masks = [
            torch.tensor(coeff_vector != 0.0, dtype=torch.bool, device=self.device).squeeze()  # move to gpu if required
            for coeff_vector in self.coeff_vectors
        ]

        return sparsity_masks

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Abstract method. Specific method should compute the coefficient based on feature matrix X and observations y.
        Note that we expect X and y to be numpy arrays, i.e. this module is non-differentiable.

        Args:
            x (np.ndarray): Feature matrix of size (n_samples, n_features)
            y (np.ndarray): observations of size (n_samples, n_outputs)

        Returns:
            (np.ndarray): Coefficients of size (n_samples, n_outputs)
        """
        pass


class Library(LightningModule):
    def __init__(self) -> None:
        """Abstract baseclass for the library module."""
        super().__init__()
        # self.w_array = torch.Tensor(requires_grad=False)
        self.norms = None

    def forward(self, input):
        """Compute the library (time derivatives and thetas) from a given dataset. Also calculates the norms
        of these, later used to calculate the normalized coefficients.

        Args:
            input (Tuple[TensorList, TensorList]): (prediction, data) tuple of size ((n_samples, n_outputs),
            (n_samples, n_dims))

        Returns:
            Tuple[TensorList, TensorList]: Temporal derivative and libraries of size ([(n_samples, 1) x n_outputs]),
            [(n_samples, n_features)x n_outputs])
            :param input:
            :param w_array:
        """

        time_derivs, thetas = self.library(input)

        self.norms = [
            (torch.norm(time_deriv) / torch.norm(theta, dim=0, keepdim=True)).detach().squeeze()
            for time_deriv, theta in zip(time_derivs, thetas)
        ]
        return time_derivs, thetas

    @abstractmethod
    def library(self, input):
        """Abstract method. Specific method should calculate the temporal derivative and feature matrices.
        These should be a list; one temporal derivative and feature matrix per output.

        Args:
        input (Tuple[TensorList, TensorList]): (prediction, data) tuple of size ((n_samples, n_outputs), (n_samples,
        n_dims))

        Returns:
        Tuple[TensorList, TensorList]: Temporal derivative and libraries of size ([(n_samples, 1) x n_outputs]),
        [(n_samples, n_features)x n_outputs])
        """
        pass


from deepymod.data.base_lightning import Dataset
from deepymod.model.func_approx import NN, Siren, FeedForward, Convo
from deepymod.model.constraint import LeastSquares
from deepymod.model.sparse_estimators_lightning import Threshold, PDEFIND
from deepymod.training.sparsity_scheduler import Periodic
from deepymod.model.custom_library import CustomTemperatureLibrary, Library2D, FiniteDiff
from deepymod.model.library import Library2D
from deepymod.model.Jia_lib import Library_nonlinear


class DeepMoD(LightningModule):
    def __init__(
            self,
            config,
            X_train,
            Y_train,
            X_test,
            Y_test,
            **kwargs

    ) -> None:
        """The DeepMoD class integrates the various buiding blocks into one module. The function approximator
        approximates the data,
        the library than builds a feature matrix from its output and the constraint constrains these. The sparsity
        estimator is called
        during training to update the sparsity mask (i.e. which terms the constraint is allowed to use.)

        Args:
            function_approximator (torch.nn.Sequential): [description]
            library (Library): [description]
            sparsity_estimator (Estimator): [description]
            constraint (Constraint): [description]
        """
        super().__init__()

        self.save_hyperparameters()

        # CONFIG VARIABLES INITIALIZATION
        self.config = config
        self.learning_rate = self.config["learning_rate"]
        self.batch_size_ratio = int(self.config["batch_size_ratio"])
        self.periodicity = self.config["periodicity"]
        self.grid_size = 17 ** 2

        # DEEPMOD CLASSES INITIALIZATION
        self.func_approx = Convo()
        self.library = FiniteDiff()
        self.sparse_estimator = Threshold(0.1)
        self.constraint = LeastSquares()
        self.sparsity_scheduler = Periodic(self.periodicity, initial_iteration=2 * self.periodicity)

        # VARIABLES INITIALIZATION
        self.X_train = X_train
        self.X_test = X_test
        self.Y_train = Y_train
        self.Y_test = Y_test
        self.x_test_size = len(X_test)
        self.last_prediction = None,
        self.last_time_deriv = None,
        self.last_theta = None,
        self.last_val_loss = None,
        self.initial_losses = None
        self.previous_losses = None
        self.x_train_size = len(X_train)
        self.batch_size = int(self.x_train_size / self.batch_size_ratio)

    def forward(self, input: torch.Tensor) -> Tuple[torch.Tensor, TensorList, TensorList]:
        """The forward pass approximates the data, builds the time derivative and feature matrices
        and applies the constraint.

        It returns the prediction of the network, the time derivatives and the feature matrices.

        Args:
            input (torch.Tensor):  Tensor of shape (n_samples, n_outputs) containing the coordinates, first column
            should be the time coordinate.

        Returns:
            Tuple[torch.Tensor, TensorList, TensorList]: The prediction, time derivatives and and feature matrices of
            respective sizes
                                                       ((n_samples, n_outputs), [(n_samples, 1) x n_outputs]),
                                                       [(n_samples, n_features) x n_outputs])
                                                       :param input:
                                                       :param w_array:

        """

        prediction, coordinates = self.func_approx(input)
        time_derivs, thetas = self.library((prediction, coordinates))
        coeff_vectors = self.constraint((time_derivs, thetas))
        return prediction, time_derivs, thetas

    @property
    def sparsity_masks(self):
        """Returns the sparsity masks which contain the active terms."""
        return self.constraint.sparsity_masks

    def estimator_coeffs(self) -> TensorList:
        """Calculate the coefficients as estimated by the sparse estimator.

        Returns:
            (TensorList): List of coefficients of size [(n_features, 1) x n_outputs]
        """
        coeff_vectors = self.sparse_estimator.coeff_vectors
        return coeff_vectors

    def constraint_coeffs(self, scaled=False, sparse=False):
        """Calculate the coefficients as estimated by the constraint.

        Args:
            scaled (bool): Determine whether or not the coefficients should be normalized
            sparse (bool): Whether to apply the sparsity mask to the coefficients.

        Returns:
            (TensorList): List of coefficients of size [(n_features, 1) x n_outputs]
        """
        coeff_vectors = self.constraint.coeff_vectors
        if scaled:
            coeff_vectors = [
                coeff / norm[:, None]
                for coeff, norm, mask in zip(
                    coeff_vectors, self.library.norms, self.sparsity_masks
                )
            ]
        if sparse:
            coeff_vectors = [
                sparsity_mask[:, None] * coeff
                for sparsity_mask, coeff in zip(self.sparsity_masks, coeff_vectors)
            ]
        return coeff_vectors

    def compute_loss(self, prediction, target, time_derivs, thetas):

        self.k_reg = 1
        self.k_mse = 1

        mse_loss = self.k_mse * torch.mean((prediction - target) ** 2)

        reg_loss = self.k_reg * torch.stack([torch.mean((dt - theta @ coef_vector) ** 2) for dt, theta, coef_vector in
                                             zip(time_derivs, thetas,
                                                 self.constraint_coeffs(scaled=False, sparse=True))])

        total_loss = mse_loss + reg_loss
        return total_loss, mse_loss, reg_loss

    def compute_loss_validation(self, prediction, target):

        loss = torch.mean((prediction - target) ** 2)
        return loss

    def in_house_softmax(self, input):

        # calculating the softmax function using torch tensors
        # normalizing by the max to get better stability
        z = input - torch.max(input)
        expo = torch.exp(z)
        result = torch.tensor(len(z) * expo / (torch.sum(torch.exp(z)) + self.epsilon), dtype=torch.float32,
                              device=self.device)  # this returns tensor with each entry being the correspondent lambda for each sub loss
        # result = len(z) * torch.tensor([torch.exp(z[i])/torch.sum(torch.exp(z)) for i in range(len(z))],
        # dtype=torch.float32, device=self.device)
        return result

    def training_step(self, batch, batch_idx):

        x, target = batch
        prediction, time_derivs, thetas = self(x)  # self(x) is the same as doing self.forward(x)
        prediction = torch.reshape(prediction, target.shape)
        self.last_prediction = torch.clone(prediction)
        self.last_theta = [torch.clone(theta) for theta in
                           thetas]  # don't remember why is this not vectorised and with a for?
        self.last_time_deriv = [torch.clone(time_deriv) for time_deriv in time_derivs]  # same as the above
        total_loss, mse_loss, reg_loss = self.compute_loss(prediction=prediction, target=target,
                                                           time_derivs=time_derivs, thetas=thetas)

        # logging
        self.log("MSE Loss", mse_loss, prog_bar=True, logger=True, on_epoch=True, on_step=False)
        self.log("PDE Loss", reg_loss, prog_bar=True, logger=True, on_epoch=True, on_step=False)
        self.log("Train Loss", total_loss, prog_bar=True, logger=True, on_epoch=True, on_step=False)

        return total_loss

    def on_epoch_end(self):
        apply_sparsity = self.sparsity_scheduler(self.current_epoch)
        if apply_sparsity and (self.current_epoch != 0) and (
                self.current_epoch > self.sparsity_scheduler.initial_iteration):
            self.constraint.sparsity_masks = self.sparse_estimator(self.last_theta,
                                                                   self.last_time_deriv)  # need to check

        # logging coefs
        coeffs = self.constraint_coeffs()[0]
        self.log("First Coeff", coeffs[0])
        self.log("Second Coeff", coeffs[1])

    def validation_step(self, batch, batch_idx, dataloader_idx):
        _input, target = batch
        prediction = self.func_approx(_input)[0]
        val_loss = self.compute_loss_validation(prediction=prediction, target=target)
        self.last_val_loss = val_loss

        prog_bar_list = [True, False, False, True, False, False, True, False, False, True, True, True]
        self.log(f"val_loss_{self.time_span[dataloader_idx]}", val_loss, on_epoch=True, logger=True,
                 prog_bar=prog_bar_list[dataloader_idx], on_step=False, add_dataloader_idx=False)

        return val_loss

    def configure_optimizers(self):

        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, betas=(0.99, 0.99), amsgrad=True)
        return {"optimizer": optimizer}

    def train_dataloader(self):

        dataset = Dataset(data=self.X_train, labels=self.Y_train)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size,
                                                 drop_last=True, num_workers=0, shuffle=True)

        return dataloader

    def val_dataloader(self):

        self.time_span = [1, 2, 3, 7, 14, 21, 30, 60, 90, 120, 360,int(len(self.X_test) / (17 * 17))]  # nr of days in each validation subset
        dataloader_list = []
        for j, i in enumerate(self.time_span):
            X_test_aux, Y_test_aux = torch.clone(self.X_test[:int(self.grid_size * i), :]), torch.clone(
                self.Y_test[:int(self.grid_size * i), :])
            val_dataset = Dataset(data=X_test_aux, labels=Y_test_aux)
            dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=len(self.X_test), num_workers=0,
                                                     drop_last=True)
            dataloader_list.append(dataloader)
        return dataloader_list
