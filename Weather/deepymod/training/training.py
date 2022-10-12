import torch
import os
from .convergence import Convergence
from ..model.deepmod import DeepMoD
from torch.utils.data import DataLoader
from ..utils.logger import Logger
import matplotlib.pyplot as plt


def train(
        model: DeepMoD,
        train_dataloader: DataLoader,
        test_dataloader: DataLoader,
        optimizer,
        sparsity_scheduler,
        split: float = 0.8,
        exp_ID: str = None,
        log_dir: str = None,
        max_iterations: int = 10000,
        write_iterations: int = 25,
        reg_coef: float = 0.5,
        **convergence_kwargs
) -> None:
    """Trains the DeepMoD model. This function automatically splits the data set in a train and test set.
    Args:
        model (DeepMoD):  A DeepMoD object.
        data (torch.Tensor):  Tensor of shape (n_samples x (n_spatial + 1)) containing the coordinates, first column should be the time coordinate.
        target (torch.Tensor): Tensor of shape (n_samples x n_features) containing the target data.
        optimizer ([type]):  Pytorch optimizer.
        sparsity_scheduler ([type]):  Decides when to update the sparsity mask.
        split (float, optional):  Fraction of the train set, by default 0.8.
        exp_ID (str, optional): Unique ID to identify tensorboard file. Not used if log_dir is given, see pytorch documentation.
        log_dir (str, optional): Directory where tensorboard file is written, by default None.
        max_iterations (int, optional): [description]. Max number of epochs , by default 10000.
        write_iterations (int, optional): [description]. Sets how often data is written to tensorboard and checks train loss , by default 25.
    """
    # write checkpoint to same folder as tb output.
    logger = Logger(exp_ID= None, log_dir="/")
    #n_features = train_dataloader[0][1].shape[-1]
    n_features = 1
    # n_features = model.func_approx.modules()
    # Training
    loss_list = []
    mse_list = []
    reg_list = []
    mse_test_list = []
    convergence = Convergence(**convergence_kwargs)
    for iteration in torch.arange(0, max_iterations):
        # Training variables defined as: loss, mse, regularisation
        batch_losses = torch.zeros((3, n_features, len(train_dataloader)), device="cpu" )
        for batch_idx, train_sample in enumerate(train_dataloader):
            data_train, target_train = train_sample
            # ================== Training Model ============================
            prediction, time_derivs, thetas = model(data_train)
            batch_losses[1, :, batch_idx] = torch.mean((prediction - target_train) ** 2, dim=-2)  # loss per output
            batch_losses[2, :, batch_idx] = torch.stack([torch.mean((dt - theta @ coeff_vector) ** 2)
                                                         for dt, theta, coeff_vector in zip(
                    time_derivs,
                    thetas,
                    model.constraint_coeffs(scaled=False, sparse=True),
                )
                                                         ]
                                                        )
            batch_losses[0, :, batch_idx] = (
                    batch_losses[1, :, batch_idx] + reg_coef*batch_losses[2, :, batch_idx]
            )

            # Optimizer step
            optimizer.zero_grad()
            batch_losses[0, :, batch_idx].sum().backward()
            optimizer.step()

        loss, mse, reg = torch.mean(batch_losses.cpu().detach(), axis=-1)


        if iteration % write_iterations == 0:
            # ================== Validation costs ================
            with torch.no_grad():
                batch_mse_test = torch.zeros(
                    (n_features, len(test_dataloader)), device="cpu"
                )
                for batch_idx, test_sample in enumerate(test_dataloader):
                    data_test, target_test = test_sample
                    prediction_test = model.func_approx(data_test)[0]
                    batch_mse_test[:, batch_idx] = torch.mean(
                        (prediction_test - target_test) ** 2, dim=-2
                    )  # loss per output
            mse_test = batch_mse_test.cpu().detach().mean(dim=-1)
            # ====================== Logging =======================
            _ = model.sparse_estimator(
                thetas, time_derivs
            )  # calculating estimator coeffs but not setting mask
            logger(
                iteration,
                loss.view(-1).mean(),
                mse.view(-1),
                reg.view(-1),
                model.constraint_coeffs(sparse=True, scaled=True),
                model.constraint_coeffs(sparse=True, scaled=False),
                model.estimator_coeffs(),
                MSE_test=mse_test,
            )

            print()
            loss_list.append(loss)
            mse_list.append(mse)
            mse_test_list.append(mse_test)
            reg_list.append(reg)

            # ================== Sparsity update =============#
            # Updating sparsity
            update_sparsity = sparsity_scheduler(
                iteration, torch.sum(mse_test), model, optimizer
            )
            if update_sparsity:
                model.constraint.sparsity_masks = model.sparse_estimator(
                    thetas, time_derivs
                )

            # ================= Checking convergence
            l1_norm = torch.sum(
                torch.abs(
                    torch.cat(model.constraint_coeffs(sparse=True, scaled=True), dim=1)
                )
            )
            converged = convergence(iteration, l1_norm)
            if converged:
                break

    plt.figure(figsize=(15, 5))
    plt.plot(loss_list,label="Training Loss")
    plt.plot(mse_test_list,label="Test Loss")
    #plt.plot(mse_list)
    #plt.plot(reg_list)
    plt.grid(True)
    plt.legend()
    plt.show()
    logger.close(model)
