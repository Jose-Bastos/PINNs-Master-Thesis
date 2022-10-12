""" Contains the train module that governs training Deepymod """
import torch
from .convergence import Convergence
from ..model.deepmod_new import DeepMoD
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


def train(
        model: DeepMoD,
        train_dataloader: DataLoader,
        test_dataloader: DataLoader,
        optimizer,
        sparsity_scheduler,
        split: float = 0.8,
        max_epochs: int = 10000,
        write_epochs: int = 25,
        reg_coef: float = 1,
        k_mse_loss: float = 1,
        device: str = {},
        **convergence_kwargs
) -> None:
    """Trains the DeepMoD model. This function automatically splits the data set in a train and test set.

    Args:
        model (DeepMoD):  A DeepMoD object.
        data (torch.Tensor):  Tensor of shape (n_samples x (n_spatial + 1)) containing the coordinates, first column
        should be the time coordinate.
        target (torch.Tensor): Tensor of shape (n_samples x n_features) containing the target data.
        optimizer ([type]):  Pytorch optimizer.
        sparsity_scheduler ([type]):  Decides when to update the sparsity mask.
        split (float, optional):  Fraction of the train set, by default 0.8.
        exp_ID (str, optional): Unique ID to identify tensorboard file. Not used if log_dir is given, see pytorch
        documentation.
        log_dir (str, optional): Directory where tensorboard file is written, by default None.
        max_epochs (int, optional): [description]. Max number of epochs , by default 10000.
        write_epochs (int, optional): [description]. Sets how often data is written to tensorboard and checks
        train loss , by default 25.
    """
    # write checkpoint to same folder as tb output.
    n_features = 1

    # Training
    loss_list = []
    mse_list = []
    reg_list = []
    mse_test_list = []
    convergence = Convergence(**convergence_kwargs)
    model = model.to(device)
    for epoch in torch.arange(0, max_epochs):
        # Training variables defined as: loss, mse, regularisation
        batch_losses = torch.zeros((3, n_features, len(train_dataloader)), device=device)
        for batch_idx, (data_input, target_train) in enumerate(train_dataloader):
            # ================== Training Model ============================
            w_array = data_input[:, -2:]
            model.library.w_array = torch.clone(w_array)
            data_train = torch.clone(data_input[:, :3])
            prediction, time_derivs, thetas = model(data_train)
            mse_loss = k_mse_loss * torch.mean((prediction - target_train) ** 2)  # mse loss for the batch

            reg_loss = reg_coef * torch.stack(
                [torch.mean((dt - theta @ coeff_vector) ** 2) for dt, theta, coeff_vector in zip(
                    time_derivs,
                    thetas,
                    model.constraint_coeffs(scaled=False, sparse=True),
                )
                 ])  # regularization loss, loss because of the library terms

            total_loss = reg_loss + mse_loss
            batch_losses[2, :, batch_idx] = torch.clone(reg_loss)  # losses per batch
            batch_losses[1, :, batch_idx] = torch.clone(mse_loss)
            batch_losses[0, :, batch_idx] = torch.clone(total_loss)

            # Optimizer step
            optimizer.zero_grad()
            total_loss.sum().backward()
            optimizer.step()

        loss, mse, reg = torch.mean(batch_losses.cpu().detach(), axis=-1)  # mean losses for all batches

        if epoch % write_epochs == 0:
            # ================== Test Losses ================#
            with torch.no_grad():
                batch_mse_test = torch.zeros(
                    (n_features, len(test_dataloader)), device=device)
                for batch_idx, (test_sample, target_test) in enumerate(test_dataloader):
                    data_test = test_sample[:, :3]
                    prediction_test = model.func_approx(data_test)[0]
                    batch_mse_test[:, batch_idx] = torch.mean((prediction_test - target_test) ** 2,
                                                              dim=-2)  # loss per batch

            mse_test = batch_mse_test.cpu().detach().mean(dim=-1)  # mean loss for all batches

            # ====================== Logging =======================
            _ = model.sparse_estimator(
                thetas, time_derivs
            )  # calculating estimator coeffs but not setting mask

            logger(
                epoch,
                loss.view(-1),
                mse.view(-1),
                reg.view(-1),
                constraint_coeffs=model.constraint_coeffs(sparse=True, scaled=True),
                unscaled_constraint_coeffs=model.constraint_coeffs(sparse=True, scaled=False),
                estimator_coeffs=model.estimator_coeffs(),
                MSE_test=mse_test,
            )

            print()
            loss_list.append(loss)
            mse_list.append(mse)
            mse_test_list.append(mse_test)
            reg_list.append(reg)

            # ================== Sparsity update =============
            # Updating sparsity
            update_sparsity = sparsity_scheduler(
                epoch, torch.sum(mse_test), model, optimizer
            )
            if update_sparsity:
                model.constraint.sparsity_masks = model.sparse_estimator(
                    thetas, time_derivs
                )

            # ================= Checking convergence

            converged = convergence(epoch, mse_test=mse_test)
            if converged:
                break

    plt.figure(figsize=(15, 5))
    plt.plot(mse_list, label=" MSE Training Loss")
    plt.plot(loss_list, label="Total Training Loss")
    plt.plot(mse_test_list, label="Test Loss")
    plt.grid(True)
    plt.legend()
    plt.show()
    logger.close(model)
