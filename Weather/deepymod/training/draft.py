# commom imports
import sys
import numpy as np
import pandas as pd
import os
import tqdm
from datetime import datetime
from matplotlib import pyplot as plt
import math
import pickle
import torch
import seaborn as sns
from sklearn import preprocessing
from torch.autograd import grad
import hvplot.pandas
import holoviews as hv

hv.extension("bokeh")

# deepmod imports, folder is local
sys.path.insert(0, './DeePyMoD/src/')
# DeepMoD functions
from deepymod.model.deepmod_new import DeepMoD
from deepymod.model.func_approx import NN
from deepymod.data.base_new import small_pipeline
from deepymod.model.library import Library1D
from deepymod import Library
from deepymod.model.custom_library import CustomTemperatureLibrary
from deepymod.model.constraint import LeastSquares
from deepymod.model.sparse_estimators import Threshold, PDEFIND
from deepymod.training.training_new import train
from deepymod.training.sparsity_scheduler import TrainTestPeriodic
from deepymod.model.func_approx import Siren
import torch.multiprocessing

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

# device will be cpu for now, changes are required to make it cuda compatible
device = "cpu"
print("Device is:", device)

# auxiliar functions
df = pd.read_pickle("/home/josesantos/DeepMod/pythonProjectII/dataframe.pk.zip")
df.head()
# getting only temperature at surface level and at point(28,-15), that's why it's 1D
case1_df = df[(df.Para == "TMP") & (df.Alt == "surface") & (df.Lat == 28) & (df.Lon == -15)]
ugrid_df = df[(df.Para == "U-GRID") & (df.Alt == "surface") & (df.Lat == 28) & (df.Lon == -15)]
vgrid_df = df[(df.Para == "V-GRID") & (df.Alt == "surface") & (df.Lat == 28) & (df.Lon == -15)]


# converting temperature to Celsiu and then normalizing
temporary_df = case1_df.copy()
temporary_df.Val -= 273.15

# scaled to [-1,1]
temporary_df["Val"] = 1 + (((temporary_df["Val"] - temporary_df["Val"].min()) * (-1 - 1)) / (
        temporary_df["Val"].max() - temporary_df["Val"].min()))

# converting date to days, starting at 0 at the 1st inicial day
inicial_time = temporary_df["Houridx"].iloc[0]
temporary_df.Houridx -= inicial_time

final_df = temporary_df.copy()
final_df["Houridx"] = final_df["Houridx"] / 24.0

temperature_array = final_df["Val"].to_numpy()
time_array = final_df["Houridx"].to_numpy()

from scipy.fft import rfft, rfftfreq

total_time_points = time_array.shape[0]
yf = rfft(temperature_array)
xf = rfftfreq(total_time_points)
sorted_yf = sorted(np.abs(yf), reverse=True)
frequencies = []
for i, coef in enumerate(np.abs(yf)):
    if coef in sorted_yf[:10]:
        frequencies.append((xf[i]))

# ---------------------------------------------------------------------------------------------------------------------#

# creating the dataset loader
t_train = torch.tensor(time_array, dtype=torch.float32,device=device).view(-1, 1)
# we have only one point in space so I will just fill x_train with ones, is not important
x_train = torch.ones_like(t_train).to(device)

Y = torch.tensor(temperature_array, dtype=torch.float32, device=device).view(-1, 1)
X = torch.cat((t_train, x_train), dim=1).to(device)

n_train = int(X.shape[0] * 0.8)
X_train, X_test = X[:n_train, :], X[n_train:, :]
Y_train, Y_test = Y[:n_train, :], Y[n_train:, :]

train_dataloader = small_pipeline(data=X_train, labels=Y_train,
                                  loader_params={"batch_size": 180, "drop_last": True, "num_workers": 4,"pin_memory":True}, device=device)

test_dataloader = small_pipeline(data=X_test, labels=Y_test,
                                 loader_params={"batch_size": 180, "drop_last": True, "num_workers": 4,"pin_memory":True}, device=device)

network = Siren(2, [30, 30, 30, 30], 1)
library = CustomTemperatureLibrary()
estimator = Threshold(0.1)
sparsity_scheduler = TrainTestPeriodic(periodicity=50, patience=50, delta=1e-5)
constraint = LeastSquares()
model = DeepMoD(network, library, estimator, constraint).to(device)
optimizer = torch.optim.AdamW(model.parameters(), betas=(0.99, 0.99), amsgrad=True, lr=1e-3)
train(
    model,
    train_dataloader,
    train_dataloader,
    optimizer,
    sparsity_scheduler,
    exp_ID="Test",
    write_epochs=25,
    max_epochs=5000,
    delta=1e-4,
    patience=50,
    device=device
)
