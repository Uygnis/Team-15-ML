# Import Library
import glob
import os
from datetime import datetime, timedelta, time
import math
import numpy as np
import pandas as pd
import wandb
import optuna
import torch
from torch import nn
import plotly.express as px

from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import Callback

from torchmetrics import MeanAbsolutePercentageError

import matplotlib.pyplot as plt

from sklearn.preprocessing import MaxAbsScaler
from sklearn.metrics import r2_score, mean_absolute_percentage_error, mean_squared_error,mean_absolute_error

from darts import TimeSeries, concatenate
from darts.models import TFTModel, LinearRegressionModel, ARIMA
from darts.metrics import mape, rmse, r2_score, mae, smape
from darts.utils.statistics import check_seasonality, plot_acf
from darts.utils.timeseries_generation import datetime_attribute_timeseries
from darts.utils.likelihood_models import (QuantileRegression, GaussianLikelihood)
from darts.dataprocessing.transformers import Scaler, MissingValuesFiller
import warnings

warnings.filterwarnings("ignore")
import logging

# before starting, we define some constants

DAY_DURATION = 24   # 1 day, 1hr frequency
DAY_PREDICT = 3     # Number of days we want to predict ahead

np.random.seed(455)
num_samples = 200

figsize = (9, 6)
lowest_q, low_q, high_q, highest_q = 0.01, 0.1, 0.9, 0.99
label_q_outer = f"{int(lowest_q * 100)}-{int(highest_q * 100)}th percentiles"
label_q_inner = f"{int(low_q * 100)}-{int(high_q * 100)}th percentiles"


# log train loss in 
class LossLogger(Callback):
    def __init__(self):
        self.train_loss = []
        self.val_loss = []

    # will automatically be called at the end of each epoch
    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.train_loss.append(float(trainer.callback_metrics["train_loss"]))
        wandb.log({"train_loss": float(trainer.callback_metrics["train_loss"])})
        # will automatically be called at the end of each epoch
    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if not trainer.sanity_checking:
            self.val_loss.append(float(trainer.callback_metrics["val_loss"]))
            wandb.log({"loss": float(trainer.callback_metrics["val_loss"])})
    
early_stopper = EarlyStopping("val_loss", min_delta=0.001, patience=3, verbose=True)
loss_logger = LossLogger()
    # detect if a GPU is available
if torch.cuda.is_available():
    pl_trainer_kwargs = {
        "accelerator": "gpu",
        "devices": [0],
        "callbacks": [early_stopper, loss_logger],
    }
    num_workers = 4
else:
    pl_trainer_kwargs = {"callbacks": [early_stopper, loss_logger]}
    num_workers = 0




# Make data
# Create list to store all file names with half-hourly records
li =[]
for name in glob.glob('/root/Code/data/*.csv'):
    df = pd.read_csv(name, index_col=None, header=0, parse_dates=["DATE"])
    li.append(df)
# merge all records
ds = pd.concat(li, axis=0, ignore_index=True).drop(["PERIOD", "INFORMATION TYPE", "LCP ($/MWh)", "TCL (MW)", "TCL(MW)" , "SOLAR(MW)"], axis='columns')

len(ds. index) 
interval = 30

# Because we want hourly records, we manipulate data to get the average price/hour 
rows_to_drop = [] # list to store all records to be dropped
print("Manipulating data...")
for i in range(len(ds. index)):
        # convert string to datetime object
    d = ds.at[i, "DATE"]
    result = d + timedelta(minutes=interval*(i%48))
    ds.at[i, "DATE"] = result
    if i%2==0:
        ds.at[i, "USEP ($/MWh)"] = (ds.at[i, "USEP ($/MWh)"] + ds.at[i+1, "USEP ($/MWh)"])/2    # get the average price/hour 
        ds.at[i,"DEMAND (MW)"] = (ds.at[i, "DEMAND (MW)"] + ds.at[i+1, "DEMAND (MW)"])/2    # get the average demand
    else:
        rows_to_drop.append(int(i))

ds = ds.drop(rows_to_drop)   # drop all even records
ds = ds.reset_index(drop=True)
    
series = TimeSeries.from_dataframe(ds, time_col = 'DATE' , value_cols =['USEP ($/MWh)', 'DEMAND (MW)']) # convert to timeseries

# plot on plotly
fig = px.line(ds, x='DATE', y='USEP ($/MWh)', title='Electricity Prices')
fig.add_scatter(x=ds['DATE'], y=ds['DEMAND (MW)'], mode='lines')
fig.write_html("overall.html")

converted_series = []
for col in ["USEP ($/MWh)", "DEMAND (MW)"]:
    converted_series.append(
        series[col]
    )
converted_series = concatenate(converted_series, axis=1)

n_fc = DAY_PREDICT * DAY_DURATION   #predict n day ahead as test
print(f"length to predict: {n_fc}")
# define train/validation cutoff time
# converted_series = converted_series[pd.Timestamp("20100101") :]
training_cutoff_pricing = converted_series.time_index[-n_fc]
print(f'training cutoff: {training_cutoff_pricing}')


# use electricity pricing as target, create train and validation sets and transform data
series_pricing = converted_series["USEP ($/MWh)"]

transformer_pricing = Scaler()
train_pricing, val_pricing = series_pricing.split_before(training_cutoff_pricing)

train_pricing_transformed = transformer_pricing.fit_transform(train_pricing)
val_pricing_transformed = transformer_pricing.transform(val_pricing)
series_pricing_transformed = transformer_pricing.transform(series_pricing)

print(f"length to validate: {len(val_pricing_transformed)}") 

# use electricity demand as past covariates and transform data
covariates_demand = converted_series["DEMAND (MW)"]

cov_demand_train, cov_demand_val = covariates_demand.split_before(training_cutoff_pricing)
transformer_demand = Scaler()
transformer_demand.fit(cov_demand_train)
covariates_demand_transformed = transformer_demand.transform(covariates_demand)


# use the last 7 days as past input data
input_chunk_length_pricing = 7* DAY_DURATION

def eval_model_pricing(model, n, actual_series, val_series):
    pred_series = model.predict(n=n, num_samples=num_samples)

    # plot actual series
    plt.figure(figsize=figsize)
    actual_series[: pred_series.end_time()].plot(label="actual")

    # plot prediction with quantile ranges
    pred_series.plot(
        low_quantile=lowest_q, high_quantile=highest_q, label=label_q_outer
    )
    pred_series.plot(low_quantile=low_q, high_quantile=high_q, label=label_q_inner)

    plt.title("MAPE: {:.2f}%".format(mape(val_series, pred_series)))
    plt.legend()
    plt.savefig('nice_fig.png')
    
    frame = pd.DataFrame(pred_series.quantile_df())
    frame = frame.reset_index()
    fig = px.line(actual_series[: pred_series.end_time()].pd_dataframe(), y='USEP ($/MWh)', title='Electricity Prices')
    fig.add_scatter(x=frame['DATE'], y=frame['USEP ($/MWh)_0.5'], mode='lines')
    # fig.write_html("file.html")
    wandb.log({"chart": fig, "mape":mape(val_series, pred_series), "mae":mae(val_series, pred_series)})
    
print(f"train pricing end: {train_pricing.end_time()}")
print(f"val pricing start: {val_pricing.start_time()}")
print(f"val pricing end: {val_pricing.end_time()}")
print(f"train pricing freq: {train_pricing.freq}")
print(f"series cutoff: {train_pricing.end_time() - (2 * n_fc) * train_pricing.freq}")

with wandb.init(project="Electricity Prices", config = None):
    config = wandb.config
# use `add_encoders` as we don't have future covariates
    my_model_pricing = TFTModel(
        input_chunk_length=input_chunk_length_pricing,
        output_chunk_length=n_fc,
        hidden_size=config.hidden_size,
        num_attention_heads=config.att_heads,
        lstm_layers=config.lstm_layers,
        batch_size=16,
        n_epochs=3,
        dropout=config.dropout,
        add_encoders={"cyclic": {"future": ["day"],},
                      "datetime_attribute": {'future': ['hour']},},
        add_relative_index=False,
        optimizer_cls=torch.optim.SGD,
        optimizer_kwargs={"lr": config.learning_rate},
        random_state=config.rand,
        pl_trainer_kwargs=pl_trainer_kwargs,
    )

    # fit the model with past covariates
    model_val_set = series_pricing_transformed[-((2 * n_fc) + input_chunk_length_pricing ) : -n_fc] # target series must have at least input_chunk_length + output_chunk_length time steps
    my_model_pricing.fit(
        series =train_pricing_transformed, val_series=model_val_set, val_past_covariates=covariates_demand_transformed, past_covariates=covariates_demand_transformed, verbose=True
    )

    eval_model_pricing(
        model=my_model_pricing,
        n=len(val_pricing_transformed),
        actual_series=series_pricing_transformed[
            train_pricing.end_time() - (2 * n_fc) * train_pricing.freq :
        ],
        val_series=val_pricing_transformed,
    )
