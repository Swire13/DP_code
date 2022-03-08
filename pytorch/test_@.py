from pytorch_forecasting.data.examples import get_stallion_data
import numpy as np
data = get_stallion_data()  # load data into pandas dataframe

data["time_idx"] = data["date"].dt.year * 12 + data["date"].dt.month
data["time_idx"] -= data["time_idx"].min()
 # adding features
 # categories must be string
data["month"] = data.date.dt.month.astype(str).astype("category")
data["log_volume"] = np.log(data.volume + 1e-8)
data["avg_volume_by_sku"] = (
    data
    .groupby(["time_idx", "sku"], observed=True)
    .volume.transform("mean")
)
data["avg_volume_by_agency"] = (
    data
    .groupby(["time_idx", "agency"], observed=True)
    .volume.transform("mean")
)
# we will encode special days into one variable and apply Reverse one-hot encoding
special_days = [
    "easter_day", "good_friday", "new_year", "christmas",
    "labor_day", "independence_day", "revolution_day_memorial",
    "regional_games", "fifa_u_17_world_cup", "football_gold_cup",
    "beer_capital", "music_fest"
]
data[special_days] = (
    data[special_days]
    .apply(lambda x: x.map({0: "-", 1: x.name}))
    .astype("category")
)
# sample data
data.sample(10, random_state=521)

from pytorch_forecasting.data import (
    TimeSeriesDataSet,
    GroupNormalizer
)
max_prediction_length = 6  # forecast of 6 months
max_encoder_length = 24  # using history of 24 months
training_cutoff = data["time_idx"].max() - max_prediction_length
training = TimeSeriesDataSet(
    data[lambda x: x.time_idx <= training_cutoff],
    time_idx="time_idx",
    target="volume",
    group_ids=["agency", "sku"],
    min_encoder_length=0,  # allowing predictions without history
    max_encoder_length=max_encoder_length,
    min_prediction_length=1,
    max_prediction_length=max_prediction_length,
    static_categoricals=["agency", "sku"],
    static_reals=[
        "avg_population_2017",
        "avg_yearly_household_income_2017"
    ],
    time_varying_known_categoricals=["special_days", "month"],
    # group of categorical variables can be treated as one variable
    variable_groups={"special_days": special_days},
    time_varying_known_reals=[
        "time_idx",
        "price_regular",
        "discount_in_percent"
    ],
    time_varying_unknown_categoricals=[],
    time_varying_unknown_reals=[
        "volume",
        "log_volume",
        "industry_volume",
        "soda_volume",
        "avg_max_temp",
        "avg_volume_by_agency",
        "avg_volume_by_sku",
    ],
    target_normalizer=GroupNormalizer(
        groups=["agency", "sku"]
    ),  # use softplus with beta=1.0 and normalize by group
    add_relative_time_idx=True,  # add as feature
    add_target_scales=True,  # add as feature
    add_encoder_length=True,  # add as feature
)
# creating validation set (predict=True) which means to predict the
# last max_prediction_length points in time for each series
validation = TimeSeriesDataSet.from_dataset(
    training, data, predict=True, stop_randomization=True
)
# create dataloaders for model
batch_size = 128
train_dataloader = training.to_dataloader(
    train=True, batch_size=batch_size, num_workers=0
)
val_dataloader = validation.to_dataloader(
    train=False, batch_size=batch_size * 10, num_workers=0
)

import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    EarlyStopping,
)
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_forecasting.metrics import QuantileLoss
from pytorch_forecasting.models import TemporalFusionTransformer
# training will stop, when validation loss does not improve
early_stop_callback = EarlyStopping(
    monitor="val_loss",
    min_delta=1e-4,
    patience=10,
    verbose=False,
    mode="min"
)
logger = TensorBoardLogger("lightning_logs")
# creating trainer
trainer = pl.Trainer(
    max_epochs=5,
    gpus=0,
    gradient_clip_val=0.1,
    limit_train_batches=30,  # running validation for every 30 batches
    logger=logger,
)
# initialise the model
tft = TemporalFusionTransformer.from_dataset(
    training,
    learning_rate=0.03,
    hidden_size=16,  # biggest influence network size
    attention_head_size=1,
    dropout=0.2,
    hidden_continuous_size=8,
    output_size=7,  # by default QuantileLoss has 7 quantiles
    loss=QuantileLoss(),
    log_interval=10,  # log example for every 10 batches
    reduce_on_plateau_patience=4,  # reduce learning automatically
)

## my edit to tensorflow problem
import tensorflow as tf
import tensorboard as tb
tf.io.gfile = tb.compat.tensorflow_stub.io.gfile

tft.size() # 29.6k parameters in model
# fit network
trainer.fit(
    tft,
    train_dataloader=train_dataloader,
    val_dataloaders=val_dataloader
)
