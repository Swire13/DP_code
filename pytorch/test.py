#source: https://towardsdatascience.com/introducing-pytorch-forecasting-64de99b9ef46
#source: https://github.com/jdb78/pytorch-forecasting

# imports for training
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
# import dataset, network to train and metric to optimize
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer, QuantileLoss, NaNLabelEncoder

### DATA PROCESSING ###
# load data: this is pandas dataframe with at least a column for
# * the target (what you want to predict)
# * the timeseries ID (which should be a unique string to identify each timeseries)
# * the time of the observation (which should be a monotonically increasing integer)
data = pd.read_json('/home/matej/PycharmProjects/DP_code/data/marsWeather_till_17_1_2022_test.json')

data['terrestrial_date']=pd.to_datetime(data['terrestrial_date'])
# data.set_index(data.id,inplace=True)

# all unique season values
# mask for every season
# add new column 0...N
# append all season together


data['sol_normalized'] = (data['sol']-501)%30
data['month_normalized'] = ((data['sol']-501)/30).astype('int32')

data["time_idx"] = data["terrestrial_date"].dt.year * 12 + data["terrestrial_date"].dt.month
data["time_idx"] -= data["time_idx"].min()

# data = data[data['max_temp'].notna()]


### TRAINING DATA PREPARATION ###

# define the dataset, i.e. add metadata to pandas dataframe for the model to understand it
# max_encoder_length = 36 # predpoklad: 3*686
# max_prediction_length = 7 # predpoklad: 686

max_encoder_length = 30 # predpoklad: 3*686
max_prediction_length = 7 # predpoklad: 686

# training_cutoff = "YYYY-MM-DD"  # day for cutoff
# training_cutoff = "2014-03-01"
# training_cutoff = (data["time_idx"].max()).astype('int32') - max_prediction_length
training_cutoff=1

training = TimeSeriesDataSet(
    data[lambda x: x.month_normalized <= training_cutoff],
    time_idx = "sol_normalized",  # column name of time of observation
    target = "max_temp",  # column name of target to predict
    group_ids = [ "month_normalized" ],

    # max_encoder_length=max_encoder_length,  # how much history to use
    # max_prediction_length=max_prediction_length,  # how far to predict into future

    min_encoder_length         = 5,
    max_encoder_length         = 5,
    min_prediction_length      = 2,
    max_prediction_length      = 2,

    # allow_missing_timesteps=True,
    # covariates static for a timeseries ID
    # static_categoricals=[ ... ],
    # static_reals=[ ... ],
    # covariates known and unknown in the future to inform prediction
    # time_varying_known_categoricals=[ ... ],
    # time_varying_known_reals=[ ... ],
    # time_varying_unknown_categoricals=[ ... ],
    time_varying_unknown_reals=[ "max_temp" ]
)

training.get_parameters()

# create validation dataset using the same normalization techniques as for the training dataset

# training.categorical_encoders={'max_temp': NaNLabelEncoder(add_nan=True)}

validation = TimeSeriesDataSet.from_dataset(
    training,
    data,
    min_prediction_idx=training.index.time.max() + 1,
    stop_randomization=True
)
# min_prediction_idx=training.index.time.max() + 1, # problem s indexom!

# convert datasets to dataloaders for training
batch_size = 5
train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=2)
val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size, num_workers=2)

# create PyTorch Lighning Trainer with early stopping
early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=1, verbose=False, mode="min")
lr_logger = LearningRateMonitor()
trainer = pl.Trainer(
    max_epochs=100,
    gpus=0,  # run on CPU, if on multiple GPUs, use accelerator="ddp"
    gradient_clip_val=0.1,
    limit_train_batches=30,  # 30 batches per epoch
    callbacks=[lr_logger, early_stop_callback],
    logger=TensorBoardLogger("lightning_logs")
)

# define network to train - the architecture is mostly inferred from the dataset, so that only a few hyperparameters have to be set by the user
tft = TemporalFusionTransformer.from_dataset(
    # dataset
    training,
    # architecture hyperparameters
    hidden_size=32,
    attention_head_size=1,
    dropout=0.1,
    hidden_continuous_size=16,
    # loss metric to optimize
    loss=QuantileLoss(),
    # logging frequency
    log_interval=2,
    # optimizer parameters
    learning_rate=0.03,
    reduce_on_plateau_patience=4
)
print(f"Number of parameters in network: {tft.size()/1e3:.1f}k")


##### vyzera, ze funguje - odtialto je problem s indexovanim, pravdepodobne v rmaci trianing dat... musim najst korektne riesenie na zapis indexov...
#IndexError: list index out of range


# find the optimal learning rate
res = trainer.tuner.lr_find(
    tft, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader, early_stop_threshold=1000.0, max_lr=0.3,
)
# and plot the result - always visually confirm that the suggested learning rate makes sense
print(f"suggested learning rate: {res.suggestion()}")
fig = res.plot(show=True, suggest=True)
fig.show()

# fit the model on the data - redefine the model with the correct learning rate if necessary
trainer.fit(
    tft, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader,
)