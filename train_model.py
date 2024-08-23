import os
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from keras.callbacks import CSVLogger
from Scripts.read_config import config 
from Scripts.train_model import PlotLossAccuracy, model, DataGen, \
                                Label_Generator, MLflowModelCheckpoint, \
                                plot_model_schematic
import mlflow
import mlflow.tensorflow
import logging
from dotenv import load_dotenv

load_dotenv()  

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration and Hyperparameters
SEED = 42
IMAGE_WIDTH = 128
IMAGE_HEIGHT = 384
CHANNELS = 7
FILTERS = [32, 64, 128, 256, 512, 1024, 2048]
NUM_BLOCKS = len(FILTERS) - 1
DROPOUT_RATE = 0.4
LEARNING_RATE = 1e-5
OPTIMIZER_NAME = 'Adam'
BATCH_SIZE = 8
L1_RATE = 1e-6
EPOCHS = 132
MODEL_NAME = 'testing_code_model'  # Replace with your model name

# Paths
MODEL_SAVE_PATH = os.path.join(config.output_data_fp, MODEL_NAME, f"{MODEL_NAME}.keras")
TRAIN_PATH = os.path.join(config.output_data_fp, 'train')
TEST_PATH = os.path.join(config.output_data_fp, 'test')
CHECKPOINT_FILEPATH = os.path.join(config.output_data_fp, MODEL_NAME)

# Seeding
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    print(f"Set random seed to {seed}")

set_seed(SEED)

# Load IDs
def load_ids(path, count, prefix_length=3):
    ids = next(os.walk(path))[1]
    all_ids = [str(i).zfill(prefix_length) for i in range(count)]
    valid_ids = [i for i in all_ids if i in ids]
    print(f"Loaded {len(valid_ids)} IDs from {path}")
    return valid_ids

total_ids = load_ids(TRAIN_PATH, 3000)
test_ids = load_ids(TEST_PATH, 2000)

# Load training labels
train_labels = np.load(os.path.join(config.output_data_fp, 'train_label.npy'), allow_pickle=True)
train_ids, valid_ids = train_test_split(
    total_ids, test_size=0.35, shuffle=True, random_state=SEED, stratify=train_labels
)
print(f"Split data into {len(train_ids)} training and {len(valid_ids)} validation IDs")

# Create directories if not exists
os.makedirs(CHECKPOINT_FILEPATH, exist_ok=True)
print(f"Created checkpoint directory at {CHECKPOINT_FILEPATH}")

# Model Definition and Compilation
optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, name=OPTIMIZER_NAME)
iou_lfe = tf.keras.metrics.BinaryIoU(name='iou_l', target_class_ids=[1])
iou_no = tf.keras.metrics.BinaryIoU(name='iou_n', target_class_ids=[0])

model = model(do=DROPOUT_RATE, f=FILTERS, image_h=IMAGE_HEIGHT,
               image_w=IMAGE_WIDTH, channels=CHANNELS,
                 l1_rate=L1_RATE)
model.compile(
    optimizer=optimizer,
    loss='binary_crossentropy',
    metrics=['accuracy', iou_lfe, iou_no]
)
print("Model compiled with optimizer, loss, and metrics")

# Data Generators
train_gen = DataGen(train_ids, TRAIN_PATH, image_h=IMAGE_HEIGHT, image_w=IMAGE_WIDTH, batch_size=BATCH_SIZE)
valid_gen = DataGen(valid_ids, TRAIN_PATH, image_h=IMAGE_HEIGHT, image_w=IMAGE_WIDTH, batch_size=BATCH_SIZE)
test_gen = DataGen(test_ids, TEST_PATH, image_h=IMAGE_HEIGHT, image_w=IMAGE_WIDTH, batch_size=1)
test_label_gen = Label_Generator(test_ids, TEST_PATH, 1)
print("Data generators created for training, validation, and testing")

# Steps per epoch
train_steps = len(train_ids) // BATCH_SIZE
valid_steps = len(valid_ids) // BATCH_SIZE
print(f"Training steps per epoch: {train_steps}, Validation steps per epoch: {valid_steps}")

# Callbacks
plt_callback = PlotLossAccuracy()
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=MODEL_SAVE_PATH,
    save_weights_only=False,
    monitor='val_loss',
    mode='auto',
    save_best_only=True
)
csv_logger = CSVLogger(os.path.join(CHECKPOINT_FILEPATH, "model_history_log.csv"), append=True)
print("Callbacks initialized")

# MLflow Logging
#!mlflow ui
#mlflow.set_tracking_uri("http://localhost:5000")  # or your MLflow server URI
mlflow.set_experiment("your_experiment_name")  
print("MLflow tracking URI set and experiment configured")

with mlflow.start_run() as run:
    print("MLflow run started")

    # Log parameters
    mlflow.log_param("SEED", SEED)
    mlflow.log_param("IMAGE_WIDTH", IMAGE_WIDTH)
    mlflow.log_param("IMAGE_HEIGHT", IMAGE_HEIGHT)
    mlflow.log_param("CHANNELS", CHANNELS)
    mlflow.log_param("DROPOUT_RATE", DROPOUT_RATE)
    mlflow.log_param("LEARNING_RATE", LEARNING_RATE)
    mlflow.log_param("OPTIMIZER_NAME", OPTIMIZER_NAME)
    mlflow.log_param("BATCH_SIZE", BATCH_SIZE)
    mlflow.log_param("L1_RATE", L1_RATE)
    mlflow.log_param("EPOCHS", EPOCHS)
    mlflow.log_param("MODEL_NAME", MODEL_NAME)
    print("Parameters logged to MLflow")

    # Train the model
    history = model.fit(
        train_gen,
        validation_data=valid_gen,
        steps_per_epoch=train_steps,
        validation_steps=valid_steps,
        epochs=EPOCHS,
        verbose=1,
        callbacks=[model_checkpoint_callback, plt_callback, csv_logger, MLflowModelCheckpoint(CHECKPOINT_FILEPATH)]
    )
    print("Model training completed")

    # Log metrics
    for epoch, metrics in enumerate(history.history):
        mlflow.log_metrics(metrics, step=epoch)
        print(f"Metrics for epoch {epoch} logged to MLflow")

    # Save and log the final model
    model.load_weights(os.path.join(config.output_data_fp, MODEL_NAME))
    model_save_path = os.path.join(config.output_data_fp, MODEL_NAME)
    
    #tf.keras.models.save_model(model, model_save_path)
    mlflow.tensorflow.log_model(model, artifact_path="model")
    print("Final model saved and logged to MLflow")