import os
import numpy as np
from tensorflow import keras

from src.models import ModelSM
from src.datasets import LemonDatasetCOCO, LemonDataLoader
from src.augmentations import Augmentor
from utils.paths import Path

# Config parameters
BACKBONE = 'efficientnetb3'
BATCH_SIZE = 1
CLASSES = ['image_quality','illness','gangrene','mould','blemish','dark_style_remains','artifact','condition','pedicel']
LR = 0.0001
EPOCHS = 2
IMG_SIZE = (1056,1056)

# Prepare augmentation
augmentor = Augmentor(img_size=IMG_SIZE)

# Call model
model = ModelSM(
    backbone=BACKBONE,
    batch_size=BATCH_SIZE,
    classes=CLASSES
)

# Create train and val dataloader
train_dataset = LemonDatasetCOCO(
    images_dir=Path.get_x_train_dir(),
    annot_file=Path.get_y_train_file(),
    classes=CLASSES,
    augmentation=augmentor.get_training_augmentation(),
    preprocessing=augmentor.get_preprocessing(model.get_preprocess_input_fn())
)
valid_dataset = LemonDatasetCOCO(
    images_dir=Path.get_x_val_dir(),
    annot_file=Path.get_y_val_file(),
    classes=CLASSES,
    augmentation=augmentor.get_validation_augmentation(),
    preprocessing=augmentor.get_preprocessing(model.get_preprocess_input_fn())
)

print("Training dataset length:", len(train_dataset))
print("Validation dataset length:", len(valid_dataset))

# Create dataloaders
train_dataloader = LemonDataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_dataloader = LemonDataLoader(valid_dataset, batch_size=1, shuffle=False)

# check shapes for errors
assert train_dataloader[0][0].shape == (BATCH_SIZE, *IMG_SIZE, 3)
assert train_dataloader[0][1].shape == (BATCH_SIZE, *IMG_SIZE, model.get_num_classes())


model.create_model(
    learning_rate=LR,
    class_weights=None
)

# Define callbacks for learning rate scheduling and best checkpoints saving
callbacks = [
    keras.callbacks.ModelCheckpoint('./best_model.h5', save_weights_only=True, save_best_only=True, mode='min'),
    keras.callbacks.ReduceLROnPlateau(),
]

# Train model
history = model.fit_generator(
    train_generator=train_dataloader,
    valid_generator=valid_dataloader,
    epochs=EPOCHS,
    callbacks=callbacks
)

# Plot training & validation iou_score values
plt.figure(figsize=(30, 5))
plt.subplot(121)
plt.plot(history.history['iou_score'])
plt.plot(history.history['val_iou_score'])
plt.title('Model iou_score')
plt.ylabel('iou_score')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')

# Plot training & validation loss values
plt.subplot(122)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()