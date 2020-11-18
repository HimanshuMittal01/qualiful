from src.models import ModelSM
from src.datasets import LemonDatasetCOCO, LemonDataLoader
from src.augmentations import Augmentor
from utils.paths import Path

# Config parameters
BACKBONE = 'efficientnetb3'
CLASSES = ['image_quality','illness']
IMG_SIZE = (256,256)
WEIGHTS = 'best_model.h5' # Path to model weights

# Prepare augmentation
augmentor = Augmentor(img_size=IMG_SIZE)

# Call model
model = ModelSM(
    backbone=BACKBONE,
    classes=CLASSES,
    weights=WEIGHTS,
    decoder_block_type='transpose'
)

dataset = LemonDatasetCOCO(
    images_dir=Path.get_x_val_dir(),
    annot_file=Path.get_y_val_file(),
    img_size=IMG_SIZE,
    classes=CLASSES,
    augmentation=augmentor.get_validation_augmentation(),
    preprocessing=augmentor.get_preprocessing(model.get_preprocess_input_fn())
)

dataloader = LemonDataLoader(dataset, batch_size=1, shuffle=False)

# Predict on 24th image
x0, y0 = dataloader[23]
pred = model.predict(
    x=x0
)
