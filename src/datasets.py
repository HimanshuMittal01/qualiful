from pycocotools.coco import COCO
import os
import cv2
from tensorflow import keras
import numpy as np
from utils import cocoFunctions

class LemonDatasetCOCO:
    """
    """
    CLASSES = ['image_quality','illness','gangrene','mould','blemish','dark_style_remains','artifact','condition','pedicel']

    def __init__(
        self,
        images_dir,
        annot_file,
        img_size,
        classes=None,
        augmentation=None,
        preprocessing=None
    ):
        # Initiate COCO API
        self.coco = COCO(annot_file)
        self.img_size = img_size

        # Load images in dict format using COCO
        self.ids = self.coco.getImgIds()
        self.images_dir = images_dir
        self.images_objs = [img for img in self.coco.loadImgs(self.coco.getImgIds())]

        # Get class values
        self.classes = classes
        if self.classes is None:
            self.classes = [_cat['name'] for _cat in self.coco.loadCats(self.coco.getCatIds())]
        
        self.class_values = [self.coco.getCatIds(_cls)[0] for _cls in self.classes]

        # Get preprocessing and augmentation fn
        self.augmentation = augmentation
        self.preprocessing = preprocessing
    
    def __getitem__(self, i):
        # Read data
        image = cv2.imread(os.path.join(self.images_dir, os.path.basename(self.images_objs[i]['file_name'])))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask  = cocoFunctions.getNormalMask(self.coco, self.images_objs[i], self.classes)

        # Extract certain classes from mask
        masks = [(mask==v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype(np.float32)

        # Add background if mask is not binary
        if mask.shape[-1] != 1:
            background = 1 - mask.sum(axis=-1, keepdims=True)
            mask = np.concatenate((mask, background), axis=-1)
        
        # Apply augmentations
        if self.augmentation is not None:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        # Apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        image = cv2.resize(image, self.img_size)
        mask = cv2.resize(mask, self.img_size)

        return image, mask

    def __len__(self):
        return len(self.ids)

class LemonDataset:
    """
    """
    CLASSES = ['image_quality','illness','gangrene','mould','blemish','dark_style_remains','artifact','condition','pedicel']

    def __init__(
        self,
        images_dir,
        masks_dir,
        img_size,
        classes=None,
        augmentation=None,
        preprocessing=None
    ):
        self.ids = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps  = [os.path.join(masks_dir, image_id) for image_id in self.ids]
        self.img_size = img_size

        # Convert str names to class values on masks
        self.class_values = [self.CLASSES.index(_cls.lower()) for _cls in classes]

        self.augmentation = augmentation
        self.preprocessing = preprocessing
    
    def __getitem__(self, i):
        # Read data
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask  = cv2.imread(self.masks_fps[i], cv2.IMREAD_GRAYSCALE)

        # Extract certain classes from mask
        masks = [(mask==v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype(np.float32)

        # Add background if mask is not binary
        if mask.shape[-1] != 1:
            background = 1 - mask.sum(axis=-1, keepdims=True)
            mask = np.concatenate((mask, background), axis=-1)
        
        # Apply augmentations
        if self.augmentation is not None:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        # Apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        image = cv2.resize(image, self.img_size)
        mask = cv2.resize(mask, self.img_size)

        return image, mask
    
    def __len__(self):
        return len(self.ids)

class LemonDataLoader(keras.utils.Sequence):
    """Load data from dataset and form batches
    """
    def __init__(self, dataset, batch_size=2, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(dataset))

        self.on_epoch_end()
    
    def __getitem__(self, i):
        # Collect batch data
        start = i*self.batch_size
        stop = (i+1)*self.batch_size
        data = []
        for j in range(start, stop):
            data.append(self.dataset[j])
        
        # Transpose list of lists
        batch = [np.stack(samples, axis=0) for samples in zip(*data)]
        return tuple(batch)
    
    def __len__(self):
        """Denotes the number of batches per epoch"""
        return len(self.indexes) // self.batch_size
    
    def on_epoch_end(self):
        """Callback function to shuffle indexes each epoch"""
        if self.shuffle:
            self.indexes = np.random.permutation(self.indexes)
