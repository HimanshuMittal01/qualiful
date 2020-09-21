import os
import numpy as np
from tensorflow import keras
import segmentation_models as sm

sm.set_framework('tf.keras')

class ModelSM:
    def __init__(self, backbone, batch_size, classes):
        self.backbone = backbone
        self.batch_size = batch_size
        self.classes = classes

        # define network parameters
        self.num_classes = 1 if len(self.classes) == 1 else (len(self.classes) + 1)
        self.activation = 'sigmoid' if self.num_classes == 1 else 'softmax'

        # Status of the model
        self.is_compiled = False
    
    def get_preprocess_input_fn(self):
        return sm.get_preprocessing(self.backbone)
    
    def create_model(self, learning_rate, class_weights=None):
        """
        TODO: Add parameters to make custom model
        """
        
        self.model = sm.Unet(self.backbone, classes=self.num_classes, activation=self.activation)

        # define optomizer
        self.optim = keras.optimizers.Adam(learning_rate)

        # Define loss
        if class_weights is None:
            class_weights = np.ones(self.num_classes)
        
        dice_loss = sm.losses.DiceLoss(class_weights=np.array(class_weights)) 
        focal_loss = sm.losses.BinaryFocalLoss() if self.num_classes == 1 else sm.losses.CategoricalFocalLoss()
        total_loss = dice_loss + (1 * focal_loss)

        metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]

        # compile keras model with defined optimozer, loss and metrics
        self.model.compile(self.optim, total_loss, metrics)

        # Model is compiled
        self.is_compiled = True
    
    def fit_generator(self, train_generator, valid_generator, epochs, callbacks=[]):
        """Train the model
        """
        if not self.is_compiled:
            raise Exception("Create model before fitting")

        # Train model
        history = self.model.fit_generator(
            train_generator, 
            steps_per_epoch=len(train_generator), 
            epochs=epochs, 
            callbacks=callbacks, 
            validation_data=valid_generator, 
            validation_steps=len(valid_generator),
        )

        return history
    
    def get_num_classes(self):
        return self.num_classes
