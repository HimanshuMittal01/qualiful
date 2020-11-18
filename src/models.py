import os
import numpy as np
from tensorflow import keras
import src.segmentation_models as sm

sm.set_framework('tf.keras')

class ModelSM:
    def __init__(self, backbone, classes, weights=None, decoder_block_type='transpose'):
        self.backbone = backbone
        self.classes = classes
        self.weights=weights
        self.decoder_block_type = decoder_block_type

        # define network parameters
        self.num_classes = 1 if len(self.classes) == 1 else (len(self.classes) + 1)
        self.activation = 'sigmoid' if self.num_classes == 1 else 'softmax'

        # Status of the model
        if self.weights is None:
            self.is_compiled = False
        else:
            self.is_compiled = True
            self._create_model()
            
    def _create_model(self):
        self.model = sm.Unet(
            self.backbone,
            classes=self.num_classes,
            activation=self.activation,
            weights=self.weights,
            decoder_block_type=self.decoder_block_type
        )
    
    def get_preprocess_input_fn(self):
        return sm.get_preprocessing(self.backbone)
    
    def create_model(self, learning_rate, class_weights=None):
        """
        TODO: Add parameters to make custom model
        """
        
        self._create_model()

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
    
    def predict(self, x):
        # Predict
        return self.model.predict(x)
    
    def get_num_classes(self):
        return self.num_classes
