# Import libraries
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
from imblearn.over_sampling import RandomOverSampler
import numpy as np

# Define data paths (modify according to your directory structure)
train_sem_dir = "SEM_Training_Data/positive"  # Path to training SEM images
validation_sem_dir = "SEM_Validation_Data/positive"  # Path to validation SEM images
mixed_data_dir = "materialScienceImages"  # Path to your mixed dataset (SEM and non-SEM)
destination_dir = "Classified_SEM_Images"  # Path to save separated images
img_width, img_height = 224, 224  # Adjust image size as needed 

# Callbacks
tensorboard_callback = TensorBoard(log_dir='./classifierLogs', histogram_freq=1)
checkpoint_callback = ModelCheckpoint(filepath='sem_classifier.h5', monitor='val_loss', save_best_only=True)

# Data augmentation for SEM images
train_datagen_sem = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
train_datagen_mixed = ImageDataGenerator(rescale=1./255)

# Function to flow data from directory and apply oversampling
def flow_from_directory_with_oversampling(directory, datagen):
    # Load images from directory
    generator = datagen.flow_from_directory(
        directory,
        target_size=(img_width, img_height),
        batch_size=32,
        class_mode='binary',
        shuffle=True
    )
    
    # Extract images and labels
    images, labels = [], []
    for _ in range(len(generator)):
        batch_images, batch_labels = generator.next()
        images.append(batch_images)
        labels.append(batch_labels)
    
    images = np.vstack(images)
    labels = np.hstack(labels)
    
    # Oversample SEM data
    oversample = RandomOverSampler(sampling_strategy='auto')
    images_resampled, labels_resampled = oversample.fit_resample(images.reshape((images.shape[0], -1)), labels)
    images_resampled = images_resampled.reshape((-1, img_width, img_height, 3))
    
    return images_resampled, labels_resampled

# Train data generators
train_images_sem, train_labels_sem = flow_from_directory_with_oversampling(train_sem_dir, train_datagen_sem)
train_images_mixed, train_labels_mixed = flow_from_directory_with_oversampling(mixed_data_dir, train_datagen_mixed)

# Combine SEM and mixed data for training
train_images = np.vstack((train_images_sem, train_images_mixed))
train_labels = np.hstack((train_labels_sem, train_labels_mixed))

# Validation data generator for SEM images
validation_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = validation_datagen.flow_from_directory(
    validation_sem_dir,
    target_size=(img_width, img_height),
    batch_size=32,
    class_mode='binary'
)

# Load pre-trained model
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))

# Freeze pre-trained model layers
for layer in base_model.layers:
    layer.trainable = False

# Add new classification layers
x = base_model.output
x = Flatten()(x)
prediction = Dense(1, activation='sigmoid')(x)  # Sigmoid for binary classification

# Build the final model
model = Model(inputs=base_model.input, outputs=prediction)

# Compile the model
model.compile(optimizer=Adam(lr=0.0001),  # Adjust learning rate as needed
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=3)

# Train the model
model.fit(train_images, train_labels,
          epochs=10,  # Adjust epochs as needed
          validation_data=validation_generator,
          callbacks=[early_stopping, tensorboard_callback, checkpoint_callback])
