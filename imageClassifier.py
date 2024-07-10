# Import libraries
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from imblearn.over_sampling import RandomOverSampler  # Import for oversampling
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import ModelCheckpoint

# Define data paths (modify according to your directory structure)
train_sem_dir = "SEM_Training_Data"  # Path to training SEM images
validation_sem_dir = "SEM_Validation_Data"  # Path to validation SEM images
mixed_data_dir = "materialScienceImages"  # Path to your mixed dataset (SEM and non-SEM)
destination_dir = "Classified_SEM_Images"  # Path to save separated images
img_width, img_height = 224, 224  # Adjust image size as needed 

tensorboard_callback = TensorBoard(log_dir='./classifierLogs', histogram_freq=1)
checkpoint_callback = ModelCheckpoint(filepath='sem_classifier.h5', monitor='val_loss', save_best_only=True)

# Data augmentation for SEM images
train_datagen_sem = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)

# Randomly sample images from mixed dataset (negative examples)
train_datagen_mixed = ImageDataGenerator(rescale=1./255)

# Oversample SEM training data
oversample = RandomOverSampler(sampling_strategy='auto')  # Adjust strategy if needed

def flow_from_directory_with_oversampling(directory):
  # Load images from directory
  imgs, labels = train_datagen_sem.flow_from_directory(
      directory,
      target_size=(img_width, img_height),
      batch_size=32,
      class_mode='binary'
  )
  
  # Oversample SEM data
  if oversample:
    imgs, labels = oversample.fit_resample(imgs, labels)
  
  return imgs, labels

# Train data generators
train_generator_sem, train_labels_sem = flow_from_directory_with_oversampling(train_sem_dir)
train_generator_mixed, train_labels_mixed = train_datagen_mixed.flow_from_directory(
    mixed_data_dir,
    target_size=(img_width, img_height),
    batch_size=32,  # Adjust batch size as needed (consider mixed data size)
    class_mode='binary',
    subset='training'  # Sample from training subset of mixed data
)

# Combine SEM and randomly sampled mixed data for training
train_generator = zip(train_generator_sem, train_generator_mixed)
train_labels = np.concatenate((train_labels_sem, train_labels_mixed))

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
# You can experiment with different weights for SEM and non-SEM classes
model.compile(optimizer=Adam(lr=0.0001),  # Adjust learning rate as needed
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=3)

# Train the model
model.fit(train_generator,
          epochs=10,  # Adjust epochs as needed
          validation_data=validation_generator,
          callbacks=[early_stopping, tensorboard_callback, checkpoint_callback])
