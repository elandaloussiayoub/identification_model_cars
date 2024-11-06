import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import os

# Définissez les chemins pour le dossier d'entraînement et de test
train_dir = "C:\\Users\\a883714\\.cache\\kagglehub\\datasets\\jutrera\\stanford-car-dataset-by-classes-folder\\versions\\2\\car_data\\car_data\\train"
test_dir = "C:\\Users\\a883714\\.cache\\kagglehub\\datasets\\jutrera\\stanford-car-dataset-by-classes-folder\\versions\\2\\car_data\\car_data\\test"

# Prétraitement des images
train_datagen = ImageDataGenerator(rescale=1.0/255)
test_datagen = ImageDataGenerator(rescale=1.0/255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# Chargement d'un modèle pré-entraîné et personnalisation pour la classification
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(train_generator.num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Geler les couches du modèle de base pour utiliser les poids pré-entraînés
for layer in base_model.layers:
    layer.trainable = False


print("Compiler le modèle")

# Compiler le modèle
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print("Entraîner le modèle")

# Entraîner le modèle
model.fit(train_generator, validation_data=test_generator, epochs=10)

print("sauvegarder le modèle entrainé")
# Sauvegarder le modèle entraîné
model.save("car_classification_models.h5")
