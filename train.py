import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore (stop unecessary linting)
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout # type: ignore

# Preprocess images
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(
    "data/Training", 
    target_size=(150,150),
    batch_size=32,
    class_mode="categorical"
)

test_data = test_datagen.flow_from_directory(
    "data/Testing", 
    target_size=(150,150),
    batch_size=32,
    class_mode="categorical"
)

# Define CNN model
model = Sequential([
    Conv2D(32, (3,3), activation="relu", input_shape=(150,150,3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation="relu"),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(4, activation="softmax")  # 4 classes
])

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Train the model
history = model.fit(train_data, validation_data=test_data, epochs=10)

# Save the model
model.save("brain_tumor_model.h5")
