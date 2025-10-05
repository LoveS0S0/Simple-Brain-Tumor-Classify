from tensorflow.keras.models import load_model # type: ignore 
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from tensorflow.keras.utils import load_img, img_to_array # type: ignore
import numpy as np

print("Starting evaluation...")

# --- Load saved model ---
model = load_model("brain_tumor_model.h5")  # or use .keras if you saved it that way

# --- Evaluate on test data ---
test_datagen = ImageDataGenerator(rescale=1./255)
test_data = test_datagen.flow_from_directory(
    "data/Testing",
    target_size=(150,150),
    batch_size=32,
    class_mode="categorical"
)

loss, accuracy = model.evaluate(test_data)
print(f"\nTest Loss: {loss}")
print(f"Test Accuracy: {accuracy}\n")

# --- Predict a single image ---
# Change this path to the image you want to test
img_path = "data/Testing/meningioma/Tetrite1.png" #input the file name of the image you want to be scanned, this CNN does not handle outside data well
img = load_img(img_path, target_size=(150,150))  # <-- use load_img from utils
img_array = img_to_array(img)                     # <-- use img_to_array from utils
img_array = np.expand_dims(img_array, axis=0) / 255.0  # scale like training

prediction = model.predict(img_array)
predicted_class_index = np.argmax(prediction)

# Map index to label (adjust to your classes)
class_indices = {'Glioma': 0, 'Meningioma': 1, 'Pituitary': 2, 'NoTumor': 3}
class_labels = {v: k for k, v in class_indices.items()}
predicted_class = class_labels[predicted_class_index]

print(f"Prediction for {img_path}: {predicted_class}")
print(f"Softmax probabilities: {prediction}")
