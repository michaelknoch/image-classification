from tensorflow import keras
import numpy as np
from PIL import Image

model = keras.models.load_model("model.h5")

img = Image.open("./demo.jpg")
img = img.resize((224, 224))
img_array = np.array(img)
img_array = np.expand_dims(img_array, axis=0)

prediction = model.predict(img_array)
print(prediction)
class_index = np.argmax(prediction)

print(f"The predicted class is: {class_index}")