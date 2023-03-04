from tensorflow import keras
import numpy as np
from PIL import Image
from resize_image import resize_image 

model = keras.models.load_model("model.h5")

size = 224
image = Image.open("./cat.jpg")
image = resize_image(image, size, size)

img_array = np.array(img)
img_array = np.expand_dims(img_array, axis=0)

prediction = model.predict(img_array)
print(prediction)
class_index = np.argmax(prediction)

print(f"The predicted class is: {class_index}")