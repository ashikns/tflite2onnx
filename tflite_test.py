import numpy as np
import tensorflow as tf
import math

from PIL import Image
from matplotlib import pyplot

interpreter = tf.lite.Interpreter(model_path='D:/DeepLearning/mediapipe_meet/segm_full_v679.tflite')
interpreter.allocate_tensors()

image = Image.open('E:/Pictures/Camera Roll/segm_test_256_144.bmp')
image_data = image.tobytes()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_shape = input_details[0]['shape']
input_data = np.frombuffer(image_data, dtype='uint8').reshape(input_shape)
input_float = (input_data / 255.0).astype('float32')

interpreter.set_tensor(input_details[0]['index'], input_float)
interpreter.invoke()
output_data = interpreter.get_tensor(output_details[0]['index'])

output_pixels = np.zeros(256 * 144 * 3, dtype='uint8')

for i in range(144):
    for j in range(256):
        val = (output_data[0, i, j, 0], output_data[0, i, j, 1])
        shift = max(val)
        denom = math.exp(val[0] - shift) + math.exp(val[1] - shift)
        prob = math.exp(val[1] - shift) / denom
        pixel_r = int(prob * 255)

        output_pixels[(i * 256 + j) * 3] = pixel_r
        output_pixels[(i * 256 + j) * 3 + 1] = 0
        output_pixels[(i * 256 + j) * 3 + 2] = 0

#output_image = Image.frombytes('RGB', (256, 144), input_data.tobytes())
output_image = Image.frombytes('RGB', (256, 144), output_pixels.tobytes())
output_image.save('test_out.bmp')

print('done')