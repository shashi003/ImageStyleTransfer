import tensorflow_hub as tfhub
import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np
import cv2

#predefined model

model = tfhub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2') 


# image preprocessing
def load_image(img_path):
    
    img= tf.io.read_file(img_path)
    img=tf.image.decode_image(img,channels=3)
    img= tf.image.convert_image_dtype(img, tf.float32)
    img = img[tf.newaxis,:]    
    
    return img

content_image= load_image('./imgs/main_imgs/pic.jpg')
style_image= load_image('./imgs/style_imgs/starrynight.jpg')

plt.imshow(np.squeeze(content_image))
plt.show()

