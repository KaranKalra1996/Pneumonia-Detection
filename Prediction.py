import matplotlib.pyplot as plt
import numpy as np
import cv2
import tensorflow as tf
import keras
from PIL import Image
from keras import models
from keras.models import Model
from keras.preprocessing import image
import os

class P_Model():
    model = keras.models.load_model(r"C:\Users\KARAN KALRA\Medical Imaging Project\Pneumonia-Detection-from-Chest-X-Ray-Images-with-Deep-Learning-master\code\model.hdf5")
    model._make_predict_function()
    
    def __init__(self):
        self.train_generator_label={'NORMAL': 0, 'PNEUMONIA': 1}
        self.img_path = r"C:\Users\KARAN KALRA\Medical Imaging Project\API\images\new_image.jpeg"
        
    
    def load_image(self,show=False):

        img = image.load_img(r"C:\Users\KARAN KALRA\Medical Imaging Project\API\images\new_image.jpeg", target_size=(150, 150))
        img_tensor = image.img_to_array(img,data_format="channels_first")                    # (height, width, channels)
        img_tensor = np.expand_dims(img_tensor, axis=0)         # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
        img_tensor /= 120.                                      # imshow expects values in the range [0, 1]

        if show:
            plt.imshow(img_tensor[0][0])
            plt.axis('off')
            plt.show()

        return img_tensor

    
    def check(self,data):
        
        data_dict = data.to_dict(flat=False)
        result = []
        for single_img in list(data_dict.keys()):
            npimg = np.fromfile(data[single_img], np.uint8)
            # convert numpy array to image
            img = cv2.imdecode(npimg, cv2.IMREAD_GRAYSCALE)
            cv2.imwrite(r"C:\Users\KARAN KALRA\Medical Imaging Project\API\images\new_image.jpeg",img)
            img1 = self.load_image(show=False)

            try:
                pred = self.model.predict_classes(img1)
                for single_key in list(self.train_generator_label.keys()):
                    if int(pred[0]) == self.train_generator_label[single_key]:
                        result_label = single_key


                result.append({"class":result_label,
                               "probability":round(max(self.model.predict_proba(img1)[0])*100,2),
                               "file_key_name":single_img,
                               "file_name":data[single_img].filename})#request.files[single_img].filename})
                os.remove(r"C:\Users\KARAN KALRA\Medical Imaging Project\API\images\new_image.jpeg")
            except Exception as e:
                result.append({"error":str(e)+"error"})
        res = {"result":"yes",
                "data":result}
        #print(res)
        return res



