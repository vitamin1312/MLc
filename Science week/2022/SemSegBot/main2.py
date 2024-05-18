import cv2
import numpy as np
import tensorflow as tf
from scipy.io import loadmat

class begin():
    def __init__(self, user_id):
        self.user_id = str(user_id)

    def decode_segmentation_masks(self,mask, colormap, n_classes):
        r = np.zeros_like(mask).astype(np.uint8)
        g = np.zeros_like(mask).astype(np.uint8)
        b = np.zeros_like(mask).astype(np.uint8)
        for l in range(n_classes):
            idx = mask == l
            r[idx] = colormap[l, 0]
            g[idx] = colormap[l, 1]
            b[idx] = colormap[l, 2]
        rgb = np.stack([r, g, b], axis=2)
        return rgb


    def infer(self,model, image_tensor):
        predictions = model.predict(np.expand_dims((image_tensor), axis=0))
        predictions = np.squeeze(predictions)
        predictions = np.argmax(predictions, axis=2)
        return predictions


    def get_overlay(self,image, colored_mask):
        image = tf.keras.preprocessing.image.array_to_img(image)
        image = np.array(image).astype(np.uint8)
        overlay = cv2.addWeighted(image, 0.35, colored_mask, 0.65, 0)
        return overlay


    colormap = loadmat(
        "human_colormap.mat"
    )["colormap"]
    colormap = colormap * 100
    colormap = colormap.astype(np.uint8)


    IMAGE_SIZE = 512

    #model = tf.keras.models.load_model('DeepLabV3Plus_352_augment', compile=False)

    def processing(self,model,colormap):
        model=model
        colormap=colormap
        IMAGE_SIZE = 512
        image = cv2.imread(f'photos/{str(self.user_id)}.jpg')
        size = tuple(reversed(image.shape[:-1]))


        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
        img = img / 127.5 - 1

        prediction = self.infer(model, img)
        mask = self.decode_segmentation_masks(prediction, colormap, 20)
        mask = cv2.resize(mask, size)
        mask_show = cv2.cvtColor(mask, cv2.COLOR_RGB2BGR)
        cv2.imwrite(f'photos/{self.user_id}_segmentation_mask.jpg', mask_show)

        overlay = self.get_overlay(image, mask_show)
        cv2.imwrite(f"photos/{self.user_id}_processed.jpg", overlay)
