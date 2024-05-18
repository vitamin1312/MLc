import cv2
import numpy as np
import tensorflow as tf
from scipy.io import loadmat
from imutils.video import VideoStream
from imutils.video import FPS
from matplotlib import pyplot as plt

colormap = loadmat(
    "./instance-level_human_parsing/instance-level_human_parsing/human_colormap.mat"
)["colormap"]
colormap = colormap * 100
colormap = colormap.astype(np.uint8)


def infer(model, image_tensor):
    predictions = model.predict(np.expand_dims((image_tensor), axis=0))
    predictions = np.squeeze(predictions)
    predictions = np.argmax(predictions, axis=2)
    return predictions


def decode_segmentation_masks(mask, colormap, n_classes):
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


def get_overlay(image, colored_mask):
    image = tf.keras.preprocessing.image.array_to_img(image)
    image = np.array(image).astype(np.uint8)
    overlay = cv2.addWeighted(image, 0.35, colored_mask, 0.65, 0)
    return overlay

IMAGE_SIZE = 352
print('---------------loading model----------------')
model = tf.keras.models.load_model(
    'DeepLabV3Plus_352', compile=False)

print('---------------model is ready---------------')


vs = VideoStream(src=0).start()

while True:
#     ret, img = cam.read()
    image = vs.read()
    cv2.imshow('camera', image)

    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))/127.5 - 1

    prediction = infer(model, img)
    mask = decode_segmentation_masks(prediction, colormap, 20)
    mask = cv2.resize(mask, (640, 480))
    mask_show = cv2.cvtColor(mask, cv2.COLOR_RGB2BGR)
    cv2.imshow('segmentation mask', mask_show)

    overlay = get_overlay(image, mask_show)
    cv2.imshow('image + mask', overlay)

    if cv2.waitKey(10) == 27:
        break

cv2.destroyAllWindows()
