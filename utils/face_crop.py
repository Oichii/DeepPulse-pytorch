"""
Frames preparation
"""
import os
import glob
import cv2
import numpy as np
import json
import pandas as pd
from tqdm import tqdm


def face_crop(path='PURE', save_path='face_crop'):
    """
    Crops face from every image of the sequence and saves it.
    :param path: path to folder containing sequence of frames as .png images
    :param save_path: path to save cropped images
    """
    if not os.path.exists(path + '/' + save_path):
        os.makedirs(path + '/' + save_path)
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    sequence_dir = os.path.join(path)

    faces_list = []
    frames = glob.glob(sequence_dir + '/*.png')
    if len(frames) > 0:
        print(frames)
        image = np.array(cv2.imread(frames[0]))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.2, 2)

        (x, y, w, h) = faces[0]
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0))
        y = y - 30

        prev_y = y
        prew_h = h
        prew_x = x
        prew_w = w

        cropped = image[y:y + int(h / 2), x:x + w]
        faces_list.append({'x': int(x), 'y': int(y), 'w': int(w), 'h': int(h)})

        for fr in tqdm(range(len(frames))):
            image = np.array(cv2.imread(frames[fr]))
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 2)

            cropped = image[y:y + int(h * 3 / 2), x:x + w]

            # optional save of bounding rectangle s
            faces_list.append({'x': int(x), 'y': int(y), 'w': int(w), 'h': int(h)})

            cv2.imwrite(path+'/'+save_path+'/{0:05d}.png'.format(fr), cropped)

        ff = {'rectangles': faces_list}
        with open(path + '/' + save_path + '.json', 'a') as f:
            json.dump(ff, f)


if __name__ == '__main__':
    face_crop('../PURE/01-01', 'cropped')
