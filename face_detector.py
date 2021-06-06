from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
import PIL
from PIL import Image
import cv2
import numpy as np
import os
import sys
import time
import matplotlib.pyplot as plt


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class FaceDetector(object):
    def __init__(self, image_size=160, keep_all=False):
        """
            mtcnn: face detector
            extractor: face feature extractor
            Args:
                image_size: face image size
                keep_all: detect all faces or single face
        """
        self.keep_all = keep_all
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        # self.device = 'cpu'
        print('Using ', self.device)
        t = time.time()
        self.mtcnn = MTCNN(
            image_size=image_size, margin=0, min_face_size=20,
            thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
            keep_all=False, device=self.device
        )
        self.extractor = InceptionResnetV1(
            pretrained='vggface2').eval().to(self.device)
        print('Fininsh load model in {} sec'.format(time.time()-t))

    def detect(self, image, save_path=None):
        faces = self.mtcnn(image, save_path=save_path)
        return faces

    def get_boxes(self, image):
        boxes, _ = self.mtcnn.detect(image)
        return boxes

    def extract_feature(self, tensor):
        if not self.keep_all:
            tensor = tensor.unsqueeze(0)
        embeddings = self.extractor(tensor.to(self.device))
        embeddings = embeddings.detach().cpu().numpy()
        return embeddings if self.keep_all else np.squeeze(embeddings)

    def extract_bbox(self, image, save_path=None):
        try:
            faces = self.detect(image, save_path)
            embeddings = self.extract_feature(faces)
            return embeddings
        except Exception as err:
            # TODO: Logging here
            print(err)
            return None

    def extract_face(self, image, save_path=None):
        try:
            faces = self.detect(image, save_path)
            embeddings = self.extract_feature(faces)
            return embeddings
        except Exception as err:
            # TODO: Logging here
            print(err)
            return None

    def extract_face_from_folder(self, folder, save_prefix=None):
        # Use for single face per image. Modify save_path for multi face
        if not self.keep_all:
            all_embeddings = []
            for image_name in sorted(os.listdir(folder)):
                image_path = os.path.join(folder, image_name)
                print(image_path)
                image = pil_loader(image_path)
                save_path = os.path.join(
                    save_prefix, image_name) if save_prefix is not None else None

                all_embeddings.append(self.extract_face(image, save_path))
            return np.array(all_embeddings)