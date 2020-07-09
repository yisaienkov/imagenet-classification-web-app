import json

import numpy as np
import torch
import cv2
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2


def load_mobilenet_v2():
    model = torch.hub.load('pytorch/vision:v0.6.0', 'mobilenet_v2', pretrained=True)
    model.eval()
    return model


def get_preprocess():
    return A.Compose(
        [
            A.Resize(256, 256, p=1.),
            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                max_pixel_value=255.0,
                p=1.0
            ),
            ToTensorV2(p=1.0),
        ],
        p=1.0
    )


def load_image(image_path, preprocess=None):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
    image = preprocess(image=image)['image']
    image = image.unsqueeze(0)
    return image


def predict_probabilities(image, model):
    with torch.no_grad():
        output = torch.nn.functional.softmax(model(image)[0], dim=0).numpy()
    return output


def get_classes_by_numbers(indexes, labels_dict):
    names = []
    for i in range(len(indexes)):
        names.append(labels_dict.get(str(indexes[i])))
    return names


def predict_class(image_path, model_config, count=3):
    image = load_image(image_path, model_config.transforms)
    image.to(model_config.device)

    probs = predict_probabilities(image, model_config.model)

    max_indexes = probs.argsort()[-count:][::-1]
    max_values = probs[max_indexes].tolist()

    names = get_classes_by_numbers(max_indexes, model_config.labels)

    return max_values, names


class ModelConfig:
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = load_mobilenet_v2()
    model.to(device)

    transforms = get_preprocess()

    with open('resources/imagenet_labels.json', 'r') as f:
        labels = json.load(f)

