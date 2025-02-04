import torch
import numpy as np

class LabelSmooth:
    def __call__(self, labels, num_classes, epsilon=0.1):
        smooth_labels = (1 - epsilon) * labels + (epsilon / num_classes)
        return smooth_labels
