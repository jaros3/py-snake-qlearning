import tensorflow as tf
from tensorflow.layers import Layer
from tensorflow import Tensor
from tkinter import Frame, Label, PhotoImage
from typing import List, NamedTuple
from PIL import Image, ImageTk
import numpy as np
from matplotlib import pyplot as plt

from brain import Brain


class WidthHeight (NamedTuple):
    width: int
    height: int


class Display:
    SCALE = 2
    SPACING = 1
    PER_ROW = 4

    def __init__ (self, frame: Frame, brain: Brain):
        self.frame = frame
        self.brain = brain
        self.layers = brain.observed_layers
        self.labels = [self.make_label (layer) for layer in self.layers]

    def make_label (self, layer_output: tf.Tensor) -> Label:
        label = Label (self.frame)

        width_height = self.image_size (layer_output)
        image = Image.new ('L', width_height)
        photo_image = ImageTk.PhotoImage (image)
        label.config (image = photo_image)

        label.pack ()
        return label

    def update_images (self, sight: np.ndarray) -> None:
        activations: List[np.ndarray] = self.brain.display_activations (sight)
        for i, label in enumerate (self.labels):
            image = Image.fromarray (self.generate_image (activations[i]), mode = 'L')
            photo_image = ImageTk.PhotoImage (image)
            label.config (image = photo_image)
            label.image = photo_image

    def image_size (self, layer_output: tf.Tensor) -> WidthHeight:
        batch, channels, layer_height, layer_width = layer_output.shape
        single_width = layer_width * self.SCALE + self.SPACING
        single_height = layer_height * self.SCALE + self.SPACING
        width = single_width * self.PER_ROW - 1
        height = single_height // self.PER_ROW
        return WidthHeight (width, height)

    def generate_image (self, activation: np.ndarray) -> np.ndarray:
        batch, channels, layer_height, layer_width = activation.shape
        single_width = layer_width * self.SCALE + self.SPACING
        single_height = layer_height * self.SCALE + self.SPACING
        width = single_width * self.PER_ROW - self.SPACING
        height = single_height * (channels // self.PER_ROW)
        result = np.zeros ((height, width))  # -1..1

        row, column = 0, 0
        for channel in range (channels):
            base_y = row * single_height
            base_x = column * single_width
            for layer_y in range (layer_height):
                for layer_x in range (layer_width):
                    for offset_y in range (self.SCALE):
                        for offset_x in range (self.SCALE):
                            x = base_x + layer_x * self.SCALE + offset_x
                            y = base_y + layer_y * self.SCALE + offset_y
                            result[y, x] = activation[0, channel, layer_y, layer_x]

            column += 1
            if column >= self.PER_ROW:
                column = 0
                row += 1
        as_bytes = np.uint8 ((result / 2 + 0.5).clip (0, 1) * 255)
        return as_bytes
