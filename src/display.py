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
    SPACING = 1
    def __init__ (self, frame: Frame, output: tf.Tensor):
        self.output = output

        batch, self.channels, self.output_height, self.output_width = self.output.shape
        self.scale = 40 // self.output_width
        self.cell_width = self.output_width * self.scale + self.SPACING
        self.cell_height = self.output_height * self.scale + self.SPACING
        self.per_row = 164 // self.cell_width
        self.width = self.cell_width * self.per_row - 1
        self.height = self.cell_height * ((self.channels + self.per_row - 1) // self.per_row)
        self.size = WidthHeight (self.width, self.height)

        self.label = Label (frame)
        image = Image.new ('L', self.size)
        self.photo_image = ImageTk.PhotoImage (image)
        self.label.config (image = self.photo_image)
        self.label.pack ()

    def update (self, activation: np.ndarray) -> None:
        image = Image.fromarray (self.generate_image (activation), mode = 'L')
        self.photo_image = ImageTk.PhotoImage (image)
        self.label.config (image = self.photo_image)

    def generate_image (self, activation: np.ndarray) -> np.ndarray:
        result = np.zeros ((self.height, self.width))  # -1..1

        row, column = 0, 0
        for channel in range (self.channels):
            base_y = row * self.cell_height
            base_x = column * self.cell_width
            for layer_y in range (self.output_height):
                for layer_x in range (self.output_width):
                    for offset_y in range (self.scale):
                        for offset_x in range (self.scale):
                            x = base_x + layer_x * self.scale + offset_x
                            y = base_y + layer_y * self.scale + offset_y
                            result[y, x] = activation[0, channel, layer_y, layer_x]

            column += 1
            if column >= self.per_row:
                column = 0
                row += 1
        as_bytes = np.uint8 ((result / 2 + 0.5).clip (0, 1) * 255)
        return as_bytes


class Displays:
    def __init__ (self, frame: Frame, brain: Brain):
        self.frame = frame
        self.brain = brain
        self.items = [Display (frame, layer) for layer in brain.observed_layers]

    def update_images (self, sight: np.ndarray) -> None:
        activations: List[np.ndarray] = self.brain.display_activations (sight)
        display: Display
        for i, display in enumerate (self.items):
            display.update (activations[i])
