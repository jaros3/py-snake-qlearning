import tensorflow as tf
from tkinter import Canvas, Frame, Label
from typing import List, NamedTuple, Optional
from PIL import Image, ImageTk
import numpy as np

from brain import Brain
from color import tkinter_rgb


class WidthHeight (NamedTuple):
    width: int
    height: int


class Display:
    SPACING = 1
    def __init__ (self, displays: 'Displays', frame: Frame, output: tf.Tensor):
        self.displays = displays
        self.output = output

        batch, self.channels, self.output_height, self.output_width = self.output.shape
        self.scale = 40 // self.output_width
        self.cell_width = self.output_width * self.scale + self.SPACING
        self.cell_height = self.output_height * self.scale + self.SPACING
        self.per_row = 41 * 8 // self.cell_width
        self.width = self.cell_width * self.per_row - 1
        self.height = self.cell_height * ((self.channels + self.per_row - 1) // self.per_row)
        self.size = WidthHeight (self.width, self.height)

        self.canvas = Canvas (frame, width = self.width, height = self.height)
        self.canvas.bind ('<Button-1>', self.click)
        self.canvas.pack ()

        self.tracking: Optional[int] = None
        # self.label = Label (frame)
        # image = Image.new ('L', self.size)
        # self.photo_image = ImageTk.PhotoImage (image)
        # self.label.config (image = self.photo_image)
        # self.label.pack ()

    def update (self, activation: np.ndarray) -> None:
        # image = Image.fromarray (self.generate_image (activation), mode = 'L')
        # self.photo_image = ImageTk.PhotoImage (image)
        # self.label.config (image = self.photo_image)
        self.update_canvas (activation)

    def update_canvas (self, activation: np.ndarray) -> None:
        self.canvas.delete ('all')

        row, column = 0, 0
        for channel in range (self.channels):
            base_y = row * self.cell_height
            base_x = column * self.cell_width
            for layer_y in range (self.output_height):
                for layer_x in range (self.output_width):
                    x = base_x + layer_x * self.scale
                    y = base_y + layer_y * self.scale
                    grey = activation[0, channel, layer_y, layer_x] / 2 + 0.5
                    color = tkinter_rgb (grey, grey, grey)
                    self.canvas.create_rectangle (x, y, x + self.scale, y + self.scale, fill = color, outline = '')
            column += 1
            if column >= self.per_row:
                column = 0
                row += 1

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

    def click (self, e) -> None:
        col = e.x // self.cell_width
        row = e.y // self.cell_height
        i = row * self.per_row + col
        self.tracking = i if i < self.channels else None
        self.displays.tracking = self.displays.items.index (self)


class Displays:
    UPDATE_PERIOD = 10

    def __init__ (self, frame: Frame, brain: Brain):
        self.frame = frame
        self.brain = brain
        self.items = [Display (self, frame, layer) for layer in brain.observed_layers]
        self.enabled = False
        self.tracking: Optional[int] = None

    def update_images (self, sight: np.ndarray) -> None:
        if not self.enabled or self.brain.game.current_step % self.UPDATE_PERIOD != 0:
            return
        activations: List[np.ndarray] = self.brain.display_activations (sight)
        display: Display
        for i, display in enumerate (self.items):
            display.update (activations[i])

    def key_press (self, e) -> None:
        if e.char == 'd':
            self.enabled = not self.enabled
        elif e.char == 't':
            if self.tracking is not None:
                self.items[self.tracking].tracking = None
            self.tracking = None
