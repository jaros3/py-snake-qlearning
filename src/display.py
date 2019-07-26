import tensorflow as tf
import tensorflow.keras.layers as kl
from tkinter import Canvas, Frame, Label
from typing import List, NamedTuple, Optional, Tuple
from PIL import Image, ImageTk
import numpy as np
from abc import ABC, abstractmethod

from brain import Brain
from const import *
from color import tkinter_rgb


class WidthHeight (NamedTuple):
    width: int
    height: int


def gaussian_kernel (size: int, std: float):
    """Makes 2D gaussian Kernel for convolution."""
    distribution = tf.distributions.Normal (0.0, std)
    values = distribution.prob (tf.range (start = -size, limit = size + 1, dtype = tf.float32))
    kernel = tf.einsum ('i,j->ij', values, values)
    kernel = kernel / tf.reduce_sum (kernel)
    kernel = tf.stack ([kernel for i in range (SIGHT_CHANNELS)], axis = 2)
    kernel = kernel[:, :, :, np.newaxis]
    return kernel


class Display (ABC):
    SPACING = 1
    TRACK_STEPS = 10

    def __init__ (self, displays: 'Displays', frame: Frame, layer_index: int, layer: kl.Layer) -> None:
        self.displays = displays
        self.layer_index = layer_index
        self.layer = layer

        batch, self.channels, self.output_height, self.output_width = self.shape
        self.scale = 40 // self.output_width
        self.cell_width = self.output_width * self.scale + self.SPACING
        self.cell_height = self.output_height * self.scale + self.SPACING
        self.per_row = 41 * 8 // self.cell_width
        self.width = self.cell_width * self.per_row - 1
        self.height = self.cell_height * ((self.channels + self.per_row - 1) // self.per_row)
        self.size = WidthHeight (self.width, self.height)

        self._value: np.ndarray = np.zeros ((1, ))

        self.canvas = Canvas (frame, width = self.width, height = self.height)
        self.canvas.bind ('<Button-1>', self.click)
        self.canvas.pack ()

        self.target, self.optimizer_step, self.reset_optimizer, self.blur_image = self.prepare_tracking ()

        self.tracking: Optional[int] = None

        # self.label = Label (frame)
        # image = Image.new ('L', self.size)
        # self.photo_image = ImageTk.PhotoImage (image)
        # self.label.config (image = self.photo_image)
        # self.label.pack ()

    @property
    def value (self) -> np.ndarray:
        return self._value
    @value.setter
    def value (self, activations: np.ndarray) -> None:
        self._set_value (activations)
    @abstractmethod
    def _set_value (self, activations: np.ndarray) -> None:
        pass
    @abstractmethod
    def activation (self, channel: int, y: int, x: int) -> float:
        pass

    @property
    @abstractmethod
    def shape (self) -> Tuple[int, int, int, int]:
        pass

    def prepare_tracking (self) -> Tuple[Optional[tf.Tensor], Optional[tf.Operation], Optional[tf.Operation], Optional[tf.Operation]]:
        return None, None, None, None

    def update (self) -> None:
        self.update_canvas ()

        # image = Image.fromarray (self.generate_image (), mode = 'L')
        # self.photo_image = ImageTk.PhotoImage (image)
        # self.label.config (image = self.photo_image)

    def update_canvas (self) -> None:
        self.canvas.delete ('all')

        row, column = 0, 0
        for channel in range (self.channels):
            base_y = row * self.cell_height
            base_x = column * self.cell_width
            for layer_y in range (self.output_height):
                for layer_x in range (self.output_width):
                    x = base_x + layer_x * self.scale
                    y = base_y + layer_y * self.scale
                    grey = self.activation (channel, layer_y, layer_x) / 2 + 0.5
                    color = tkinter_rgb (grey, grey, grey)
                    self.canvas.create_rectangle (x, y, x + self.scale, y + self.scale, fill = color, outline = '')
            column += 1
            if column >= self.per_row:
                column = 0
                row += 1

    def generate_image (self) -> np.ndarray:
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
                            result[y, x] = self.activation (channel, layer_y, layer_x)

            column += 1
            if column >= self.per_row:
                column = 0
                row += 1
        as_bytes = np.uint8 ((result / 2 + 0.5).clip (0, 1) * 255)
        return as_bytes

    def click (self, e) -> None:
        pass

    def draw_tracking (self, canvas: Canvas, sight: np.ndarray) -> None:
        pass


class OutputDisplay (Display):
    def __init__ (self, displays: 'Displays', frame: Frame, layer_index: int) -> None:
        layer = displays.brain.observed_layers[layer_index]
        self.output = layer.output
        super ().__init__ (displays, frame, layer_index, layer)

    def _set_value (self, activations: np.ndarray) -> None:
        self._value = activations[self.layer_index]
    def activation (self, channel: int, y: int, x: int) -> float:
        return self._value[0, channel, y, x]

    @property
    def shape (self) -> Tuple[int, int, int, int]:
        return self.output.shape

    def prepare_tracking (self) -> Tuple[tf.Tensor, tf.Operation, tf.Operation, tf.Operation]:
        self.target = tf.placeholder (tf.float32, shape = self.shape)
        brain = self.displays.brain
        layer_index = next ((i
                             for i, layer in enumerate (brain.estimate_actions.layers)
                             if layer is self.layer))
        loss = tf.losses.mean_squared_error (brain.visual_brain.layers[layer_index].output, self.target)
        optimizer = tf.train.AdamOptimizer (0.1)
        self.optimizer_step = optimizer.minimize (loss, var_list = [brain.sight_var])
        self.reset_optimizer = tf.variables_initializer (optimizer.variables ())

        self.blur_image = tf.assign (
            brain.sight_var,
            tf.nn.depthwise_conv2d (
                brain.sight_var, self.displays.gaussian_kernel, (1, 1, 1, 1), padding = 'SAME', data_format = 'NCHW'))
        return self.target, self.optimizer_step, self.reset_optimizer, self.blur_image

    def click (self, e) -> None:
        col = e.x // self.cell_width
        row = e.y // self.cell_height
        i = row * self.per_row + col
        if i < self.channels:
            self.tracking = i
            self.displays.tracking = self.displays.items.index (self)
        else:
            self.tracking = None
            self.displays.tracking = None

    def draw_tracking (self, canvas: Canvas, sight: np.ndarray) -> None:
        brain = self.displays.brain
        session = brain.session
        target = session.run (self.output, feed_dict = { brain.sight_input: sight })  # (1, channels, height, width)
        target[0, 0:self.tracking, :, :] = 0
        target[0, self.tracking + 1:, :, :] = 0
        session.run (brain.copy_weights + [self.reset_optimizer])
        session.run (brain.set_sight_var, feed_dict = { brain.sight_value: sight })

        for i in range (self.TRACK_STEPS):
            session.run (self.optimizer_step, feed_dict = { self.target: target })
            session.run (self.blur_image)
        image = session.run (brain.sight_var)

        for y in range (SIGHT_DIAMETER):
            for x in range (SIGHT_DIAMETER):
                apple = image[0, CHANNEL_APPLE, y, x]
                obstacle = image[0, CHANNEL_OBSTACLE, y, x]
                color = tkinter_rgb (apple, obstacle, 0)
                canvas.create_rectangle (x * SCALE, y * SCALE, (x + 1) * SCALE, (y + 1) * SCALE, fill = color)


class WeightDisplay (Display):
    def __init__ (self, displays: 'Displays', frame: Frame, layer_index: int) -> None:
        layer = [layer
                 for layer in displays.brain.estimate_actions.layers
                 if isinstance (layer, kl.Conv2D)][layer_index]
        self.weight = layer.variables[0]
        super ().__init__ (displays, frame, layer_index, layer)

    def _set_value (self, activations: np.ndarray) -> None:
        self._value = self.displays.brain.session.run (self.layer.variables[0])
    def activation (self, channel: int, y: int, x: int) -> float:
        channel_in = channel % SIGHT_CHANNELS
        channel_out = channel // SIGHT_CHANNELS
        return self._value[y, x, channel_in, channel_out]

    @property
    def shape (self) -> Tuple[int, int, int, int]:
        height, width, channels_in, channels_out = self.weight.shape
        return 1, channels_in * channels_out, height, width


class Displays:
    UPDATE_PERIOD = 10
    GAUSSIAN_KERNEL_SIZE = 3
    GAUSSIAN_STDDEV = 0.5

    def __init__ (self, frame: Frame, brain: Brain):
        self.frame = frame
        self.brain = brain
        self.gaussian_kernel = gaussian_kernel (self.GAUSSIAN_KERNEL_SIZE, self.GAUSSIAN_STDDEV)
        self.items: List[Display] = [WeightDisplay (self, frame, 0)] + \
                                    [OutputDisplay (self, frame, i) for i, layer in enumerate (brain.observed_layers)]
        self.enabled = False
        self.tracking: Optional[int] = None

    def update_images (self, sight: np.ndarray) -> None:
        if not self.enabled or self.brain.game.current_step % self.UPDATE_PERIOD != 0:
            return
        activations: List[np.ndarray] = self.brain.display_activations (sight)
        display: Display
        for display in self.items:
            display.value = activations
            display.update ()

    def key_press (self, e) -> None:
        if e.char == 'd':
            self.enabled = not self.enabled
        elif e.char == 't':
            if self.tracking is not None:
                self.items[self.tracking].tracking = None
            self.tracking = None

    def draw_tracking (self, canvas: Canvas, sight: np.ndarray) -> None:
        self.items[self.tracking].draw_tracking (canvas, sight)
