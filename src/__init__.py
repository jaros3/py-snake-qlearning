from tkinter import *

from const import *
from dir import Dir
from display import Displays
from game import Game


game = Game ()
Dir.static_init ()


with game.brain.session.as_default ():
    window = Tk ()

    canvas = Canvas (window, bg = 'black', width = SIGHT_DIAMETER * SCALE, height = SIGHT_DIAMETER * SCALE)
    # canvas = Canvas (window, bg = 'black', width = WIDTH * SCALE, height = HEIGHT * SCALE)
    game.draw (canvas)
    canvas.grid (row = 0, column = 0)

    label_text = StringVar (value = 'Attempt:\nScore:')
    game.set_text (label_text)
    label = Label (window, bg = 'grey', fg = 'black', textvariable = label_text)
    label.grid (row = 1, column = 0)

    display_frame = Frame (window, bg = 'black')
    display_frame.grid (row = 0, column = 1, rowspan = 2)
    game.displays = Displays (display_frame, game.brain)


    def game_body () -> None:
        game.step_and_learn ()
        game.draw (canvas)
        game.set_text (label_text)
        window.after (10, game_body)

    window.after (10, game_body)

    def second_elapsed ():
        game.seconds += 1
        window.after (1000, second_elapsed)

    window.after (1000, second_elapsed)

    window.mainloop ()
