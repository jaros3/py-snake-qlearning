from tkinter import *

from const import *
from dir import Dir
from game import Game


game = Game ()
Dir.static_init ()


with game.brain.session.as_default ():
    window = Tk ()

    canvas = Canvas (window, bg = 'black',
                     width = SIGHT_DIAMETER * SCALE,
                     height = SIGHT_DIAMETER * SCALE)
    # canvas = Canvas (window, bg = 'black', width = game.width * game.SCALE, height = game.height * game.SCALE)
    canvas.pack (fill = X)
    game.draw (canvas)

    label_text = StringVar (value = 'Attempt:\nScore:')
    game.set_text (label_text)
    label = Label (window, bg = 'grey', fg = 'black', textvariable = label_text)
    label.pack (fill = X)


    def game_body ():
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
