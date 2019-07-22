from tkinter import *
from game import Game
from dir import Dir


game = Game ()
Dir.static_init (game)


with game.brain.session.as_default ():
    window = Tk ()

    canvas = Canvas (window, bg = 'black',
                     width = game.SIGHT_DIAMETER * game.SCALE,
                     height = game.SIGHT_DIAMETER * game.SCALE)
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
        window.after (1, game_body)

    window.after (1, game_body)

    window.mainloop ()
