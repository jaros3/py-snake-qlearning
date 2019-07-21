from tkinter import *
from game import Game
from dir import Dir


game = Game ()
Dir.static_init (game)


window = Tk ()

canvas = Canvas (window, bg = 'black', width = 500, height = 500)
canvas.pack (fill = X)

label_text = StringVar (value = 'Attempt:\nScore:')
label = Label (window, bg = 'grey', fg = 'black', textvariable = label_text)
label.pack (fill = X)

game.draw (canvas, label_text)


def game_body ():
    game.step ()
    game.draw (canvas, label_text)
    window.after (50, game_body)

window.after (50, game_body)

window.mainloop ()
