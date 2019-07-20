from tkinter import *
from game import Game
from dir import Dir


game = Game ()
Dir.static_init (game)


window = Tk ()

canvas = Canvas (window, bg = 'black', width = 500, height = 500)
canvas.pack ()

game.draw (canvas)


def game_body ():
    game.step ()
    game.draw (canvas)
    window.after (50, game_body)

window.after (50, game_body)

window.mainloop ()
