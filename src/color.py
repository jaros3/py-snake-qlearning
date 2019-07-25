

def tkinter_rgb (r: float, g: float, b: float) -> str:  # 0..1
    r = max (0, min (255, int (round (r * 255))))
    g = max (0, min (255, int (round (g * 255))))
    b = max (0, min (255, int (round (b * 255))))
    return f'#{r:02x}{g:02x}{b:02x}'
