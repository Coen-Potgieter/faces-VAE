import pygame as py
import numpy as np
import sys


class Slider:
    def __init__(self, pos: tuple, lo: int, hi: int, init_val: int, line_dims: tuple, radius: int, slider_col: tuple, line_col: tuple, v_orientation=False):
        '''
        Creates a circular Slider with an attached value

        Parameters:
        - pos (tuple): Coordinates (x, y) for the slider placement
        - lo, hi, init_val (int): Specifies min, max and initial values
        - line_dims (tuple): (length, thickness)
        - radius (int): Radius for slider
        - slider_col, line_col (tuple): 3 element tuple specifying RGB values 0-255
        - v_orientation (bool): Sliders are set to vertical orientation if True

        Raises:
        - ValueError: if init_val does not fall in range of lo and hi
        '''

        if not lo <= init_val <= hi:
            raise ValueError(
                f"init_val={init_val} is outside of the bounds {lo}-{hi}")

        self.hi, self.lo, self.val = hi, lo, init_val
        self.vertical = v_orientation
        self.xpos, self.ypos = pos
        self.line_dims = line_dims
        l_length, l_thick = self.line_dims

        slid_factor = (init_val-lo) / (hi-lo)
        center_offset = l_thick/2 - radius

        if self.vertical:
            self.line = py.Surface((l_thick, l_length))
            init_x = self.xpos + center_offset
            init_y = self.ypos + l_length*(1-slid_factor) - radius
        else:
            self.line = py.Surface((l_length, l_thick))
            init_y = self.ypos + center_offset
            init_x = self.xpos + l_length*slid_factor - radius

        self.slide_rect = py.Rect(init_x, init_y, radius*2, radius*2)
        self.line.fill(line_col)
        self.slide_surf = py.Surface(
            (self.slide_rect.width, self.slide_rect.height), py.SRCALPHA)

        py.draw.circle(surface=self.slide_surf,
                       color=slider_col,
                       center=(radius, radius),
                       radius=radius)

    def slide(self, mouse_pos: tuple):
        '''
        Moves the slider according to the given mouse position and updates self.val

        Parameters:
        - mouse_pos (tuple): Coordinates (x, y) of the current mouse position

        Note:
        - I think this code is a little bit verbose but I'm keeping it like this
            for readability
        '''
        if self.vertical:
            slider_pos = mouse_pos[1] - self.slide_rect.height/2
            min_y = self.ypos - self.slide_rect.height/2
            max_y = self.ypos + self.line.get_height() - self.slide_rect.height/2

            if slider_pos < min_y:
                slider_pos = min_y
            elif slider_pos > max_y:
                slider_pos = max_y

            self.slide_rect.y = slider_pos
            self.val = self.lo + ((self.hi-self.lo) / self.line.get_height()) * \
                (max_y - slider_pos)
        else:
            slider_pos = mouse_pos[0] - self.slide_rect.width/2
            min_x = self.xpos - self.slide_rect.width/2
            max_x = self.xpos + self.line.get_width() - self.slide_rect.width/2

            if slider_pos < min_x:
                slider_pos = min_x
            elif slider_pos > max_x:
                slider_pos = max_x

            self.slide_rect.x = slider_pos
            self.val = self.lo + ((self.hi-self.lo) / self.line.get_width()) * \
                (slider_pos + self.slide_rect.width/2 - self.xpos)

    def draw(self, win):
        win.blit(self.line, (self.xpos, self.ypos))
        win.blit(self.slide_surf, (self.slide_rect.x,
                 self.slide_rect.y))


class Canvas:
    def __init__(self, pos, pixel_dims, canvas_dims, init_r):
        '''
        Creates a canvas that one can sketch on

        Parameters:
        - pos (tuple): Coordinates (x, y) for the canvas placement
        - pixel_dims (tuple): Dimensions fo the array to convert (height, width)
        - disp_dims (tuple): Dimensions of the display (width, height)
        - init_r (int): Integer value of the pen radius

        Note:
        - in our self.pixels np array, a value of 0 means that pixel has been altered
            ie, sketching. And a value of 1 mens open space
        - Canvas is white while sketchings are black
        '''

        self.pos = pos
        self.canvas_dims = canvas_dims
        self.pixel_dims = pixel_dims
        self.r = init_r

        self.surf = py.Surface((pixel_dims[1], pixel_dims[0]))
        self.pixels = np.ones(pixel_dims)

    def sketch(self, mouse_pos):
        '''
        Parameters:
        - mouse_pos (tuple): Coordinates (x, y) of the current mouse position

        Note:
        - Both scale factor and canvas position changes the position of the actual
            pixel we are trying to click, so these calculation takes all that into 
            account
        - This should be called with a try/except call since this can cause 
            IndexErrors
        '''
        y_scale = self.pixel_dims[0] / self.canvas_dims[1]
        x_scale = self.pixel_dims[1] / self.canvas_dims[0]
        x_pos = int((mouse_pos[0] - self.pos[0]) * x_scale)
        y_pos = int((mouse_pos[1] - self.pos[1]) * y_scale)

        # won't even lie this np.ogrid wizardry is from ChatGPT
        rows, cols = self.pixel_dims
        y_indices, x_indices = np.ogrid[:rows, :cols]
        mask = (y_indices - y_pos)**2 + (x_indices - x_pos)**2 <= self.r**2
        self.pixels[mask] = 0

    def draw(self, win):
        '''
        Blits our self.pixels numpy array onto our previously initialised surface
            then blits it

        Note:
        - rgb_arr is made to fit the format that py.surfarray.blit_array() wants:
            -> 3D array
            -> Integer values rangeing from 0-255 
            -> (cols, rows) 
        '''
        rgb_arr = np.repeat(np.expand_dims(
            (self.pixels.T * 255), axis=-1), 3, axis=-1)

        py.surfarray.blit_array(self.surf, rgb_arr)
        scaled_surf = py.transform.scale(self.surf, self.canvas_dims)
        win.blit(scaled_surf, self.pos)

    def change_radius(self, r):
        self.r = r

    def get_np(self):
        return self.pixels

    def clear(self):
        self.pixels[:, :] = 1


class Display:
    def __init__(self, pos, pixel_dims, disp_dims):
        '''
        Creates a display that displays a numpy array

        Parameters:
        - pos (tuple): Position to place the display (x, y)
        - pixel_dims (tuple): Dimensions fo the array to convert (height, width)
        - disp_dims (tuple): Dimensions of the display (width, height)
        '''
        self.pos = pos
        self.canvas_dims = disp_dims
        self.surf = py.Surface((pixel_dims[1], pixel_dims[0]))

    def update(self, arr):
        '''
        Assigns given `arr` to `self.pixels`

        Parameters:
        - arr (2D array): array to be displayed (height, width) 0-1

        Note:
        - This is all to prepare for `py.surfarray.blit_array()` which takes only takes
            in 3D arrays of int values 0-255. Also it takes (width, height) 
            (pygame sucks)
        '''
        ph = np.expand_dims(np.transpose(arr) * 255, axis=-1)
        self.pixels = np.repeat(ph, 3, axis=-1)

    def draw(self, win):
        '''
        Blits our self.pixels numpy array onto our previously initialised surface
            then blits it

        Parameters:
        - win

        '''
        py.surfarray.blit_array(self.surf, self.pixels)
        scaled_surf = py.transform.scale(self.surf, self.canvas_dims)
        win.blit(scaled_surf, self.pos)


def draw_val(win, pos, text, font, text_col, bg_RGBA):
    '''
    Draws the text given with a background display

    Parameters:
    - win: Pygame master display
    - pos (tuple): Coordinates (x, y) for the text placement
    - text (str): String to be shown
    - font: pygame font being used
    - text_col (tuple): 3 element tuple specifying the RGB values of the text 
    - bg_RGBA (tuple): 4 element tuple specifying the RGBA values of the background
        if None then won't be displayed
    '''
    bg_pos = (pos[0] - 12, pos[1] - 30)
    val_pos = (bg_pos[0]+2, bg_pos[1] + 2)

    bg_surf = py.Surface((48, 25))
    bg_surf.fill(bg_RGBA[:-1]), bg_surf.set_alpha(bg_RGBA[-1])
    win.blit(bg_surf, bg_pos)

    val_surf = font.render(text, True, text_col)
    win.blit(val_surf, val_pos)


def run_model(model, canvas, display):
    '''
    Runs our input through the auto-encoder and uses the output to updates the disaply

    Parameters:
    - win
    - canvas
    - display
    '''
    arr = np.expand_dims(np.expand_dims(
        canvas.get_np(), axis=0), axis=-1)
    outp = model.predict(arr, verbose=0)
    display.update(outp[0, :, :, 0])


def draw_screen(win, bg, display, canvas, sliders: list, text_infos: tuple):
    '''

    Parameters:
    - win
    - bg (tuple): RGB info of the background colour 
    - display
    - canvas
    - sliders (list): each element is a slider object
    - text_infos (tuple): each element is a tuple (text, pos, col, font)
    '''
    win.fill(bg)
    display.draw(win)
    for slider in sliders:
        slider.draw(win)
    canvas.draw(win)
    for elem in text_infos:
        text, pos, col, font = elem
        surf = font.render(text, True, col)
        win.blit(surf, pos)


def main(model=None):
    '''
    Pygame interface for generating faces from user's skecthings

    Steps for "decent" results:
    - Start off by drawing the face shape, like a circle or oval.
    - Then colour in kind of where the mousth should be 
    '''
    # --------------- Pygame init things --------------- #
    py.font.init()
    fps = 60
    w_width, w_height = (1000, 600)
    win = py.display.set_mode((w_width, w_height))
    clock = py.time.Clock()

    # ------------------- Appearance Variables ------------------- #
    bg = (50, 50, 50)

    val_font = py.font.SysFont("sfcamera", 17, bold=True)
    value_text_col = (0, 0, 0)
    value_bg_rgba = (255, 244, 224, 200)

    misc_font = py.font.SysFont("Arial", 18, bold=True)
    misc_text_col = (255, 127, 91)

    guess_text = "Toggle Guess [Space]"
    guess_pos = (630, 490)
    guess_text_info = (guess_text, guess_pos, misc_text_col, misc_font)

    clear_text = "Clear Canvas [C]"
    clear_pos = (200, 490)
    clear_text_info = (clear_text, clear_pos, misc_text_col, misc_font)

    radius_text = "Pen Radius"
    radius_pos = (450, w_height-30)
    radius_text_info = (radius_text, radius_pos, misc_text_col, misc_font)

    all_text_info = (guess_text_info, clear_text_info, radius_text_info)

    slider_col = (246, 177, 122)
    line_col = (255, 255, 255)

    # ------------------- Init Sliders & Display ------------------- #
    slider = Slider(pos=(420, 550), lo=3, hi=10, init_val=5, line_dims=(150, 4),
                    radius=10, slider_col=slider_col, line_col=line_col,
                    v_orientation=False)

    canvas = Canvas(pos=(70, 30), pixel_dims=(171, 186), canvas_dims=(400, 450),
                    init_r=5)

    display = Display(pos=(520, 30), pixel_dims=(
        160, 176), disp_dims=(400, 450))

    # ------------------- Init variables for Loop ------------------- #
    prev_mouse = True
    slide_lock = False
    guess = False

    run_model(model, canvas, display)

    while 1:

        mouse1 = py.mouse.get_pressed()[0]

        if mouse1:
            mouse_pos = py.mouse.get_pos()

            if slider.slide_rect.collidepoint(mouse_pos):
                slide_lock = True

            if slide_lock:
                slider.slide(mouse_pos)
                draw_screen(win, bg, display, canvas, (slider,), all_text_info)

                draw_val(win=win, pos=(slider.slide_rect.x,
                                       slider.slide_rect.y),
                         text="{:05.2f}".format(slider.val), font=val_font,
                         text_col=value_text_col, bg_RGBA=value_bg_rgba)

                py.display.update()

            else:
                try:
                    canvas.sketch(mouse_pos)
                except IndexError:
                    pass
                else:
                    if guess:
                        run_model(model, canvas, display)
                        display.draw(win)
                    canvas.draw(win)
                    py.display.update()

        if (not mouse1) and prev_mouse:

            draw_screen(win, bg, display, canvas, (slider,), all_text_info)
            py.display.update()
            canvas.change_radius(slider.val)
            slide_lock = False

        prev_mouse = mouse1
        clock.tick(fps)
        for event in py.event.get():
            if event.type == py.QUIT:
                py.quit()
                sys.exit()
            if event.type == py.KEYDOWN:
                if event.key == py.K_SPACE:
                    run_model(model, canvas, display)
                    display.draw(win)
                    py.display.update()
                    guess = not guess

                if event.key == py.K_c:
                    canvas.clear()
                    canvas.draw(win)
                    py.display.update()


if __name__ == "__main__":
    main()
