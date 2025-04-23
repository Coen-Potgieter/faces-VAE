import pygame as py
import sys
import numpy as np


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
        self.radius = radius

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
        Moves the slider according to the given mouse position

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

    def change_val(self, new_val):
        '''
        This is vile
        '''
        self.val = new_val
        l_length, _ = self.line_dims
        slid_factor = (self.val-self.lo) / (self.hi-self.lo)

        if self.vertical:
            self.slide_rect.y = self.ypos + \
                l_length*(1-slid_factor) - self.radius
        else:
            self.slide_rect.x = self.xpos + l_length*slid_factor - self.radius

    def draw(self, win):
        win.blit(self.line, (self.xpos, self.ypos))
        win.blit(self.slide_surf, (self.slide_rect.x,
                 self.slide_rect.y))


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


def new_face(model, latent_vectors, display, sliders):
    """
    Generates a new face from latent vectors

    Parameters:
    - model: Keras decoder
    - latent_vectors (2D array): Output of the autoencoder (examples, node_acts)
    - display
    - sliders

    Returns:
    - inp (2D array): input for the deocder on next update (examples, node_acts)
    """
    # r = np.random.randint(0, 500)
    r = np.random.randint(0, latent_vectors.shape[0])
    print("Showing Face:", r)


    inp = latent_vectors[r:r+1, :]
    outp = model.predict(inp, verbose=0)
    display.update(outp[0, :, :, 0])
    for idx, slider in enumerate(sliders):
        slider.change_val(inp[0, idx])
    return inp


def draw_screen(win, bg, display, sliders, text_info):
    '''

    Parameters:
    - win
    - bg (tuple): RGB info of the background colour 
    - display
    - sliders (list): each element is a slider object
    - text_infos (tuple): (text, pos, col, font)
    '''
    text, pos, col, font = text_info
    win.fill(bg)
    display.draw(win)
    for slider in sliders:
        slider.draw(win)
    val_surf = font.render(text, True, col)
    win.blit(val_surf, pos)


def run(model, latent_vectors):
    '''
    UI for generaring images by playing with the latent vectors

    Parameters:
    - model (keras model): Decoder model that builds images
    - latent_vectors (2D array): Output of the encoder (examples, node_acts)

    Steps for "decent" results:
    - Insight gained from `latent_space_inference()`:
        - Mean = 1.30
        - Standard Deviation = 0.55
    - From this I decided to make node values range from 0-3 (3 is over 2 STDs away from mean)
    - Keeping values around the mean (1.30) will, most likely, return the best results 
    - Tampering with the zero nodes could give "deformed" results

    Note:
    - Sliders/nodes follow a row-wise arrangement with "most influential" being first 
    - Can play around with colours/fonts 
    - Changing layout is a bit more involved
    - Zero nodes are separated from the rest
    - [SPACE] will return a given example that came out of the autoencoder (ie. a good result)
    - Finally, the method I use to determine the level of influence a node has is 
        completely baseless. (Total activation per node across a number of examples)
    '''
    # --------------- Pygame init things --------------- #
    py.font.init()
    win_width, win_height = 900, 600
    win = py.display.set_mode((win_width, win_height))
    fps = 60
    clock = py.time.Clock()

    # ------------------- Appearance Variables ------------------- #
    GREEN1 = (225, 238, 188)
    GREEN2 = (144, 198, 124)
    GREEN3 = (103, 174, 110)
    GREEN4 = (50, 142, 110)
    GREEN5 = (31, 125, 83, 0)

    # Sliders (the `slider` is the block that the runs along the `line`)
    line_dims = (80, 4)    # (length, thickness)
    slider_radius = 8  # (length, thickness)

    # Latent vector text (pop-up)
    val_font = py.font.SysFont("sfcamera", 17, bold=True)
    
    # Instruction text
    instr_font = py.font.SysFont("Arial", 18, bold=True)
    instr_pos = (0, 0)
    instr_text = "Generate Face: [SPACE]"

    # Colours
    bg = (25, 25, 25)
    value_text_col = GREEN5
    value_bg_rgba = [GREEN1[i] if i < 3 else 200 for i in range(4)]
    slider_col = GREEN1
    line_col = (255,255,255)
    instr_text_col = GREEN1


    text_info = (instr_text, instr_pos, instr_text_col, instr_font)
    # ------------------- Init Latent vectors ------------------- #
    lo = 0
    hi = 3
    latent_vectors = np.clip(latent_vectors, lo, hi)
    np.random.shuffle(latent_vectors)

    inp = latent_vectors[0:1, :]
    outp = model.predict(inp, verbose=0)

    # These two lines are to sort the sliders on order of influence
    total_activation = np.sum(latent_vectors, axis=0)
    descending_acts = np.argsort(total_activation)[::-1]

    # ------------------- Init Sliders & Display ------------------- #
    sliders = [None for _ in range(inp.shape[1])]
    act_idx = 0
    # Having to 2 sets of nested for loops is ugly but sorts the nodes based on influence
    for y in range(4):
        for x in range(13):
            # This ensures the "most influential" sliders are to be placed first
            slider_idx = descending_acts[act_idx]
            sliders[slider_idx] = Slider(pos=(520 + 29*x, 30 + 100*y),
                                         lo=lo, hi=hi, init_val=inp[0,
                                                                    slider_idx],
                                         line_dims=line_dims, radius=slider_radius,
                                         slider_col=slider_col, line_col=line_col,
                                         v_orientation=True)
            act_idx += 1

    # These are the 0 nodes
    line_dims = (50, 3)
    slider_radius = 5
    for y in range(2):
        for x in range(14):
            # This ensures the "most influential" sliders are to be placed first
            slider_idx = descending_acts[act_idx]
            sliders[slider_idx] = Slider(pos=(560 + 20*x, 450 + 70*y),
                                         lo=lo, hi=hi, init_val=inp[0,
                                                                    slider_idx],
                                         line_dims=line_dims, radius=slider_radius,
                                         slider_col=slider_col, line_col=line_col,
                                         v_orientation=True)
            act_idx += 1

    display = Display(pos=(0, 0),
                      pixel_dims=(160, 176),
                      disp_dims=(500, win_height))

    # ------------------- Init variables for Loop ------------------- #
    slide_lock = False
    prev_mouse1 = False

    display.update(outp[0, :, :, 0])

    draw_screen(win, bg, display, sliders, text_info)
    py.display.update()

    while 1:
        clock.tick(fps)
        mouse1 = py.mouse.get_pressed()[0]

        if mouse1:
            mouse_pos = py.mouse.get_pos()
            if slide_lock:
                target_silder = sliders[target_idx]  # get slider being clicked
                changing_val = target_silder.val    # get val for slider
                target_silder.slide(mouse_pos)
                inp[0, target_idx] = changing_val  # change input
                outp = model.predict(inp, verbose=0)  # get output from decoder
                display.update(outp[0, :, :, 0])

                draw_screen(win, bg, display, sliders, text_info)

                # draws value when slider is clicked
                draw_val(win=win, pos=(target_silder.slide_rect.x,
                                       target_silder.slide_rect.y),
                         text="{:05.2f}".format(changing_val), font=val_font,
                         text_col=value_text_col, bg_RGBA=value_bg_rgba)

            # gets slider being clicked
            for idx, single_slider in enumerate(sliders):
                if single_slider.slide_rect.collidepoint(mouse_pos):
                    slide_lock = True
                    target_idx = idx

            py.display.update()

        # only updates on mouse release
        elif (not mouse1) and (prev_mouse1):
            slide_lock = False
            draw_screen(win, bg, display, sliders, text_info)
            py.display.update()

        prev_mouse1 = mouse1
        for event in py.event.get():
            if event.type == py.QUIT:
                py.quit()
                sys.exit()
            if event.type == py.KEYDOWN:
                if event.key == py.K_SPACE:
                    inp = new_face(model, latent_vectors,
                                   display, sliders)
                    draw_screen(win, bg, display, sliders, text_info)
                    py.display.update()
