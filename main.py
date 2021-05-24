# import the pygame module, so you can use it
import pygame
import numpy as np
from fractals import init_gpu, get_fractal_func, cuda_kernel_from_func
from numba import cuda
import math

from matplotlib import cm

# define a main function
def main():
     
    # initialize the pygame module
    pygame.init()
    # load and set the logo
    #logo = pygame.image.load("logo32x32.png")
    #pygame.display.set_icon(logo)
    pygame.display.set_caption("minimal program")
     
    # create a surface on screen that has the size of 240 x 180
    DISPLAY_SIZE = (800,800)
    DISPLAY_RATIO = DISPLAY_SIZE[0]/ DISPLAY_SIZE[1]
    MAX_ITERS = 60
    FRACTAL = 'julia_exp'

    screen = pygame.display.set_mode(DISPLAY_SIZE)

    running = True

    cuda_info = init_gpu()
    cuda_kernel = cuda_kernel_from_func(*get_fractal_func(FRACTAL))

    @cuda.jit('void(float32,float32,float32,float32,uint8[:,:],uint8,float32,float32,float32)')
    def generate_fractal(min_x,max_x,min_y,max_y,image,max_iters,k1,k2,k3):
        height = image.shape[0]
        width = image.shape[1]
        pixel_size_x = (max_x - min_x)/width
        pixel_size_y = (max_y - min_y)/height
        
        startX, startY = cuda.grid(2)
        stridex, stridey = cuda.gridsize(2)
        
        for x in range(startX, width, stridex):
            real = min_x + x * pixel_size_x
            for y in range(startY, height, stridey):
                imag = min_y + y * pixel_size_y 
                image[y, x] = cuda_kernel(real, imag, max_iters,k1,k2,k3)

    # main loop
    screen_lims = [[-3,3],[-3/DISPLAY_RATIO,3/DISPLAY_RATIO]]
    speed = 0.02
    i = 0
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        pressed_keys=pygame.key.get_pressed()
        x_range = screen_lims[0][1] - screen_lims[0][0]
        y_range = screen_lims[1][1] - screen_lims[1][0]

        #Panning:
        screen_lims[0][0] += (pressed_keys[pygame.K_RIGHT] - pressed_keys[pygame.K_LEFT])*speed*x_range
        screen_lims[0][1] += (pressed_keys[pygame.K_RIGHT] - pressed_keys[pygame.K_LEFT])*speed*x_range
        screen_lims[1][0] += (pressed_keys[pygame.K_DOWN] - pressed_keys[pygame.K_UP])*speed*x_range
        screen_lims[1][1] += (pressed_keys[pygame.K_DOWN] - pressed_keys[pygame.K_UP])*speed*x_range

        #Zooming:
        zoom = 1.0 - speed*(pressed_keys[pygame.K_w] - pressed_keys[pygame.K_s])

        screen_lims[0][0] += 0.5*(x_range-zoom*x_range)
        screen_lims[0][1] -= 0.5*(x_range-zoom*x_range)
        screen_lims[1][0] += 0.5*(y_range-zoom*y_range)
        screen_lims[1][1] -= 0.5*(y_range-zoom*y_range)

        #Fractal:
        rendered_fractal = np.zeros((DISPLAY_SIZE[1],DISPLAY_SIZE[0]), dtype = np.uint8)

        g_image = cuda.to_device(rendered_fractal)
        generate_fractal[cuda_info['griddim'], cuda_info['blockdim']](screen_lims[0][0],screen_lims[0][1],screen_lims[1][0],screen_lims[1][1], g_image, MAX_ITERS, 2.0, -0.58 + 0.04*math.sin(10*i),0.1)
        g_image.to_host()

        #Coloring:
        cmap = cm.get_cmap('Reds')
        rendered_fractal = 255*cmap(rendered_fractal.T/MAX_ITERS)
        rendered_fractal = rendered_fractal[:,:,:3].astype(np.uint8)

        surf = pygame.surfarray.make_surface(rendered_fractal)
        screen.blit(surf, (0, 0))
        pygame.display.update()
        
        i += 0.01
     
if __name__=="__main__":
    main()