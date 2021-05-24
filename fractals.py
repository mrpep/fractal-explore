import numpy as np

from numba import cuda
from numba import *
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32
import cmath
import math

def init_gpu(blockdim=(8, 8), griddim=(100,100)):
    total_threads = np.prod(blockdim)
    total_blocks = np.prod(griddim)

    #nstates = int(total_threads * total_blocks)
    #rng_states = create_xoroshiro128p_states(nstates, seed=1)

    return {'blockdim': blockdim, 'griddim': griddim}

def mandelbrot_power(x,y,max_iters = 50,power_z=2.0,power_c=1.0):
    r_c, theta_c =cmath.polar(complex(x,y))
    c = math.pow(r_c,power_c)*cmath.exp(1j*theta_c*power_c)
    z = complex(0,0)
    i = 0
    while i<max_iters and abs(z) <=2:
        r, theta = cmath.polar(z)
        z = math.pow(r,power_z)*cmath.exp(1j*theta*power_z) + c
        i += 1
    return i

def mandelbrot_cosine(x,y,max_iters = 50,power_z=2.0,power_c=1.0):
    r_c, theta_c =cmath.polar(complex(x,y))
    c = math.pow(r_c,power_c)*cmath.exp(1j*theta_c*power_c)
    z = complex(0,0)
    i = 0
    while i<max_iters and abs(z) <=2:
        r, theta = cmath.polar(z)
        z = cmath.cos(math.pow(r,power_z)*cmath.exp(1j*theta*power_z)) + c
        i += 1
    return i

def burning_ship(x,y,max_iters=50,power_z=1.0,power_c=1.0):
    c=complex(x,y)
    z = 0
    i=0
    while i<max_iters and abs(z) <=2:
        z = complex(abs(z.real),abs(z.imag))*complex(abs(z.real),abs(z.imag)) + c
        i += 1
    return i

def julia_exp(x,y,max_iters=50,power_z=1.0,cx=0,cy=0):
    c=complex(cx,cy)
    z=complex(x,y)
    i=0
    while i<max_iters and abs(z) <=2:
        r, theta = cmath.polar(z)
        z = cmath.exp(math.pow(r,power_z)*cmath.exp(1j*theta*power_z)) + c
        i += 1
    return i

def get_fractal_func(fractal_type):
    fractals = {'mandelbrot_power': (mandelbrot_power, 'uint8(float32,float32,uint8,float32,float32)'),
                'mandelbrot_cosine': (mandelbrot_cosine, 'uint8(float32,float32,uint8,float32,float32)'),
                'burning_ship': (burning_ship, 'uint8(float32,float32,uint8,float32,float32)'),
                'julia_exp': (julia_exp, 'uint8(float32,float32,uint8,float32,float32,float32)')}

    return fractals[fractal_type]

def cuda_kernel_from_func(func, signature):
    print(signature)
    return cuda.jit(signature,device=True)(func)




