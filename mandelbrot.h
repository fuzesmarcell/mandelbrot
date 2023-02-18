#ifndef MANDELBROT_H
#define MANDELBROT_H

#define MANDELBROT_MAX_ITERATIONS 1000

void mandelbrot_scalar(int* iter_buffer, int width, int height);
void mandelbrot_quad(int* iter_buffer, int width, int height);
void mandelbrot_oct(int* iter_buffer, int width, int height);

void mandelbrot_gpu(int* iter_buffer, int width, int height);

#endif // MANDELBROT_H