#ifndef UNICODE
#define UNICODE
#endif
#define WIN32_LEAN_AND_MEAN
#define _CRT_SECURE_NO_DEPRECATE
#include <Windows.h>
#include <stdlib.h>
#include <stdio.h>

#include "mandelbrot.h"
#include "mandelbrot_gpu.h"

#ifdef _DEBUG
#define Assert(Cond) do { if (!(Cond)) __debugbreak(); } while (0)
#else
#define Assert(Cond) (void)(Cond)
#endif
#define HR(hr) do { HRESULT _hr = (hr); Assert(SUCCEEDED(_hr)); } while (0)

#define ArraySize(arr) (sizeof(arr) / sizeof((arr)[0]))

#define ID_SCALAR 1
#define ID_SSE 2
#define ID_AVX 3
#define ID_CUDA 4

LARGE_INTEGER g_freq;
DWORD* g_pixels;
BITMAPINFO g_bmp_info;
int* g_iters;
int g_iters_width;
int g_iters_height;

enum MandelbrotBackend {
	BACKEND_SCALAR = 1,
	BACKEND_SSE,
	BACKEND_AVX,
	BACKEND_CUDA,
} g_backend;

static LRESULT CALLBACK window_proc(HWND window, UINT message, WPARAM wparam, LPARAM lparam) {

	switch (message) {

	case WM_CREATE: {
		HINSTANCE instance = GetModuleHandleW(NULL);
		CreateWindowExW(0, L"Button", L"Choose backend",
						WS_CHILD | WS_VISIBLE | BS_GROUPBOX,
						10, 10, 120, 150, window, (HMENU)0, instance, NULL);
		CreateWindowExW(0, L"Button", L"Scalar",
						WS_CHILD | WS_VISIBLE | BS_AUTORADIOBUTTON,
						20, 30, 100, 30, window, (HMENU)ID_SCALAR, instance, NULL);
		CreateWindowExW(0, L"Button", L"SSE",
						WS_CHILD | WS_VISIBLE | BS_AUTORADIOBUTTON,
						20, 55, 100, 30, window, (HMENU)ID_SSE, instance, NULL);
		CreateWindowExW(0, L"Button", L"AVX",
						WS_CHILD | WS_VISIBLE | BS_AUTORADIOBUTTON,
						20, 80, 100, 30, window, (HMENU)ID_AVX, instance, NULL);
		CreateWindowExW(0, L"Button", L"CUDA",
						WS_CHILD | WS_VISIBLE | BS_AUTORADIOBUTTON,
						20, 105, 100, 30, window, (HMENU)ID_CUDA, instance, NULL);
	} break;

	case WM_COMMAND: {
		if (HIWORD(wparam) == BN_CLICKED) {
			switch (LOWORD(wparam)) {
			case ID_SCALAR: {
				g_backend = BACKEND_SCALAR;
			} break;

			case ID_SSE: {
				g_backend = BACKEND_SSE;
			} break;

			case ID_AVX: {
				g_backend = BACKEND_AVX;
			} break;

			case ID_CUDA: {
				g_backend = BACKEND_CUDA;
			} break;

			}
			InvalidateRect(window, NULL, TRUE);
		}
	} break;

	case WM_CLOSE: {
		DestroyWindow(window);
	} break;

	case WM_DESTROY: {
		PostQuitMessage(0);
	} break;

	case WM_SIZE: {
		if (g_pixels != NULL) {
			free(g_pixels);
		}
		if (g_iters != NULL) {
			free(g_iters);
		}

		RECT dim;
		GetClientRect(window, &dim);

		g_bmp_info.bmiHeader.biSize = sizeof(g_bmp_info.bmiHeader);
		g_bmp_info.bmiHeader.biWidth = dim.right - dim.left;
		g_bmp_info.bmiHeader.biHeight = dim.bottom - dim.top;
		g_bmp_info.bmiHeader.biPlanes = 1;
		g_bmp_info.bmiHeader.biBitCount = 32;
		g_bmp_info.bmiHeader.biCompression = BI_RGB;
		size_t bytes_per_pixel = (g_bmp_info.bmiHeader.biBitCount / 8);

		// In case we need to pass this to SIMD/AVX routines we
		// should make this a multiple of 8
		g_iters_width = (g_bmp_info.bmiHeader.biWidth + 7) & ~0x7;
		g_iters_height = g_bmp_info.bmiHeader.biHeight;

		g_pixels = (DWORD*)malloc(g_bmp_info.bmiHeader.biWidth * g_bmp_info.bmiHeader.biHeight * bytes_per_pixel);
		g_iters = (int*)malloc(g_iters_width*g_iters_height*sizeof(int));

		InvalidateRect(window, NULL, TRUE);
	} break;

	case WM_PAINT: {
		PAINTSTRUCT paint;
		HDC hdc = BeginPaint(window, &paint);

		RECT dim;
		GetClientRect(window, &dim);

		LARGE_INTEGER begin;
		QueryPerformanceCounter(&begin);

		switch (g_backend) {
		case BACKEND_SCALAR: {
			mandelbrot_scalar(g_iters, g_iters_width, g_iters_height);
		} break;
		case BACKEND_SSE: {
			mandelbrot_quad(g_iters, g_iters_width, g_iters_height);
		} break;
		case BACKEND_AVX: {
			mandelbrot_oct(g_iters, g_iters_width, g_iters_height);
		} break;
		case BACKEND_CUDA: {
			mandelbrot_gpu(g_iters, g_iters_width, g_iters_height);
		} break;
		default: {
			mandelbrot_scalar(g_iters, g_iters_width, g_iters_height);
		} break;
		}

		LARGE_INTEGER end;
		QueryPerformanceCounter(&end);
		double time_elapsed = (double)(end.QuadPart - begin.QuadPart) / g_freq.QuadPart;

		char buffer[256];
		sprintf(buffer, "mandelbrot: %fms\n", time_elapsed*1000.);
		OutputDebugStringA(buffer);

		for (LONG j = dim.top; j < dim.bottom; j++) {
			for (LONG i = dim.left; i < dim.right; i++) {
				float t = ((float)(g_iters[j*g_iters_width + i]) / (float)(MANDELBROT_MAX_ITERATIONS));

				float r = (1 - t)*255.f + t*25.f;
				float g = (1 - t)*255.f + t*25.f;
				float b = (1 - t)*255.f + t*25.f;

				g_pixels[j*g_bmp_info.bmiHeader.biWidth + i] = (BYTE)(b) | ((BYTE)(g) << 8) | ((BYTE)(r) << 16);
			}
		}

		HBRUSH brush = CreateSolidBrush(RGB(0, 0, 0));
		FillRect(hdc, &dim, brush);
		DeleteObject(brush);

		StretchDIBits(hdc,
					  0, 0, (dim.right - dim.left), (dim.bottom - dim.top),
					  0, 0, (dim.right - dim.left), (dim.bottom - dim.top),
					  g_pixels, &g_bmp_info, DIB_RGB_COLORS, SRCCOPY);

		EndPaint(window, &paint);
	} break;

	default: {
		return DefWindowProcW(window, message, wparam, lparam);
	} break;

	}

	return 0;
}

int WINAPI WinMain(HINSTANCE instance, HINSTANCE previous, LPSTR cmdline, int cmdshow) {

	QueryPerformanceFrequency(&g_freq);

	WNDCLASSEXW window_class = {};
	window_class.cbSize = sizeof(window_class);
	window_class.lpfnWndProc = window_proc;
	window_class.hInstance = GetModuleHandleW(NULL);
	window_class.lpszClassName = L"mandelbrot_set_wnd_class";

	if (!RegisterClassExW(&window_class)) {
		ExitProcess(0);
	}

	DWORD style = WS_OVERLAPPEDWINDOW;

	RECT window_rect = { 0, 0, 320*3, 180*3 };
	AdjustWindowRectEx(&window_rect, style, FALSE, 0);

	int width = window_rect.right - window_rect.left;
	int height = window_rect.bottom - window_rect.top;

	HWND window = CreateWindowExW(0,
								  window_class.lpszClassName,
								  L"Mandelbrot set", style,
								  CW_USEDEFAULT, CW_USEDEFAULT, width, height,
								  NULL, NULL, window_class.hInstance, NULL);
	if (!window) {
		ExitProcess(0);
	}

	ShowWindow(window, cmdshow);
	UpdateWindow(window);

	for (;;) {
		MSG message;
		BOOL result = GetMessageW(&message, NULL, 0, 0);
		if (!result) {
			ExitProcess(0);
		}

		Assert(result > 0);

		TranslateMessage(&message);
		DispatchMessageW(&message);
	}

	return 0;
}