#include "mandelbrot.h"

#include <xmmintrin.h>
#include <emmintrin.h>
#include <smmintrin.h>
#include <immintrin.h>

#include <assert.h>

#include <omp.h>

void mandelbrot_scalar(int* iter_buffer, int width, int height) {
	float h = ((0.47f + 2.00f) / (float)(width));
	float v = ((1.12f + 1.12f) / (float)(height));
	
#pragma omp parallel for
	for (int j = 0; j < height; j++) {
		for (int i = 0; i < width; i++) {
			float x0 = (h * (float)i) - 2.00f;
			float y0 = (v * (float)j) - 1.12f;

			float x = 0.f;
			float y = 0.f;

			int iteration = 0;
			while (x*x + y*y <= 2.f*2.f && iteration < MANDELBROT_MAX_ITERATIONS) {
				float x_temp = x*x - y*y + x0;
				y = 2.f*x*y + y0;
				x = x_temp;
				iteration++;
			}

			iter_buffer[j*width + i] = iteration;
		}
	}
}

void mandelbrot_quad(int* iter_buffer, int width, int height) {

	assert(width % 4 == 0);

	if ((width % 4) != 0) {
		return;
	}

	__m128 h = _mm_set_ps1((0.47f + 2.00f) / (float)(width));
	__m128 v = _mm_set_ps1((1.12f + 1.12f) / (float)(height));

	__m128 two_sq = _mm_set_ps1(2.f*2.f);
	__m128i max_iterations = _mm_set1_epi32(MANDELBROT_MAX_ITERATIONS);

#pragma omp parallel for
	for (int j = 0; j < height; j++) {
		__m128 scale_v = _mm_mul_ps(v, _mm_set_ps1((float)j));
		__m128 y0 = _mm_sub_ps(scale_v, _mm_set_ps1(1.12f));

		for (int i = 0; i < width; i += 4) {
			__m128 scale_h = _mm_set_ps((float)(i + 3),
										(float)(i + 2),
										(float)(i + 1),
										(float)(i + 0));

			__m128 x0 = _mm_mul_ps(h, scale_h);
			x0 = _mm_sub_ps(x0, _mm_set_ps1(2.f));

			__m128 x = _mm_set_ps1(0.f);
			__m128 y = _mm_set_ps1(0.f);
			__m128i iterations = _mm_set1_epi32(0);

			for (;;) {
				// x*x + y*y <= 2.f*2.f
				__m128 x_sq_x = _mm_mul_ps(x, x);
				__m128 y_sq_y = _mm_mul_ps(y, y);
				__m128 sq_res = _mm_add_ps(x_sq_x, y_sq_y);
				__m128i in_set_mask = _mm_castps_si128(_mm_cmple_ps(sq_res, two_sq));

				if (!in_set_mask.m128i_u32[0] &&
					!in_set_mask.m128i_u32[1] &&
					!in_set_mask.m128i_u32[2] &&
					!in_set_mask.m128i_u32[3]) {
					break;
				}

				// float x_temp = x*x - y*y + x0;
				// y = 2.f*x*y + y0;
				// x = x_temp;
				__m128 x_temp = _mm_add_ps(_mm_sub_ps(x_sq_x, y_sq_y), x0);
				y = _mm_mul_ps(x, y);
				y = _mm_mul_ps(_mm_set_ps1(2.f), y);
				y = _mm_add_ps(y, y0);
				x = x_temp;

				// iteration++;
				// iteration < MANDELBROT_MAX_ITERATIONS
				__m128i iter_added = _mm_add_epi32(iterations, _mm_set1_epi32(1));
				__m128i added_iterations = _mm_and_epi32(iter_added, in_set_mask);
				__m128i old_iterations = _mm_andnot_epi32(in_set_mask, iterations);
				iterations = _mm_or_epi32(added_iterations, old_iterations);

				__m128i iter_mask = _mm_cmplt_epi32(iterations, max_iterations);

				if (!iter_mask.m128i_u32[0] ||
					!iter_mask.m128i_u32[1] ||
					!iter_mask.m128i_u32[2] ||
					!iter_mask.m128i_u32[3]) {
					break;
				}
			};

			*(__m128i*)(&iter_buffer[j*width + i]) = iterations;

		}
	}
}

void mandelbrot_oct(int* iter_buffer, int width, int height) {

	assert(width % 8 == 0);
	if ((width % 8) != 0) {
		return;
	}

	__m256 h = _mm256_set1_ps((0.47f + 2.00f) / (float)(width));
	__m256 v = _mm256_set1_ps((1.12f + 1.12f) / (float)(height));

	__m256 two_sq = _mm256_set1_ps(2.f*2.f);
	__m256i max_iterations = _mm256_set1_epi32(MANDELBROT_MAX_ITERATIONS);

#pragma omp parallel for
	for (int j = 0; j < height; j++) {
		__m256 scale_v = _mm256_mul_ps(v, _mm256_set1_ps((float)j));
		__m256 y0 = _mm256_sub_ps(scale_v, _mm256_set1_ps(1.12f));

		for (int i = 0; i < width; i += 8) {
			__m256 scale_h = _mm256_set_ps((float)(i + 7),
										   (float)(i + 6),
										   (float)(i + 5),
										   (float)(i + 4),
										   (float)(i + 3),
										   (float)(i + 2),
										   (float)(i + 1),
			                               (float)(i + 0));
			__m256 x0 = _mm256_mul_ps(h, scale_h);
			x0 = _mm256_sub_ps(x0, _mm256_set1_ps(2.f));

			__m256 x = _mm256_set1_ps(0.f);
			__m256 y = _mm256_set1_ps(0.f);
			__m256i iterations = _mm256_set1_epi32(0);

			for (;;) {
				// x*x + y*y <= 2.f*2.f
				__m256 x_sq_x = _mm256_mul_ps(x, x);
				__m256 y_sq_y = _mm256_mul_ps(y, y);
				__m256 sq_res = _mm256_add_ps(x_sq_x, y_sq_y);
				__m256i in_set_mask = _mm256_castps_si256(_mm256_cmp_ps(sq_res, two_sq, _CMP_LE_OS));
				
				if (!in_set_mask.m256i_u32[0] &&
					!in_set_mask.m256i_u32[1] &&
					!in_set_mask.m256i_u32[2] &&
					!in_set_mask.m256i_u32[3] &&
					!in_set_mask.m256i_u32[4] &&
					!in_set_mask.m256i_u32[5] &&
					!in_set_mask.m256i_u32[6] &&
					!in_set_mask.m256i_u32[7]) {
					break;
				}

				// float x_temp = x*x - y*y + x0;
				// y = 2.f*x*y + y0;
				// x = x_temp;
				__m256 x_temp = _mm256_add_ps(_mm256_sub_ps(x_sq_x, y_sq_y), x0);
				y = _mm256_mul_ps(x, y);
				y = _mm256_mul_ps(_mm256_set1_ps(2.f), y);
				y = _mm256_add_ps(y, y0);
				x = x_temp;

				// iteration++;
				// iteration < MANDELBROT_MAX_ITERATIONS
				__m256i iter_added = _mm256_add_epi32(iterations, _mm256_set1_epi32(1));
				__m256i added_iterations = _mm256_and_epi32(iter_added, in_set_mask);
				__m256i old_iterations = _mm256_andnot_epi32(in_set_mask, iterations);
				iterations = _mm256_or_epi32(added_iterations, old_iterations);

				// TODO: Understand this exactly!
				// check = ( a <= b ) = ~(a > b) & 0xF..F
				__m256i iter_mask_temp = _mm256_cmpgt_epi32(iterations, max_iterations);
				__m256i iter_mask = _mm256_andnot_epi32(iter_mask_temp, _mm256_set1_epi32(-1));
				
				if (!iter_mask.m256i_u32[0] ||
					!iter_mask.m256i_u32[1] ||
					!iter_mask.m256i_u32[2] ||
					!iter_mask.m256i_u32[3] ||
					!iter_mask.m256i_u32[4] ||
					!iter_mask.m256i_u32[5] ||
					!iter_mask.m256i_u32[6] ||
					!iter_mask.m256i_u32[7]) {
					break;
				}
			}

			*(__m256i*)(&iter_buffer[j*width + i]) = iterations;

		}
	}
}
