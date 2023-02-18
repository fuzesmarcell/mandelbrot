include(CheckLanguage)

cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(Mandelbrot VERSION 0.1.0 LANGUAGES CXX C)

add_executable(MandelbrotGUI WIN32)

target_sources(MandelbrotGUI PRIVATE
	"main.cpp"
	"mandelbrot.cpp"
	"mandelbrot.h")

check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
	enable_language(CUDA)
	target_sources(MandelbrotGUI PRIVATE
		"mandelbrot.cu"
		"mandelbrot_gpu.h")
endif()

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
	target_link_libraries(MandelbrotGUI PUBLIC OpenMP::OpenMP_CXX)
endif()