# LogTriC
Logical Triangulation in C

For a detailed explanation, see:

https://youtu.be/mbUTe9bJ-CM

You can compile any of these programs on Linux e.g. as follows:

gcc -O3 -o tric tric_whatever.c -lm

You also need to create new atoms with

gcc -o atomgen atomgen.c

and then:

./atomgen

this will generate a 40MB data file.

And then you just do:

./tric

For the OpenMP versions, do not forget the -fopenmp flag. The OpenCL versions are compiled according to your OpenCL implementation, for me it was e.g. (not optimizing):

gcc -I/opt/AMDAPPSDK-2.9-1/include -L/opt/AMDAPPSDK-2.9-1/lib/x86_64 -o tric tric_parser_ocl20170507.c -lm -lOpenCL -g

/usr/local/cuda/bin/nvcc -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -o tric tric_parser_ocl20170507.c -lm -lOpenCL -g

These programs will show you a lot of diagnostic output. Nonetheless, they work simply: you enter text. It should be parsed into numbers, then a reply should be produced. If you are wish to terminate, just press enter without entering further text.

Nino Ivanov

