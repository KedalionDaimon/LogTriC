# LogTriC
Logical Triangulation in C

tric_sledge_2.c is a raw version of the pure logical triangulation version, written in C.

You can compile it on Linux e.g. as follows:

gcc -lm -o tric tric_sledge_2.c 

and then:

./tric

will run it and show you A LOT of diagnostic output.

To run the main program, get tric_parser.c with all the auxiliary files and then:

gcc -lm -o tric tric_parser.c

whereupon you run it with:

./tric

... here, you enter text. It should be parsed into numbers, then a reply should be produced. If you are wish to terminate, just press enter without entering further text.

