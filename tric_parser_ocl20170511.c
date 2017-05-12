/* IMPORTANT ERROR CORRECTIONS:

- /* TRIANGULATE: was working on atoms[i] instead of atoms[0] -
  but triangulation is done on basis of the PERCEIVED CURRENT connection,
  this was entirely pointless!

- bubbling was taking the BUBBLE VALUE from the CONJUGATE/COHABITANT,
  instead of the ATOMS ARRAY - effectively denying all triangulation */

/* BUBBLING MULTIPROCESSOR VERSION, NO SNOWFLAKE */

/* gcc -fopenmp -floop-interchange -floop-strip-mine -floop-block -ftree-parallelize-loops=8 -O3 -o tric tric_parser_multi20170501.c -lm */

/* I have no -floop-parallelize-all. */

/*
This program serves to demonstrate an artificial intellgence operating
by means of Logical Triangulation.

It operates in several phases of execution:

1. A numeric input array is acquired. This input contains several
numbers that represent symbols of sensory perception, so-called
"elementary atoms". If the input is "empty", evaluation terminates.
That is why the atom "0" is special: nothing is really done with the
atom 0.
 
2. It is attempted to "hierarchise" that input, that is, to group
together symbols in vic-connections. If the input initially consisted
out of "K L M N A B C D" and X is symbolising A-B, then the input
shall become "K L M N X C D 0". If "Y" is symbolising "N X", then in
the next phase of hierarchisation the input shall become
"K L M Y C D 0 0". (The input array is padded with zeroes in order to
fill up the space.) This goes on until some "top atom" Z is found, so
that the input becomes "Z 0 0 0 0 0 0 0". If a connection is unknown,
it may be created as hypothesis.

3. Each hierarchisation is undertaken according to whichever vic-
connection has been the strongest. In A B C, the conenctions could be
A-B C or A B-C. That grouping will be preferred, which has a more
negative binding. vic-connections are symbols of negative value,
ana-connections are symbols of positive value.

4. Every "higher" atom "above" the elementary atoms has the following
structure: [atomnumber][first-subatom][second-subatom][atomvalue];
the first sub-atom is also called "left atom" and the second sub-atom
is called "right atom".

5. After every hierarchisation step, the "winning" vic-connection is
taken as a starting point for performing logical triangulation. That
is, if A-B is grouped, then all sorts of triangles are sought that
involve it, e.g. - A-B-F-A, + A-B=G-A, etc. This, ultimately, serves
in order to "propagate" the effects of connections throughout the
system's knowledge base (that is, OTHER connections are affected by
A-B).

6. The creation of new connections in the knowledge base entails the
forgetting of old connections. New connections are created either as
conclusions from triangulation hypotheses (vic- and ana-atoms), or as
hypotheses regarding the grouping of input atoms (vic-atoms).

7. The triangulation process is extended and repeated several times
in order to achieve greater influence on the system's knowledge from
the current input.

8. After having finished every hierarchisation step with the ensuing
triangulation, a plan is created. A plan is an extension of the
"present". If the "present" has been hierarchised to the atom Z, then
an atom R is sought, whereby R is vic-connected to Z, that is, we are
looking for an atom S which consists out of Z-R. R is output as reply.
If such atom S cannot be found, then Z is "decomposed" into its two
sub-atoms, e.g. P-Q. Then you are looking for an atom O, which is
consisting out of Q-R, and give out that R. - If you finally reach
an elementary atom E, but there is NO known atom E-R, then simply the
system does not output any answer.

9. Regarding "directionality": The REASONING of the system is NON-
DIRECTIONAL, but the REPRESENTATION of the atoms - which sub-atom
"follows" which other sub-atom (that is, in A-B, "A comes before B")
is kept constant.

10. That "plan" R is then decomposed into sub-atoms, sub-sub-atoms,
and so forth, until it is shown as a vector of elementary atoms.
These elementary atoms constitute the output to the user and the
cycle begins anew. (For the sake of better internal consideration,
the system may actually create a few "hypothetical" plans before it
"really" answers.)
*/


/* consider the compiler flag -fstrict-aliasing in gcc */

/* This program operates on uppercase only - lowercase is automatically translated. The #-sign is used as I/O-terminator. */

/* Input must be followed by a " " to be perceived. - This is useful e.g. to discern only partial-word input. */

/* plans for greatness:

historydata.txt - length of the history - 32 lines - make it 1000, and make the input length in characters something like 10k

importancedata.txt - WORDSKNOWN - 2000 lines - make it 32000

worddata.txt - WORDSKNOWN * LETTERSPERWORD

*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* BEGIN OPENCL */
#include <sys/types.h>
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#include <unistd.h>
#else
#include <CL/cl.h>
#endif

// #include "err_code.h"

//pick up device type from compiler command line or from
//the default type
#ifndef DEVICE
#define DEVICE CL_DEVICE_TYPE_GPU
// #define DEVICE CL_DEVICE_TYPE_DEFAULT
// #define DEVICE CL_DEVICE_TYPE_CPU
// #define DEVICE CL_DEVICE_TYPE_GPU
#endif
/* END OPENCL */

/* DEFS FROM LOGICAL TRIANGULATION */

#define lowestatom 10001
/* 1000000 seems to be the slowest still tolerable thing, 100k is pretty OK */
#define atomcount 3000000
#define leninput 40
/* 60 would be nice, but in ONE full cycle, it cost me ca. 6k atoms - that is too much! */
#define atomnumber 0
#define leftatom 1
#define rightatom 2
#define atomvalue 3
#define positioninatoms 4
/* VICVALUE * LENINPUT * RECONSIDERATIONS * SLEDGE < MAXVIC or */
/* VICVALUE * LENINPUT * RECONSIDERATIONS * SLEDGE < MAXVIC */
#define vicvalue -1000
/* #define vicvalue -50 or -300 or thelike*/
#define maxvic -30000
#define maxana 30000
#define reconsiderations 3
#define sledge 2

/* avoid generating more than n triangles */
/* # define maxtri atomcount */
# define maxtri 4000

/* THIS IS WITH DOUBLETTS - that is, make it double as large as desired: */
# define trilimit 4000

/* absolute range limit for triangulation - disadviseable; but faster */
/* atoms behind that range will not even be considered as possible triangles sides */
# define absolutetrilimit 4000

/* END OF LOGICAL TRIANGULATION DEFS */

/* WORDSKNOWN is how many words the system may know - the count of columns of worddata.txt */
#define WORDSKNOWN 10000
/* LETTERSPERWORD determines how many letters a word may have - more is better, the increase in memory requirements is pretty unimportant */
#define LETTERSPERWORD 20
/* INPUTLENGTH defines the maximum size of input and output in chars. */
#define INPUTLENGTH 1000

/* BY PLACING THE LARGE ARRAYS OUTSIDE THE MAIN FUNCTION, I WEASEL MY WAY AROUND STACK LIMITATIONS - ALLOCATION THEN IS IN MAIN MEMORY. */

/* MAXHISTORY defines the maximum size of input in words. */
/* If maxhistory is longer, the system may understand more differentiated input - but is harder to teach in general. */
#define MAXHISTORY leninput /* Or make this 64 */

/* inputwords[] is where the initial input is being read into */
char inputwords[INPUTLENGTH];
/* nextword is an auxiliary array used during parsing to hold the next word */
char nextword[LETTERSPERWORD];
/* readwords is an auxiliary array used during parsing to hold the extracted words */
char readwords[MAXHISTORY][LETTERSPERWORD];
/* knownwords is an important array containing the textual representation of the words that the system knows */
char knownwords[WORDSKNOWN][LETTERSPERWORD];

char textreply[INPUTLENGTH];

/* wordnumber is an array that contains the numeric (non-binary) representations of the perceived input words */
int wordnumber[MAXHISTORY];

int lasthistory[MAXHISTORY];

/* the lower the position in importantwords, the more UNimportant */
int importantwords[WORDSKNOWN];




/* FROM LOGICAL TRIANGULATION */

/* PLACE THIS OUTSIDE OF MAIN() TO GET A HEAP ALLOCATION */

/* when parallelizing, we will modify maxtriangles */
int maxtriangles = maxtri;

/* INPUTARRAY IS THE SLIDING INPUT WINDOW */
/* int inputarray[leninput] = {11, 21, 31, 41, 51, 61, 71, 81, 91, 0}; */
/* int inputarray[leninput] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 0}; */
int inputarray[leninput];

/* NOW FOLLOW ARRAYS FOR TEMPORARY COPOES OF THE INPUT */
int copyinputarray[leninput];
int extensioninput[leninput];

/* THIS IS THE ARRAY FOR PRESENTING OUTPUT OF THE SYSTEM */
int outputarray[leninput];

/* THIS IS THE SYSTEM'S MAIN KNOWLEDGE:
[atomnumber][leftatom][rightatom][atomvalue]
*/
/* main atom array */
int atoms[atomcount][4];
int atomscopy[atomcount][4];

/* triangulation arrays */

/* CONJUGATE: second triangle side;
5 atoms, not 4, because you
ALSO want to know WHERE the atom
was found. The first triangle
side is regarded as GIVEN. */
int conjugate[maxtri][5];

/* The third triangle side is
found in the COHABITANT array. */
int cohabitant[maxtri][5];

/* Cohabitants can be CONCLUSIONS,
that is SO FAR UNKNOWN ATOMS, or -
HYPOTHESES. */
int hypotheses[maxtri][4];
/* auxiliary arrays: */
int scrubhypotheses[maxtri][4];
int hypothesescount;
int scrubhypothesescount;
int alreadydone;

/* The values of triangulation -
as this can be "value * value",
the capacity should be large. */
int atomeffects[maxtri];
int atomeffectssum[maxtri];
int atomeffectsdivisor;
int conjugateeffects[maxtri];
int cohabitanteffects[maxtri];

int atomnumbertracer = 0;

int firstatom = 0;
int secondatom = 0;
int thirdatom = 0;

/* Candidate for input hierarchisation: */
int candidateatom = 0;
int candidatevalue = 0; 
int candidateleft = 0;
int candidateright = 0;
int candidateposition = 0;
int inputposition = 0;

/* solution is the highest hierarchy: */
int solution = 0;

/* plan is the super-atom of the found plan */
int plan = 0;
int valueplan = 0;

int foundatom = 0;
int lengthofoutput = 0;

/* loop counters */
int h = 0;
int i = 0;
int j = 0;
int k = 0;
int l = 0;
int m = 0;
int n = 0;

/* "bubble up" - that is, make harder to forget by
shifting to a front position - recently used atoms. */
  
int bubbleatom = 0;
int bubbleleft = 0;
int bubbleright = 0;
int bubblevalue = 0;
int bubbleposition = 0;

/* diagnostic display of what triangles are being considered: */
int maxshow = 0;
int hypotrue = 0;

/* prevent OpenCL segfault on "releasing" stuff without running */
int neverrun = 1;

/* END OF LOGICAL TRIANGULATION VARIABLES */




/* BEGIN OPENCL */
extern int output_device_info(cl_device_id );

const char *KernelSourceConjCoh = "\n" \
"__kernel void conjcoh(                   \n" \
"   __global int* katoms,                    \n" \
"   __global int* kconjugate,                    \n" \
"   __global int* kcohabitant,                    \n" \
"   __global int* katomeffects,                    \n" \
"   __global int* katomeffectssum,                    \n" \
"   __global int* kconjugateeffects,                    \n" \
"   __global int* kcohabitanteffects,                    \n" \
"   const int atoms_0_leftatom,             \n" \
"   const int atoms_0_rightatom,             \n" \
"   const int atoms_0_atomvalue,             \n" \
"   const int ktrilimit,             \n" \
"   const int katomcount)             \n" \
"{          \n" \
" #define lowestatom 10001    \n" \
" #define atomcount 3000000    \n" \
" #define leninput 60    \n" \
" #define atomnumber 0    \n" \
" #define leftatom 1    \n" \
" #define rightatom 2    \n" \
" #define atomvalue 3    \n" \
" #define positioninatoms 4    \n" \
" #define vicvalue -1000    \n" \
" #define maxvic -30000    \n" \
" #define maxana 30000    \n" \
"   int i = get_global_id(0);             \n" \
"   int j; // = get_global_id(1); // never got that to work better             \n" \
" barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);                               \n" \
"   if (i < ktrilimit) {                        \n" \
"                                                                 \n" \
"                                                                 \n" \
"                                                                 \n" \
"           if ((i > 0) &&                                                     \n" \
"                 (atoms_0_leftatom == katoms[i*4 + leftatom])    &&           \n" \
"                 (atoms_0_rightatom != katoms[i*4 + rightatom]) &&            \n" \
"                 (atoms_0_leftatom != 0)) {                                   \n" \
"                                                                              \n" \
"               kconjugate[i*5 + atomnumber]      = katoms[i*4 + atomnumber];  \n" \
"               kconjugate[i*5 + leftatom]        = katoms[i*4 + leftatom];    \n" \
"               kconjugate[i*5 + rightatom]       = katoms[i*4 + rightatom];   \n" \
"               kconjugate[i*5 + atomvalue]       = katoms[i*4 + atomvalue];   \n" \
"               kconjugate[i*5 + positioninatoms] = i;                         \n" \
"                                                                              \n" \
"               kcohabitant[i*5 + leftatom]  = atoms_0_rightatom;              \n" \
"               kcohabitant[i*5 + rightatom] = katoms[i*4 + rightatom];        \n" \
"                                                                              \n" \
"               kcohabitant[i*5 + atomnumber] = 0;                             \n" \
"               kcohabitant[i*5 + atomvalue] = 0;                              \n" \
"               kcohabitant[i*5 + positioninatoms] = 0;                        \n" \
"                                                                              \n" \
"             } else                                                           \n" \
"             if ((i > 0) &&                                                   \n" \
"                 (atoms_0_rightatom == katoms[i*4 + leftatom])   &&           \n" \
"                 (atoms_0_leftatom != katoms[i*4 + rightatom]) &&             \n" \
"                 (atoms_0_rightatom != 0)) {                                  \n" \
"                                                                              \n" \
"               kconjugate[i*5 + atomnumber]      = katoms[i*4 + atomnumber];  \n" \
"               kconjugate[i*5 + leftatom]        = katoms[i*4 + leftatom];    \n" \
"               kconjugate[i*5 + rightatom]       = katoms[i*4 + rightatom];   \n" \
"               kconjugate[i*5 + atomvalue]       = katoms[i*4 + atomvalue];   \n" \
"               kconjugate[i*5 + positioninatoms] = i;                         \n" \
"                                                                              \n" \
"               kcohabitant[i*5 + leftatom]  = atoms_0_leftatom;               \n" \
"               kcohabitant[i*5 + rightatom] = katoms[i*4 + rightatom];        \n" \
"                                                                              \n" \
"               kcohabitant[i*5 + atomnumber] = 0;                             \n" \
"               kcohabitant[i*5 + atomvalue] = 0;                              \n" \
"               kcohabitant[i*5 + positioninatoms] = 0;                        \n" \
"                                                                              \n" \
"             } else                                                           \n" \
"             if ((i > 0) &&                                                   \n" \
"                 (atoms_0_leftatom != katoms[i*4 + leftatom])    &&           \n" \
"                 (atoms_0_rightatom == katoms[i*4 + rightatom]) &&            \n" \
"                 (atoms_0_rightatom != 0)) {                                  \n" \
"                                                                              \n" \
"               kconjugate[i*5 + atomnumber]      = katoms[i*4 + atomnumber];  \n" \
"               kconjugate[i*5 + leftatom]        = katoms[i*4 + leftatom];    \n" \
"               kconjugate[i*5 + rightatom]       = katoms[i*4 + rightatom];   \n" \
"               kconjugate[i*5 + atomvalue]       = katoms[i*4 + atomvalue];   \n" \
"               kconjugate[i*5 + positioninatoms] = i;                         \n" \
"                                                                              \n" \
"               kcohabitant[i*5 + leftatom]  = atoms_0_leftatom;               \n" \
"               kcohabitant[i*5 + rightatom] = katoms[i*4 + leftatom];         \n" \
"                                                                              \n" \
"               kcohabitant[i*5 + atomnumber] = 0;                             \n" \
"               kcohabitant[i*5 + atomvalue] = 0;                              \n" \
"               kcohabitant[i*5 + positioninatoms] = 0;                        \n" \
"                                                                              \n" \
"             } else                                                           \n" \
"             if ((i > 0) &&                                                   \n" \
"                 (atoms_0_rightatom != katoms[i*4 + leftatom])   &&           \n" \
"                 (atoms_0_leftatom == katoms[i*4 + rightatom]) &&             \n" \
"                 (atoms_0_leftatom != 0))  {                                  \n" \
"                                                                              \n" \
"               kconjugate[i*5 + atomnumber]      = katoms[i*4 + atomnumber];  \n" \
"               kconjugate[i*5 + leftatom]        = katoms[i*4 + leftatom];    \n" \
"               kconjugate[i*5 + rightatom]       = katoms[i*4 + rightatom];   \n" \
"               kconjugate[i*5 + atomvalue]       = katoms[i*4 + atomvalue];   \n" \
"               kconjugate[i*5 + positioninatoms] = i;                         \n" \
"                                                                              \n" \
"               kcohabitant[i*5 + leftatom]  = atoms_0_rightatom;              \n" \
"               kcohabitant[i*5 + rightatom] = katoms[i*4 + leftatom];         \n" \
"                                                                              \n" \
"               kcohabitant[i*5 + atomnumber] = 0;                             \n" \
"               kcohabitant[i*5 + atomvalue] = 0;                              \n" \
"               kcohabitant[i*5 + positioninatoms] = 0;                        \n" \
"                                                                              \n" \
"             } else {                                                         \n" \
"                                                                              \n" \
"               kconjugate[i*5 + atomnumber]      = 0;                         \n" \
"               kconjugate[i*5 + leftatom]        = 0;                         \n" \
"               kconjugate[i*5 + rightatom]       = 0;                         \n" \
"               kconjugate[i*5 + atomvalue]       = 0;                         \n" \
"               kconjugate[i*5 + positioninatoms] = 0;                         \n" \
"                                                                              \n" \
"               kcohabitant[i*5 + leftatom]  = 0;                              \n" \
"               kcohabitant[i*5 + rightatom] = 0;                              \n" \
"                                                                              \n" \
"               kcohabitant[i*5 + atomnumber] = 0;                             \n" \
"               kcohabitant[i*5 + atomvalue] = 0;                              \n" \
"               kcohabitant[i*5 + positioninatoms] = 0;                        \n" \
"                                                                              \n" \
"             }                                                                \n" \
"                                                                 \n" \
"                                                                 \n" \
"                                                                 \n" \
"    }                                                            \n" \
"                                                                 \n" \
" barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);                               \n" \
"                                                                                    \n" \
"                                                                                    \n" \
"                                                                                    \n" \
"                                                                                    \n" \
"                                                                                    \n" \
" if (i < ktrilimit) {                                                           \n" \
"                                                                                    \n" \
"           /* Find /cohabitant/ atoms - do NOT eliminate doubletts. */              \n" \
"                                                                                    \n" \
"           /* That is, here you e.g. have the original connection */                \n" \
"           /* A-B, you found a conjugate e.g. C-A, and now you are  */              \n" \
"           /* looking to find B=C (again, without regard to vic or  */              \n" \
"           /* ana). */                                                              \n" \
"                                                                                    \n" \
"                                                                                    \n" \
"                                                                                    \n" \
"             /* the triangle must not contain a /same side/ as the  */              \n" \
"             /*  given original side, or else you get repeated atoms */             \n" \
"             /*  in the list of all atoms: */                                       \n" \
"                                                                                    \n" \
"             if ((((kcohabitant[i*5 + leftatom] == atoms_0_leftatom) &&             \n" \
"                (kcohabitant[i*5 + rightatom] == atoms_0_rightatom)) ||             \n" \
"                ((kcohabitant[i*5 + rightatom] == atoms_0_leftatom) &&              \n" \
"                (kcohabitant[i*5 + leftatom] == atoms_0_rightatom))) ||             \n" \
"                (((kconjugate[i*5 + leftatom] == atoms_0_leftatom) &&               \n" \
"                (kconjugate[i*5 + rightatom] == atoms_0_rightatom)) ||              \n" \
"                ((kconjugate[i*5 + rightatom] == atoms_0_leftatom) &&               \n" \
"                (kconjugate[i*5 + leftatom] == atoms_0_rightatom)))) {              \n" \
"                                                                                    \n" \
"               kcohabitant[i*5 + atomnumber]      = 0;                              \n" \
"               kcohabitant[i*5 + leftatom]        = 0;                              \n" \
"               kcohabitant[i*5 + rightatom]       = 0;                              \n" \
"               kcohabitant[i*5 + atomvalue]       = 0;                              \n" \
"               kcohabitant[i*5 + positioninatoms] = 0;                              \n" \
"                                                                                    \n" \
"               kconjugate[i*5 + atomnumber]       = 0;                              \n" \
"               kconjugate[i*5 + leftatom]         = 0;                              \n" \
"               kconjugate[i*5 + rightatom]        = 0;                              \n" \
"               kconjugate[i*5 + atomvalue]        = 0;                              \n" \
"               kconjugate[i*5 + positioninatoms]  = 0;                              \n" \
"             }                                                                      \n" \
"                                                                                    \n" \
" }                                                                                  \n" \
"                                                                                    \n" \
" barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);                               \n" \
"                                                                                    \n" \
" if (i < ktrilimit) {                                                           \n" \
"                                                                                    \n" \
"             /* not i = 0, because 0 is the original atom */                        \n" \
"             /* why does it work to get rid of this loop? */                        \n" \
"             for (j = 1; j < katomcount; j++) {                                   \n" \
"               if ((((kcohabitant[i*5 + leftatom]  == katoms[j*4 + leftatom]) &&    \n" \
"                     (kcohabitant[i*5 + rightatom] == katoms[j*4 + rightatom])) ||  \n" \
"                                                                                    \n" \
"                    ((kcohabitant[i*5 + rightatom] == katoms[j*4 + leftatom]) &&    \n" \
"                     (kcohabitant[i*5 + leftatom]  == katoms[j*4 + rightatom]))) && \n" \
"                                                                                    \n" \
"                     (!((kcohabitant[i*5 + rightatom] == 0) &&                      \n" \
"                        (kcohabitant[i*5 + leftatom]  == 0)))) {                    \n" \
"                                                                                    \n" \
"                 kcohabitant[i*5 + atomnumber]      = katoms[j*4 + atomnumber];     \n" \
"                 kcohabitant[i*5 + leftatom]        = katoms[j*4 + leftatom];       \n" \
"                 kcohabitant[i*5 + rightatom]       = katoms[j*4 + rightatom];      \n" \
"                 kcohabitant[i*5 + atomvalue]       = katoms[j*4 + atomvalue];      \n" \
"                 kcohabitant[i*5 + positioninatoms] = j;                            \n" \
"                                                                                    \n" \
"                 break;                                                             \n" \
"                                                                                    \n" \
"               }                                                                    \n" \
"             }                                                                      \n" \
"                                                                                    \n" \
" }                                                                                  \n" \
"                                                                                    \n" \
"                                                                                    \n" \
" barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);                               \n" \
"                                                                                    \n" \
"   /* Forget about killing doubletts, do that with halving (shift-right or *0.5) */ \n" \
"                                                                                    \n" \
"                                                                                    \n" \
"                                                                                    \n" \
"                                                                                    \n" \
"                                                                                    \n" \
"                                                                                    \n" \
"                                                                                    \n" \
"                                                                                    \n" \
"                                                                                    \n" \
"                                                                                    \n" \
"                                                                                                              \n" \
"                                                                                                              \n" \
"                                                                                                              \n" \
" if (i < ktrilimit) {                                                                                     \n" \
"                                                                                                              \n" \
"     kcohabitanteffects[i] = 0;                                                                               \n" \
"     kconjugateeffects[i] = 0;                                                                                \n" \
"     katomeffects[i] = 0;                                                                                     \n" \
"                                                                                                              \n" \
"                                                                                                              \n" \
" }                                                                                                            \n" \
"                                                                                                              \n" \
" barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);                                                         \n" \
"                                                                                                              \n" \
" if (i < ktrilimit) {                                                                                     \n" \
"                                                                                                              \n" \
" /* TRIANGULATE */                                                                                            \n" \
"                                                                                                              \n" \
" /* That is, you HAVE the atoms, now ADJUST THEIR VALUES.                                                     \n" \
" After having adjusted the values, you still need to TRANSFER                                                 \n" \
" the adjustments to the main knowledge array. */                                                              \n" \
"                                                                                                              \n" \
"   if ((!((kconjugate[i*5 + atomnumber] == 0) && (kcohabitant[i*5 + atomnumber] == 0))) &&                    \n" \
"       (!((kconjugate[i*5 + leftatom] == 0) && (kconjugate[i*5 + rightatom] == 0))) &&                        \n" \
"       (!((kcohabitant[i*5 + leftatom] == 0) && (kcohabitant[i*5 + rightatom] == 0)))) {                      \n" \
"                                                                                                              \n" \
"     if (((kconjugate[i*5 + atomvalue] >= 0) && (kcohabitant[i*5 + atomvalue] >= 0)) ||                       \n" \
"         ((kconjugate[i*5 + atomvalue] < 0) && (kcohabitant[i*5 + atomvalue] < 0))) {                         \n" \
"                                                                                                              \n" \
"       katomeffects[i] = (int) sqrt((float) (kconjugate[i*5 + atomvalue] * kcohabitant[i*5 + atomvalue]) / 4);    \n" \
"       if (katomeffects[i] > maxana) {                                                                        \n" \
"         katomeffects[i] = maxana;                                                                            \n" \
"       }                                                                                                      \n" \
"                                                                                                              \n" \
"     } else {                                                                                                 \n" \
"                                                                                                              \n" \
"       katomeffects[i] = (int) -1 * sqrt((float) (kconjugate[i*5 + atomvalue] * kcohabitant[i*5 + atomvalue]) / (-4)); \n" \
"       if (katomeffects[i] < maxvic) {                                                                        \n" \
"         katomeffects[i] = maxvic;                                                                            \n" \
"       }                                                                                                      \n" \
"                                                                                                              \n" \
"     }                                                                                                        \n" \
"                                                                                                              \n" \
"     if (((atoms_0_atomvalue >= 0) && (kconjugate[i*5 + atomvalue] >= 0)) ||                                  \n" \
"         ((atoms_0_atomvalue < 0) && (kconjugate[i*5 + atomvalue] < 0))) {                                    \n" \
"                                                                                                              \n" \
"       kcohabitanteffects[i] = (int) sqrt((float) (atoms_0_atomvalue * kconjugate[i*5 + atomvalue]) / 4);         \n" \
"       if (kcohabitanteffects[i] > maxana) {                                                                  \n" \
"         kcohabitanteffects[i] = maxana;                                                                      \n" \
"       }                                                                                                      \n" \
"                                                                                                              \n" \
"     } else {                                                                                                 \n" \
"                                                                                                              \n" \
"       kcohabitanteffects[i] = (int) -1 * sqrt((float) (atoms_0_atomvalue * kconjugate[i*5 + atomvalue]) / (-4)); \n" \
"       if (kcohabitanteffects[i] < maxvic) {                                                                  \n" \
"         kcohabitanteffects[i] = maxvic;                                                                      \n" \
"       }                                                                                                      \n" \
"                                                                                                              \n" \
"     }                                                                                                        \n" \
"                                                                                                              \n" \
"     if (((atoms_0_atomvalue >= 0) && (kcohabitant[i*5 + atomvalue] >= 0)) ||                                 \n" \
"         ((atoms_0_atomvalue < 0) && (kcohabitant[i*5 + atomvalue] < 0))) {                                   \n" \
"                                                                                                              \n" \
"       kconjugateeffects[i] = (int) sqrt((float) (atoms_0_atomvalue * kcohabitant[i*5 + atomvalue]) / 4);         \n" \
"       if (kconjugateeffects[i] > maxana) {                                                                   \n" \
"         kconjugateeffects[i] = maxana;                                                                       \n" \
"       }                                                                                                      \n" \
"                                                                                                              \n" \
"     } else {                                                                                                 \n" \
"                                                                                                              \n" \
"       kconjugateeffects[i] = (int) -1 * sqrt((float) (atoms_0_atomvalue * kcohabitant[i*5 + atomvalue]) / (-4)); \n" \
"       if (kconjugateeffects[i] < maxvic) {                                                                   \n" \
"         kconjugateeffects[i] = maxvic;                                                                       \n" \
"       }                                                                                                      \n" \
"                                                                                                              \n" \
"     }                                                                                                        \n" \
"                                                                                                              \n" \
"   } else {                                                                                                   \n" \
"                                                                                                              \n" \
"     kcohabitanteffects[i] = 0;                                                                               \n" \
"     kconjugateeffects[i] = 0;                                                                                \n" \
"     katomeffects[i] = 0;                                                                                     \n" \
"                                                                                                              \n" \
"   }                                                                                                          \n" \
" }                                                                                                            \n" \
"                                                                                                              \n" \
" barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);                                                         \n" \
"                                                                                                              \n" \
"                                                                                                              \n" \
"                                                                                                              \n" \
"                                                                                                              \n" \
"                                                                                                              \n" \
"                                                                                                              \n" \
"}          \n" \
"\n";


void checkError(cl_int err, const char *operation) {
  if (err != CL_SUCCESS) {
    fprintf(stderr, "Error during operation '%s': %d\n", operation, err);
    exit(1);
  }
}


/* END OPENCL */





int main(void) {

/* BEGIN OPENCL */

    cl_int          err;               // error code returned from OpenCL calls

    size_t dataSizeConjCoh = sizeof(int) * absolutetrilimit * 5;
    size_t dataSizeAtoms = sizeof(int) * atomcount * 4;
    size_t dataSizeEffects = sizeof(int) * absolutetrilimit;

    size_t global;                  // global domain size

    cl_device_id     device_id;     // compute device id
    cl_context       context;       // compute context
    cl_command_queue commands;      // compute command queue
    cl_program       program;       // compute program
    cl_kernel        ko_conjcoh;       // compute kernel

    cl_mem d_katoms;                     // device memory used for the atoms
    cl_mem d_kconjugate;                     // device memory used for the conjugates
    cl_mem d_kcohabitant;                     // device memory used for the conjugates
    cl_mem d_katomeffects;                     // device memory for triangulation effects
    cl_mem d_katomeffectssum;
    cl_mem d_kconjugateeffects;
    cl_mem d_kcohabitanteffects;

    // THIS SHOULD BE MOVED TO THE DELCLARATIONS; HERE, IT SHOULD ONLY BE POPULATED.
    int d_atoms_0_leftatom = 0;
    int d_atoms_0_rightatom = 0;
    int d_atoms_0_atomvalue = 0;
    const int d_ktrilimit = absolutetrilimit; 
    const int d_katomcount = atomcount; // Just experimenting here; in reality, omit that & use atomcount or trilimit.


/* END OPENCL */



int voidret;

int zerofy;
int runthroughinput;
int charstep;
int historyelement;
int loadword;

int cycleimportance;
int deletethisword;
int eachword;
int wordsequal;
int foundposition;
int eachnumber;

int int2bin;
int bin2int;

int chargearray;

int wordtoelement;
int shiftinplayer;

int terminationcounter;
int nextletter;
int escapeflag;

int testoutput;

char testletter;
int wordcounter;
int lettercounter;
int imptrac;

int eachelement;
int readpattern;

int shiftpatterns;
int shiftelements;

int dehash;

int bubbledownzero;
int foundzero;

int previousmatch;
int matchcount;
int testword;
int testmany;
int matchposition;

int resultstrength;
int loadhistory;
int endofinput;
int findendofinput;
int shifthistory;
int cmpword;
int cmpmem;

int chargehistory;
int recordedstrength;
int recordedmatch;
int currentpattern;
int resultmatch;
int chargevector;

int wordindex;
int findaction;
int referenceholder;
int continueflag;


/* test */
/* load from file - here, generate experimentally */

/*
for (i = 0; i < atomcount; i++) {
  atoms[i][0] = lowestatom + i;
  for (j = 1; j < 4; j++) {
    atoms[i][j] = 0;
  }
  

//  printf("%d %d %d %d\n", atoms[i][0], atoms[i][1], atoms[i][2], atoms[i][3]);

  
}

*/

for (i = 0; i < leninput; i++) {
/* ACTIVATE BELOW AS SOON AS YOU PROGRAM READING IN REAL INPUT! */
      inputarray[i] = 0;
  copyinputarray[i] = 0;
  extensioninput[i] = 0;
     outputarray[i] = 0;
      wordnumber[i] = 0;
}
/* THIS IS ONLY FOR STARTUP - NEVER ZERO OUT THE INPUT DURING OPERATION!
   IT WILL LIKELY ALWAYS BE A "FULL" WINDOW!


/* END OF INITIAL SETUP FROM LOGICAL TRIANGULATION */


/* Originally, interaction should have been over files.
   That has been revised to an interactive session.
   That is why some of the file handling pointers below
   have been commented out - but left in, in case I want to
   return to the earlier model. */

/* READ HISTORY ARRAY FROM FILE */

FILE *worddata;

FILE *importancedata;

FILE *historydata;

FILE *atomdata;

/* Charge the numeric history data. */

historydata = fopen("historydata.txt","r");
for (chargehistory = 0; chargehistory < MAXHISTORY; chargehistory++) {
voidret =  fscanf(historydata, "%d", &lasthistory[chargehistory]);
}

fclose(historydata);

/* BEGIN OF CHATNUMBERS INITIALIZATION */

/* END OF CHATNUMBERS INITIALIZATION AND BEGIN OF I/O INITIALIZATION */

/* Empty the inputwords array: */

for (zerofy = 0; zerofy < INPUTLENGTH; zerofy++) {
  inputwords[zerofy] = 0;
}

/* How to initialize a test string and print it: */
/* 
inputwords[0] = 0;
strncat(inputwords, "is this funny so very so much ", INPUTLENGTH - 1);
*/
/* strncat(inputwords, "isthisfunny so very so much ", INPUTLENGTH - 1); */
/* printf("%s\n", inputwords); /* */

/* Use nextword to isolate the next word from the input. */
/* Empty the next word: */
runthroughinput = 0;
for (zerofy = 0; zerofy < LETTERSPERWORD; zerofy++) {
  nextword[zerofy] = 0;
}
charstep = 0;

/* Empty the history: */
for (historyelement = 0; historyelement < MAXHISTORY ; historyelement++) {
  for (zerofy = 0; zerofy < LETTERSPERWORD; zerofy++) {
    readwords[historyelement][zerofy] = 0;
  }
}


/* READ IN ATOMS ARRAY */
atomdata = fopen("atomdata.txt", "r");
for (i = 0; i < atomcount; i++) {
voidret =  fscanf(atomdata, "%d", &atoms[i][atomnumber]);
voidret =  fscanf(atomdata, "%d", &atoms[i][leftatom]);
voidret =  fscanf(atomdata, "%d", &atoms[i][rightatom]);
voidret =  fscanf(atomdata, "%d", &atoms[i][atomvalue]);
}
fclose(atomdata);


/* READ IN IMPORTANCE ARRAY */

/* Read in importance status of each word: */

importancedata = fopen("importancedata.txt", "r");
for (imptrac = 0; imptrac < WORDSKNOWN; imptrac++) {
voidret =  fscanf(importancedata, "%d", &importantwords[imptrac]);
}
fclose(importancedata);


/* READ IN KNOWN WORDS */

wordcounter = 0;
lettercounter = 0;
worddata = fopen("worddata.txt","r");

/* Firstly, take care there is no residual memory trash in the knownwords array: */
for (eachword = 0; eachword < WORDSKNOWN; eachword++) {
  for (zerofy = 0; zerofy < LETTERSPERWORD; zerofy++) {
    knownwords[eachword][zerofy] = 0;
  }
}

/* Initialise each word of the knownwords array to an "X" that can be overwritten - so it is not empty: */
knownwords[0][0] = 'X';
knownwords[0][1] = '0';

/* Now, basically, load the known words: */

while (!feof(worddata)) {
voidret =  fscanf(worddata,"%c", &testletter);

/* Test what letter has been read next - compare to below for what it looks like when accepted: */
/*
printf("read: %c\n", testletter);
*/

  if (testletter == '\r') {
    continue;
  }

  if (testletter == '\n') {
  /* then terminate the current word and read the next one */

    knownwords[wordcounter][lettercounter] = 0;

    wordcounter++;
    lettercounter = 0;

    /* Initialize each word to something pointless, but so it is not empty: */
    if (wordcounter < WORDSKNOWN) {
      knownwords[wordcounter][0] = 'X';
      knownwords[wordcounter][1] = '0';
    } else {
      break;
    }

    continue;

  }

  /* Write letters only to the maximum letters per word, do not allow overflows;
     it is LETTERSPERWORD - 1 and not LETTERSPERWORD due to the trailing 0. */
  if (lettercounter < LETTERSPERWORD - 1) {
    knownwords[wordcounter][lettercounter] = testletter;

/* Show what the read character actually looks like for the system: */
/*
printf("read character: %c\n", knownwords[wordcounter][lettercounter]);
*/
  }
  lettercounter++;
}

fclose(worddata);

/* Check out which words you have read: */
/*
printf("\nRESULTS:\n");
printf("1.%s.1\n", knownwords[0]);
printf("2.%s.2\n", knownwords[1]);
printf("3.%s.3\n", knownwords[2]);
printf("4.%s.4\n", knownwords[3]);
*/

printf("\nSYSTEM OPERATIONAL - PRESS ENTER WITHOUT INPUT TO TERMINATE.\n");

/* GREAT GENERAL LOOP BEGINS */
/* This is the loop inside which the interaction happens - the input is read,
   then it is translated into numbers, then it is translated into elements, then
   the neural net is being retrained word by word, i.e.  */
while (1) {
  /* When correcting eventual bugs in importantwords and knownwords,
     note that importantwords' ARRAY POSITION CONTENTS (not ARRAY POSITIONS)
     are just pointers to the knownwords. */

  /* (Re-)initialize counters before actually reading the input in the loop below. */
  wordcounter = 0;
  lettercounter = 0;

  /* Clear the auxiliary array for reading single words from eventual memory residuals. */
  runthroughinput = 0;
  for (zerofy = 0; zerofy < LETTERSPERWORD; zerofy++) {
    nextword[zerofy] = 0;
  }
  charstep = 0;

  /* Clear the array in which the read words will be stored. */
  for (historyelement = 0; historyelement < MAXHISTORY ; historyelement++) {
    for (zerofy = 0; zerofy < LETTERSPERWORD; zerofy++) {
      readwords[historyelement][zerofy] = 0;
    }
  }

  /* READ IN USER INPUT, TERMINATING ON THE HASH SIGN */

  for (zerofy = 0; zerofy < INPUTLENGTH; zerofy++) {
    inputwords[zerofy] = 0;
  }


  printf(" HUMAN: ");

  /* Now, read the letters the user writes one by one.
     The point is, while parsing, you anyway have to handle them one by one,
     so some "scan at once" is not giving you any advantage. */

  for (lettercounter = 0; lettercounter < INPUTLENGTH; lettercounter++) {

    /* Here you get the next character from the input: */
    testletter = getchar();

/* printf("testletter = %c\n", testletter); */

    /* ignore '\r', you only care about '\n' */
    if (testletter == '\r') {
      continue;
    }

    /* turn tabulator to space */
    if (testletter == '\t') {
      testletter = ' ';
    }

    /* ignore hash signs in the input */
    if (testletter == '#') {
      testletter = '%';
    }

    /* A newline terminates input or exits interaction if it is the first character read.
       Newlines are substituted for the hash-sign in internal handling. */
    if (testletter == '\n') {
      if (lettercounter == 0) {
        printf("EMPTY INPUT - INTERACTION TERMINATED.\n");
        goto exitnow;
      }
      inputwords[lettercounter] = '#';
      inputwords[lettercounter + 1] = 0;
      break;
    }

    /* Force termination if it otherwise appears impossible to end the input with '#'0. */
    /* Take the last entire word of input that still could be written out. */
    if (lettercounter < INPUTLENGTH - 2) {
      inputwords[lettercounter] = testletter;
    } else {
      for (lettercounter = INPUTLENGTH - 2; lettercounter >= 0; lettercounter--) {
        if (inputwords[lettercounter] == ' ') {
          break;
        }
      }

      inputwords[lettercounter + 1] = '#';
      inputwords[lettercounter + 2] = 0;

      /* flush input */
      while ((testletter = getchar()) != '\n' && testletter != EOF){};

      break;
    }
  }

/*
printf("read input: %s\n", inputwords);
*/

  /* PARSE USER INPUT */
  /* Capitalize here all lowercase letters. I did not choose the ASCII-values
     in an eccentric idea that the target might not use ASCII, such as OSes from
     the DEC era or from IBM. */
  /* Certain non-alphabetic characters are treated as own words - see the
     default case. All else is ignored. */
  /* The basic parsing idea is that words are extended while the next character is a letter. */

  historyelement = 0;
  while ((inputwords[runthroughinput] != 0) && (historyelement < MAXHISTORY)) {

    switch(inputwords[runthroughinput]) {
      case 'a' :
        if (charstep < LETTERSPERWORD - 1) {
          nextword[charstep] = 'A';
          charstep++;
        }
        break;

      case 'b' :
        if (charstep < LETTERSPERWORD - 1) {
          nextword[charstep] = 'B';
          charstep++;
        }
        break;

      case 'c' :
        if (charstep < LETTERSPERWORD - 1) {
          nextword[charstep] = 'C';
          charstep++;
        }
        break;

      case 'd' :
        if (charstep < LETTERSPERWORD - 1) {
          nextword[charstep] = 'D';
          charstep++;
        }
        break;

      case 'e' :
        if (charstep < LETTERSPERWORD - 1) {
          nextword[charstep] = 'E';
          charstep++;
        }
        break;

      case 'f' :
        if (charstep < LETTERSPERWORD - 1) {
          nextword[charstep] = 'F';
          charstep++;
        }
        break;

      case 'g' :
        if (charstep < LETTERSPERWORD - 1) {
          nextword[charstep] = 'G';
          charstep++;
        }
        break;

      case 'h' :
        if (charstep < LETTERSPERWORD - 1) {
          nextword[charstep] = 'H';
          charstep++;
        }
        break;

      case 'i' :
        if (charstep < LETTERSPERWORD - 1) {
          nextword[charstep] = 'I';
          charstep++;
        }
        break;

      case 'j' :
        if (charstep < LETTERSPERWORD - 1) {
          nextword[charstep] = 'J';
          charstep++;
        }
        break;

      case 'k' :
        if (charstep < LETTERSPERWORD - 1) {
          nextword[charstep] = 'K';
          charstep++;
        }
        break;

      case 'l' :
        if (charstep < LETTERSPERWORD - 1) {
          nextword[charstep] = 'L';
          charstep++;
        }
        break;

      case 'm' :
        if (charstep < LETTERSPERWORD - 1) {
          nextword[charstep] = 'M';
          charstep++;
        }
        break;

      case 'n' :
        if (charstep < LETTERSPERWORD - 1) {
          nextword[charstep] = 'N';
          charstep++;
        }
        break;

      case 'o' :
        if (charstep < LETTERSPERWORD - 1) {
          nextword[charstep] = 'O';
          charstep++;
        }
        break;

      case 'p' :
        if (charstep < LETTERSPERWORD - 1) {
          nextword[charstep] = 'P';
          charstep++;
        }
        break;

      case 'q' :
        if (charstep < LETTERSPERWORD - 1) {
          nextword[charstep] = 'Q';
          charstep++;
        }
        break;

      case 'r' :
        if (charstep < LETTERSPERWORD - 1) {
          nextword[charstep] = 'R';
          charstep++;
        }
        break;

      case 's' :
        if (charstep < LETTERSPERWORD - 1) {
          nextword[charstep] = 'S';
          charstep++;
        }
        break;

      case 't' :
        if (charstep < LETTERSPERWORD - 1) {
          nextword[charstep] = 'T';
          charstep++;
        }
        break;

      case 'u' :
        if (charstep < LETTERSPERWORD - 1) {
          nextword[charstep] = 'U';
          charstep++;
        }
        break;

      case 'v' :
        if (charstep < LETTERSPERWORD - 1) {
          nextword[charstep] = 'V';
          charstep++;
        }
        break;

      case 'w' :
        if (charstep < LETTERSPERWORD - 1) {
          nextword[charstep] = 'W';
          charstep++;
        }
        break;

      case 'x' :
        if (charstep < LETTERSPERWORD - 1) {
          nextword[charstep] = 'X';
          charstep++;
        }
        break;

      case 'y' :
        if (charstep < LETTERSPERWORD - 1) {
          nextword[charstep] = 'Y';
          charstep++;
        }
        break;

      case 'z' :
        if (charstep < LETTERSPERWORD - 1) {
          nextword[charstep] = 'Z';
          charstep++;
        }
        break;

      case 'A' :
        if (charstep < LETTERSPERWORD - 1) {
          nextword[charstep] = 'A';
          charstep++;
        }
        break;

      case 'B' :
        if (charstep < LETTERSPERWORD - 1) {
          nextword[charstep] = 'B';
          charstep++;
        }
        break;

      case 'C' :
        if (charstep < LETTERSPERWORD - 1) {
          nextword[charstep] = 'C';
          charstep++;
        }
        break;

      case 'D' :
        if (charstep < LETTERSPERWORD - 1) {
          nextword[charstep] = 'D';
          charstep++;
        }
        break;

      case 'E' :
        if (charstep < LETTERSPERWORD - 1) {
          nextword[charstep] = 'E';
          charstep++;
        }
        break;

      case 'F' :
        if (charstep < LETTERSPERWORD - 1) {
          nextword[charstep] = 'F';
          charstep++;
        }
        break;

      case 'G' :
        if (charstep < LETTERSPERWORD - 1) {
          nextword[charstep] = 'G';
          charstep++;
        }
        break;

      case 'H' :
        if (charstep < LETTERSPERWORD - 1) {
          nextword[charstep] = 'H';
          charstep++;
        }
        break;

      case 'I' :
        if (charstep < LETTERSPERWORD - 1) {
          nextword[charstep] = 'I';
          charstep++;
        }
        break;

      case 'J' :
        if (charstep < LETTERSPERWORD - 1) {
          nextword[charstep] = 'J';
          charstep++;
        }
        break;

      case 'K' :
        if (charstep < LETTERSPERWORD - 1) {
          nextword[charstep] = 'K';
          charstep++;
        }
        break;

      case 'L' :
        if (charstep < LETTERSPERWORD - 1) {
          nextword[charstep] = 'L';
          charstep++;
        }
        break;

      case 'M' :
        if (charstep < LETTERSPERWORD - 1) {
          nextword[charstep] = 'M';
          charstep++;
        }
        break;

      case 'N' :
        if (charstep < LETTERSPERWORD - 1) {
          nextword[charstep] = 'N';
          charstep++;
        }
        break;

      case 'O' :
        if (charstep < LETTERSPERWORD - 1) {
          nextword[charstep] = 'O';
          charstep++;
        }
        break;

      case 'P' :
        if (charstep < LETTERSPERWORD - 1) {
          nextword[charstep] = 'P';
          charstep++;
        }
        break;

      case 'Q' :
        if (charstep < LETTERSPERWORD - 1) {
          nextword[charstep] = 'Q';
          charstep++;
        }
        break;

      case 'R' :
        if (charstep < LETTERSPERWORD - 1) {
          nextword[charstep] = 'R';
          charstep++;
        }
        break;

      case 'S' :
        if (charstep < LETTERSPERWORD - 1) {
          nextword[charstep] = 'S';
          charstep++;
        }
        break;

      case 'T' :
        if (charstep < LETTERSPERWORD - 1) {
          nextword[charstep] = 'T';
          charstep++;
        }
        break;

      case 'U' :
        if (charstep < LETTERSPERWORD - 1) {
          nextword[charstep] = 'U';
          charstep++;
        }
        break;

      case 'V' :
        if (charstep < LETTERSPERWORD - 1) {
          nextword[charstep] = 'V';
          charstep++;
        }
        break;

      case 'W' :
        if (charstep < LETTERSPERWORD - 1) {
          nextword[charstep] = 'W';
          charstep++;
        }
        break;

      case 'X' :
        if (charstep < LETTERSPERWORD - 1) {
          nextword[charstep] = 'X';
          charstep++;
        }
        break;

      case 'Y' :
        if (charstep < LETTERSPERWORD - 1) {
          nextword[charstep] = 'Y';
          charstep++;
        }
        break;

      case 'Z' :
        if (charstep < LETTERSPERWORD - 1) {
          nextword[charstep] = 'Z';
          charstep++;
        }
        break;

      case '0' :
        if (charstep < LETTERSPERWORD - 1) {
          nextword[charstep] = '0';
          charstep++;
        }
        break;

      case '1' :
        if (charstep < LETTERSPERWORD - 1) {
          nextword[charstep] = '1';
          charstep++;
        }
        break;

      case '2' :
        if (charstep < LETTERSPERWORD - 1) {
          nextword[charstep] = '2';
          charstep++;
        }
        break;

      case '3' :
        if (charstep < LETTERSPERWORD - 1) {
          nextword[charstep] = '3';
          charstep++;
        }
        break;

      case '4' :
        if (charstep < LETTERSPERWORD - 1) {
          nextword[charstep] = '4';
          charstep++;
        }
        break;

      case '5' :
        if (charstep < LETTERSPERWORD - 1) {
          nextword[charstep] = '5';
          charstep++;
        }
        break;

      case '6' :
        if (charstep < LETTERSPERWORD - 1) {
          nextword[charstep] = '6';
          charstep++;
        }
        break;

      case '7' :
        if (charstep < LETTERSPERWORD - 1) {
          nextword[charstep] = '7';
          charstep++;
        }
        break;

      case '8' :
        if (charstep < LETTERSPERWORD - 1) {
          nextword[charstep] = '8';
          charstep++;
        }
        break;

      case '9' :
        if (charstep < LETTERSPERWORD - 1) {
          nextword[charstep] = '9';
          charstep++;
        }
        break;

      case '-' :
        if (charstep < LETTERSPERWORD - 1) {
          nextword[charstep] = '-';
          charstep++;
        }
        break;

      case '+' :
        if (charstep < LETTERSPERWORD - 1) {
          nextword[charstep] = '+';
          charstep++;
        }
        break;

      /* insertion: singlequote */
      case '\'' :
        if (charstep < LETTERSPERWORD - 1) {
          nextword[charstep] = '\'';
          charstep++;
        }
        break;

      case '_' :
        if (charstep < LETTERSPERWORD - 1) {
          nextword[charstep] = '+';
          charstep++;
        }
        break;

      default :
        if (nextword[0] != 0) {
          /* printf("JUST READ: %s\n", nextword); /* */
          for (loadword = 0; loadword < (LETTERSPERWORD - 1); loadword++) {
            readwords[historyelement][loadword] = nextword[loadword];
          }
          historyelement++;
        }

        charstep = 0;
        for (zerofy = 0; zerofy < LETTERSPERWORD; zerofy++) {
          nextword[zerofy] = 0;
        }

        if (historyelement < (MAXHISTORY - 1)) {
          switch(inputwords[runthroughinput]) {
            case '.' :
              readwords[historyelement][0] = '.';

              historyelement++;
              break;

            case ',' :
              readwords[historyelement][0] = ',';

              historyelement++;
              break;

            case ':' :
              readwords[historyelement][0] = ':';

              historyelement++;
              break;

            case ';' :
              readwords[historyelement][0] = ';';

              historyelement++;
              break;

            case '<' :
              readwords[historyelement][0] = '<';

              historyelement++;
              break;

            case '>' :
              readwords[historyelement][0] = '>';

              historyelement++;
              break;

            case '|' :
              readwords[historyelement][0] = '|';

              historyelement++;
              break;

            case '\\' :
              readwords[historyelement][0] = '\\';

              historyelement++;
              break;

            case '/' :
              readwords[historyelement][0] = '/';

              historyelement++;
              break;

            case '\"' :
              readwords[historyelement][0] = '\"';

              historyelement++;
              break;

            /*
            case '\'' :
              readwords[historyelement][0] = '\'';

              historyelement++;
              break;
            */

            case '&' :
              readwords[historyelement][0] = '&';

              historyelement++;
              break;

            case '#' :
              readwords[historyelement][0] = '#';

              historyelement++;
              break;

            case '*' :
              readwords[historyelement][0] = '*';

              historyelement++;
              break;

            case '~' :
              readwords[historyelement][0] = '~';

              historyelement++;
              break;

            case '(' :
              readwords[historyelement][0] = '(';

              historyelement++;
              break;

            case ')' :
              readwords[historyelement][0] = ')';

              historyelement++;
              break;

            case '[' :
              readwords[historyelement][0] = '[';

              historyelement++;
              break;

            case ']' :
              readwords[historyelement][0] = ']';

              historyelement++;
              break;

            case '{' :
              readwords[historyelement][0] = '{';

              historyelement++;
              break;

            case '}' :
              readwords[historyelement][0] = '}';

              historyelement++;
              break;

            case '$' :
              readwords[historyelement][0] = '$';

              historyelement++;
              break;

            case '%' :
              readwords[historyelement][0] = '%';

              historyelement++;
              break;

            case '^' :
              readwords[historyelement][0] = '^';

              historyelement++;
              break;

            case '=' :
              readwords[historyelement][0] = '=';

              historyelement++;
              break;

            case '?' :
              readwords[historyelement][0] = '?';

              historyelement++;
              break;

            case '!' :
              readwords[historyelement][0] = '!';

              historyelement++;
              break;

            case '@' :
              readwords[historyelement][0] = '@';

              historyelement++;
              break;
          }
        }
      }

    runthroughinput++;
  }

  /* CHARGE THE WORD NUMBERS - TURN THE READ WORDS INTO NUMBERS */

  /* Cleanse the numeric history of read input words from memory residuals */
  for (zerofy = 0; zerofy < MAXHISTORY; zerofy++) {
    wordnumber[zerofy] = 0;
  }

  /* The array readwords contains the characters which have been read from the user
     in the above loop. The words shall be transferred to the array wordnumber[]
     which contains the history of read words from the input, but expressed in
     numbers - this is just a step away from neural network operations on the words. */

  /* That is, after having read in the characters, you now have them as words in
     "readwords", but you are still unsure whether the are known, so now, one by one,
     you check them and you give them numbers. */

  eachnumber = 0;
  for (historyelement = 0; historyelement < MAXHISTORY; historyelement++) {

    /* Terminate evaluation of characters if there is no further character read. */
    if (readwords[historyelement][0] == 0) {
      continue;
    }

    /* Check out whether a read word is equal to a known word. Notice the !(strncmp... -
       That is actually necessary, a "matching hit" is otherwise interpreted by the "if" as false! */
    wordsequal = 0;
    for (eachword = 0; eachword < WORDSKNOWN; eachword++) {
      if (!(strncmp(readwords[historyelement], knownwords[eachword], LETTERSPERWORD))) {
        wordsequal = 1;

/* Show what was matched. See this in relation to the "miss" - i.e. no hit - below. */
/*
printf("Hit: %s eachword: %d \n", readwords[historyelement], eachword);
*/
        break;
      }
    }

    /* The if-clause below is very long. It handles the "importance" of words.
       A word that has been just observed becomes more "important" - it is shifted
       towards the end of the importantwords array. The point is, the most UNimportant
       word - the one in position importantwords[1] - will be FORGOTTEN, if a new word
       has been learned and when space must be created for implanting said new word. */

    /* wordsequal == 1 means that the word has been known and does not need to be created */

    if (wordsequal == 1) {

      /* eachword from above contains where wordsknown has been found.
         This should now be made the "most important word" as it has been
         most recently observed. For that, find out WHERE in importantwords
         this value of eachword is contained, in order to push it to the
         end of importantwords, i.e. to the "most important" position
         importantwords[WORDSKNOWN - 1]. This is like a partial cshift
         in Fortran 90. */
      foundposition = 0;

      /* Changed this into a countdown-loop - important words are usually
         more often used and therefore more towards the end of the array. */
      for (cycleimportance = WORDSKNOWN - 1; cycleimportance > 0 ; cycleimportance--) {
        if (importantwords[cycleimportance] == eachword) {
          foundposition = cycleimportance;
          break;
        }
/*
printf("compared: %d and %d\n", importantwords[cycleimportance], eachword);
*/
      }

/*
printf("found eachword in: %d\n", foundposition);
*/

/* The "XXXX importance"-printfs below shall show you how the importance-array looks like after each step of the cycling. */ 
/*
printf("PRIOR importance: %d %d %d %d\n", importantwords[0], importantwords[1], importantwords[2], importantwords[3]);
*/

      /* Shifting is only necessary, of course, if the word is not ALREADY the most important word. */
      if (foundposition < (WORDSKNOWN - 1)) {
        for (cycleimportance = foundposition; cycleimportance < (WORDSKNOWN - 1); cycleimportance++) {
          importantwords[cycleimportance] = importantwords[cycleimportance + 1];
        }

/*
printf("INTERMEDIATE importance: %d %d %d %d\n", importantwords[0], importantwords[1], importantwords[2], importantwords[3]);
*/
        importantwords[WORDSKNOWN - 1] = eachword;
      } /* */

/*
printf("POSTERIOR importance: %d %d %d %d\n", importantwords[0], importantwords[1], importantwords[2], importantwords[3]);
*/
      /* The position of importantwords[0] is NOT shifted around.
         It contains the hash-sign # that terminates input. Due to its special meaning,
         it should remain exactly where it is, on the "most unimportant" place. */
      /* Make sure the zero-position is not shifted around like the rest of the array. */
      /* Firstly, find out where the desired zero has been shifted. */
      if (importantwords[0] != 0) {
        for (bubbledownzero = 1; bubbledownzero < WORDSKNOWN; bubbledownzero++) {
          if (importantwords[bubbledownzero] == 0) {
            foundzero = bubbledownzero;
            break;
          }
        }

        /* Secondly, re-shift the array so the zero is again in position zero. */
        for (bubbledownzero = foundzero; bubbledownzero > 0; bubbledownzero--) {
          importantwords[bubbledownzero] = importantwords[bubbledownzero - 1];
        }
        importantwords[0] = 0;
      }

/*
printf("POST-POSTERIOR importance: %d %d %d %d\n", importantwords[0], importantwords[1], importantwords[2], importantwords[3]);
*/

      /* HERE you charge the wordnumber array - i.e. here you establish the history of words. */
      wordnumber[eachnumber] = eachword;
/* printf("found wordnumber: %d\n", wordnumber[eachnumber]); */
      eachnumber++;


    /* else, if the word is not known, you must forget a word - the oldest unused one - and create a word */

    } else {

/*
printf("NO HIT: %s %d %d %d %d %d %d %d\n", readwords[historyelement], readwords[historyelement][0], readwords[historyelement][1], readwords[historyelement][2], readwords[historyelement][3], readwords[historyelement][4], readwords[historyelement][5], readwords[historyelement][6]);
*/

/*
printf("NEW PRIOR importance: %d %d %d %d\n", importantwords[0], importantwords[1], importantwords[2], importantwords[3]);
*/
      /* The number of the oldest unused word is now re-attributed to the newly learned word */
      deletethisword = importantwords[1];
      for (cycleimportance = 1; cycleimportance < (WORDSKNOWN - 1); cycleimportance++) {
        importantwords[cycleimportance] = importantwords[cycleimportance + 1];
      }

/*
printf("NEW INTERMEDIATE importance: %d %d %d %d\n", importantwords[0], importantwords[1], importantwords[2], importantwords[3]);
*/

      importantwords[WORDSKNOWN - 1] = deletethisword;

/*
printf("NEW POSTERIOR importance: %d %d %d %d\n", importantwords[0], importantwords[1], importantwords[2], importantwords[3]);
*/

      /* Make sure the zero-position is not shifted around like the rest of the array. */
      /* importantwords[0] should reliably always contain the hash-sign. */
      /* Firstly, find out where the desired zero has been shifted. */
      if (importantwords[0] != 0) {
        for (bubbledownzero = 1; bubbledownzero < WORDSKNOWN; bubbledownzero++) {
          if (importantwords[bubbledownzero] == 0) {
            foundzero = bubbledownzero;
            break;
          }
        }

        /* Secondly, re-shift the array so the zero is again in position zero. */
        for (bubbledownzero = foundzero; bubbledownzero > 0; bubbledownzero--) {
          importantwords[bubbledownzero] = importantwords[bubbledownzero - 1];
        }
        importantwords[0] = 0;
      }

/*
printf("NEW POST-POSTERIOR importance: %d %d %d %d\n", importantwords[0], importantwords[1], importantwords[2], importantwords[3]);
*/

      /* Implant the new word into the knownwords array, overwriting the previous word -
         as you take it with trailing zeroes and all, you have no residual memory trash issues. */
      for (loadword = 0; loadword < LETTERSPERWORD; loadword++) {
        knownwords[deletethisword][loadword] = readwords[historyelement][loadword];
      }

      /* Finally, record the newly re-created word into the numeric history of the words perceived. */
      wordnumber[eachnumber] = deletethisword;
/* printf("invented wordnumber: %d\n", wordnumber[eachnumber]); */
      eachnumber++;

    }

  }

  /* UPPER PART */

  /* Now, you have parsed the input words into the array named wordnumber[]. */


  /* So now, transfer that matches' output into the wordnumber array - instead */
  /* of input, it will now be holding output. */



/* ____START____OF____LOGICAL____TRIANGULATION____ */

    /* BEGIN SETUP OPENCL */

    // Set up platform and GPU device

    cl_uint numPlatforms;

    // Find number of platforms
    err = clGetPlatformIDs(0, NULL, &numPlatforms);
    checkError(err, "Finding platforms");
    if (numPlatforms == 0)
    {
        printf("Found 0 platforms!\n");
        return EXIT_FAILURE;
    }

    // Get all platforms
    cl_platform_id Platform[numPlatforms];
    err = clGetPlatformIDs(numPlatforms, Platform, NULL);
    checkError(err, "Getting platforms");

    // Secure a GPU
    for (i = 0; i < numPlatforms; i++)
    {
        err = clGetDeviceIDs(Platform[i], DEVICE, 1, &device_id, NULL);
        if (err == CL_SUCCESS)
        {
            break;
        }
    }

    if (device_id == NULL)
        checkError(err, "Getting device");

    err = clGetDeviceIDs(Platform[i], DEVICE, 1, &device_id, NULL); // a shim
    checkError(err, "Outputting device info");

    // Create a compute context
    context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
    checkError(err, "Creating context");

    // Create a command queue
    commands = clCreateCommandQueue(context, device_id, 0, &err);
    checkError(err, "Creating command queue");

    // Create the compute program from the source buffer
    program = clCreateProgramWithSource(context, 1, (const char **) & KernelSourceConjCoh, NULL, &err);
    checkError(err, "Creating program");

    // Build the program
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        size_t len;
        char buffer[2048];

        printf("Error: Failed to build program executable!\n\n");
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        printf("%s\n", buffer);
        return EXIT_FAILURE;
    }

    // Create the compute kernel from the program
    ko_conjcoh = clCreateKernel(program, "conjcoh", &err);
    checkError(err, "Creating kernel");

    /* END SETUP OPENCL */


    /* test */
    /*
    printf("\n\nFIRST ATOMS:\n");
    for (i = 0; i < atomcount; i++) {
      printf("%d %d %d %d\n", atoms[i][0], atoms[i][1], atoms[i][2], atoms[i][3]);
    }
    */
    /* end test */

  /* main section */

  /* COMMUNICATION "WITH THE OUTSIDE WORLD" HAPPENS
     OVER THE wordnumber[]-ARRAY. IT TRANSFERS WORDS TO
     NUMBERS AND NUMBERS TO WORDS. */

  /* TRANSFER THE NUMBERS TO THE INPUT AS HISTORY */

  lengthofoutput = 0;
  /* Find out how long the human input is that is not zero,
     so we know what to transfer: */
  for (i = leninput - 1; i >= 0; i--) {
    if (wordnumber[i] != 0) {
      lengthofoutput = i + 1;
      break;
    }
  }

  /* 
  printf("length of user input was: %d\n", lengthofoutput);
  */

  /* shift left the input array elements */
  for (i = 0; i < leninput - lengthofoutput; i++)  {
    inputarray[i] = inputarray[i + lengthofoutput];
  }

  /* implant the output array elements at the end of the shifted input */
  j = 0;
  for (i = leninput - lengthofoutput; i < leninput; i++)  {
    inputarray[i] = wordnumber[j];

    /* test */
    /*
    printf("i: %d, inputarray[i]: %d\n", i, inputarray[i]);
    */
    /* end test */

    j++;
  }
  lengthofoutput = 0;

  k = 0;
  for (chargevector = 0; chargevector < MAXHISTORY; chargevector++) {

    /* Test is the input entirely zeroes: */
    if (wordnumber[chargevector] != 0) {
      k = 1;
    }

    wordnumber[chargevector] = 0;

  }

  if (k == 0) {
    printf("Input all empty, terminating.\n");
    /* exit(0); */
    goto exitnow;
  } else {
    neverrun = 0;
  }

  /*
  printf("INPUT ARRAY: ");
  for (i = 0; i < leninput; i++) {
    printf(" %d", inputarray[i]);
  }
  printf("\n");
  */

  /* COPY THE INPUT FOR RECONSIDERATION: */
  
  /* Create a copy of the input array in order to be
  able to "reconsider" the input later so as to find
  further (and indirect) conclusions which were not
  obvious the first time input was hierarchised. */
  
  for (i = 0; i < leninput; i++) {
    copyinputarray[i] = inputarray[i];
  
    /* the extensioninput will be re-charged in the end after planning
       - the charge here will be used only  */
    extensioninput[i] = inputarray[i];
  }
  
  
  /* These two loops drive re-consideration.
  Set them to happen only once if you do not
  wish to see any reconsideration done, and if
  you just want one straight logical triangulation
  run. */
  
  for (h = 0; h < sledge; h++) {
    for (l = 0; l < reconsiderations; l++) {

    /*
    printf("sledge /daydreaming/: %d, reconsiderations: %d\n", h, l);
    */

    /*
    printf("INPUT ARRAY: ");
    for (i = 0; i < leninput; i++) {
      printf(" %d", inputarray[i]);
    }
    printf("\n");
    */

      /* restore input for re-evaluation */
      if ((l == 0) && (h < sledge - 1)) {
        /* periodically re-set the input to the stage being examined */
        for (i = 0; i < leninput; i++) {
          inputarray[i] = copyinputarray[i];
          extensioninput[i] = inputarray[i];
        }
      } else
      if ((l > 0) && (h < sledge - 1)) {
        /* otherwise, follow the extension of the input by the plan */
        for (i = 0; i < leninput; i++) {
          inputarray[i] = extensioninput[i];
        }
      } else {
        /* re-set to the original input for the actual answer. */
        for (i = 0; i < leninput; i++) {
          inputarray[i] = copyinputarray[i];
        }
      }
      

      /*
      printf("INPUT ARRAY: ");
      for (i = 0; i < leninput; i++) {
        printf(" %d", inputarray[i]);
      }
      printf("\n");
      */
      
      /* Now actually hierarchise input:
      each step here will be followed by a triangulation
      triggered using the "selected hierarchisation pair".
      If no pair is known, a pair is simply supposed, that
      is, if your input is A B C D, and you have no idea
      where to hierarchise (as e.g. all combinations are
      unknown), the system will select the last pair for
      hierarchisation (that is, it will do A B C-D 0). */
      
      
      /* Check for the atom pair with the most negative
      atom-value - hierarchise on that atom pair. If none
      is known, generate a hypothesis about the connection. */
      
      for (n = 0; n < leninput - 1; n++) {
      
        /* run through each possible input pair to determine which to "hierarchise" */
        candidateatom = 0;
        candidatevalue = maxana + 1;
        candidateposition = 0;
        inputposition = 0;
        
        for (i = 0; i < leninput; i++) {
        
          for (j = 0; j < atomcount; j++) {
        
            /* if any atom would match the proposed atom pair, count it as a candidate */
            if ((((atoms[j][leftatom] == inputarray[i])     && (atoms[j][rightatom] == inputarray[i + 1])) || 
                 ((atoms[j][leftatom] == inputarray[i + 1]) && (atoms[j][rightatom] == inputarray[i])))    &&
               (!((atoms[j][leftatom] == 0) && (atoms[j][rightatom] == 0))) &&
               (atoms[j][atomvalue] < candidatevalue)) {
/* WAS: */
/*             (atoms[j][atomvalue] <= candidatevalue)) {} -- brace match */
        
              candidateatom     = atoms[j][atomnumber];
              candidatevalue    = atoms[j][atomvalue];
              candidateleft     = inputarray[i];
              candidateright    = inputarray[i + 1];
              candidateposition = j;
              inputposition     = i;
        
            }
        
          }
        
        }
        

        /*
        printf("INPUT ARRAY: ");
        for (i = 0; i < leninput; i++) {
          printf(" %d", inputarray[i]);
        }
        printf("\n");
        */

        /* if no atom was found, create a "hypothesis" - the tail pair */
        /* assume no hypothesis, but correct that if there was a hypothesis - used for diagnostics: */
        hypotrue = 0;

        if (candidatevalue > maxana) {
        /* hypothesis creation */

        hypotrue = 1;

          /* for the next hypothesis creation: */
          atomnumbertracer = atoms[atomcount - 1][atomnumber];
        
#pragma omp barrier

          /* create space for a new atom by "forgetting" an old atom */
          #pragma omp parallel for
          for (k = atomcount - 1; k > 0 ; k--) {
            atomscopy[k][atomnumber] = atoms[k - 1][atomnumber];
            atomscopy[k][leftatom]   = atoms[k - 1][leftatom];
            atomscopy[k][rightatom]  = atoms[k - 1][rightatom];
            atomscopy[k][atomvalue]  = atoms[k - 1][atomvalue];
          }

#pragma omp barrier

          #pragma omp parallel for
          for (k = atomcount - 1; k > 0 ; k--) {
            atoms[k][atomnumber] = atomscopy[k][atomnumber];
            atoms[k][leftatom]   = atomscopy[k][leftatom];
            atoms[k][rightatom]  = atomscopy[k][rightatom];
            atoms[k][atomvalue]  = atomscopy[k][atomvalue];
          }
        
#pragma omp barrier

          inputposition = 0;
          /* find the inputposition */
          for (k = leninput - 1; k > 0; k--) {
            if (inputarray[k] != 0) {
              inputposition = k - 1;
              break;
            }
          }

          /* Eject if the hypothesis should be based on any zero atom. */
          /* Erase this if a modification of the loop variable is disallowed, like in old FORTRAN! */

          if ((inputarray[inputposition] == 0) || (inputarray[inputposition + 1] == 0)) {
            n = leninput - 1;
          }
          
          atoms[0][atomnumber] = atomnumbertracer;
          atoms[0][leftatom]   = inputarray[inputposition];
          atoms[0][rightatom]  = inputarray[inputposition + 1];
          atoms[0][atomvalue]  = vicvalue;
        
          candidateatom = atoms[0][atomnumber];
          candidatevalue = atoms[0][atomvalue];
          candidateleft = atoms[0][leftatom];
          candidateright = atoms[0][rightatom];
          candidateposition = 0;
        
        } else {
        
          /* create space for "bubbling up" a known atom */
          for (k = candidateposition; k > 0 ; k--) {
            atoms[k][atomnumber] = atoms[k - 1][atomnumber];
            atoms[k][leftatom]   = atoms[k - 1][leftatom];
            atoms[k][rightatom]  = atoms[k - 1][rightatom];
            atoms[k][atomvalue]  = atoms[k - 1][atomvalue];
          }
        
          atoms[0][atomnumber] = candidateatom;
          atoms[0][leftatom]   = candidateleft;
          atoms[0][rightatom]  = candidateright;
          atoms[0][atomvalue]  = candidatevalue + vicvalue;
        
          if (atoms[0][atomvalue] < maxvic) {
            atoms[0][atomvalue] = maxvic;
          }
        
        }
        
        
        /* "Hierarchise" the input - that is, replace the pair
        with the super-atom and "shorten" the array; so make
        A B C D into A B-C D 0, assuming you selected B-C as
        the most negative connection: */
        
        
        for (k = inputposition; k < leninput - 1; k++) {
          inputarray[k] = inputarray[k + 1];
        }
        
        inputarray[inputposition] = candidateatom;
        inputarray[leninput - 1] = 0;
      
      
        
        /* test */
        /*
        printf("\n\nBEFORE TRIANGULATION:\n");
        for (i = 0; i < atomcount; i = i + 2) {
          printf("%d %d %d %d\t\t%d %d %d %d\n", atoms[i][0], atoms[i][1], atoms[i][2], atoms[i][3],
                 atoms[i + 1][0], atoms[i + 1][1], atoms[i + 1][2], atoms[i + 1][3]);
        }
        */
        /* end test */
        
        /* DO TRIANGULATION */
#pragma omp barrier
        if ((atoms[0][leftatom] != 0) && (atoms[0][rightatom] != 0)) {
        
          /* Given a connection from input, e.g. A-B, then say, triangulation is
          possible with an atom C, so that e.g. + A-B=C-A. How do you perform
          triangulation? You HAVE A-B, SEEK B=C and SEEK OR GUESS C-A. C-A can
          even be formed as hypothesis, if it is not known. That is: A-B MUST be
          given, B=C MUST be found, C-A CAN be found in the knowledge base. To
          differentiate between these connections, I am calling A-B the "original"
          connection, B=C the "CONJUGATE" (it is conjunctioned to A-B, that is, it
          must have exactly ONE ATOM IN COMMON with A-B) and C-A is the
          "COHABITANT" side (it is co-existing with A-B and C-A by having with
          each of these connections exactly one atom in common, and not that atom
          that these two sides have in common. - One effect happens: If really all
          three A-B, B=C and C-A are known, then you can form, starting from A-B,
          TWO EQUIVALENT triangles: + A-B=C-A as well as + C-A-B=C. These I call
          doubletts. There are three ways to handle doubletts: you can either
          compute both and accept that; or you can try a mathematical adjustment;
          or you can try to find the doubletts and "kill" them, that is, not
          perform triangulation twice on what is materially the same triangle.
          Herein, the third variant is chosen. */
          
          /* STAGES OF TRIANGULATION: */
          /* FIND CONJUGATE */
          /* FORM A HYPOTHESIS */
          /* SEEK COHABITANT */
          /* FOUND: TRIANGULATE */
          /* NOT FOUND: HYPOTHESIS */
          /* MIND THE CORRECT ATOM NUMBERS! */
          /* DO THIS FOR UP TO N TRIANGULATIONS: triangulationslimit; if none found - break */
          
          /* zero out (clean up) cohabitants and conjugates */

          #pragma omp parallel for
          for (i = 0; i < maxtriangles; i++) {
            atomeffects[i] = 0;
            conjugateeffects[i] = 0;
            cohabitanteffects[i] = 0;
          
            for (j = 0; j < 5; j++) {
              conjugate[i][j] = 0;
              cohabitant[i][j] = 0;
            }
          
            for (k = 0; k < 4; k++) {
              hypotheses[i][k] = 0;
              scrubhypotheses[i][k] = 0;
            }
          
          }

#pragma omp barrier

/* BEGIN OPENCL */


/*
printf("\nSTAGE X\n");
*/

    /* Create the arrays in device memory */
    d_katoms  = clCreateBuffer(context,  CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,  dataSizeAtoms, atoms, &err);
    checkError(err, "Creating buffer d_katoms");

    d_kconjugate  = clCreateBuffer(context,  CL_MEM_READ_WRITE, dataSizeConjCoh, NULL, &err);
    checkError(err, "Creating buffer d_kconjugate");

    d_kcohabitant  = clCreateBuffer(context,  CL_MEM_READ_WRITE, dataSizeConjCoh, NULL, &err);
    checkError(err, "Creating buffer d_kcohabitant");

    /* Create the effects in device memory - not all used so far, but prepare for future use. */
    d_katomeffects  = clCreateBuffer(context,  CL_MEM_READ_WRITE, dataSizeEffects, NULL, &err);
    checkError(err, "Creating buffer d_katomeffects");

    d_katomeffectssum  = clCreateBuffer(context,  CL_MEM_READ_WRITE, dataSizeEffects, NULL, &err);
    checkError(err, "Creating buffer d_katomeffectssum");

    d_kconjugateeffects  = clCreateBuffer(context,  CL_MEM_READ_WRITE, dataSizeEffects, NULL, &err);
    checkError(err, "Creating buffer d_kconjugateeffects");

    d_kcohabitanteffects  = clCreateBuffer(context,  CL_MEM_READ_WRITE, dataSizeEffects, NULL, &err);
    checkError(err, "Creating buffer d_kcohabitanteffects");

    /* setup basic atom from where to begin triangulation: */
    d_atoms_0_leftatom = atoms[0][leftatom];
    d_atoms_0_rightatom = atoms[0][rightatom];
    d_atoms_0_atomvalue = atoms[0][atomvalue];

/*
printf("\nSTAGE Y\n");
*/

    // Enqueue kernel
    // Set the arguments to our compute kernel
    err  = clSetKernelArg(ko_conjcoh, 0, sizeof(cl_mem), &d_katoms);
    err |= clSetKernelArg(ko_conjcoh, 1, sizeof(cl_mem), &d_kconjugate);
    err |= clSetKernelArg(ko_conjcoh, 2, sizeof(cl_mem), &d_kcohabitant);

    err |= clSetKernelArg(ko_conjcoh, 3, sizeof(cl_mem), &d_katomeffects);
    err |= clSetKernelArg(ko_conjcoh, 4, sizeof(cl_mem), &d_katomeffectssum);
    err |= clSetKernelArg(ko_conjcoh, 5, sizeof(cl_mem), &d_kconjugateeffects);
    err |= clSetKernelArg(ko_conjcoh, 6, sizeof(cl_mem), &d_kcohabitanteffects);

    err |= clSetKernelArg(ko_conjcoh, 7, sizeof(int), &d_atoms_0_leftatom);
    err |= clSetKernelArg(ko_conjcoh, 8, sizeof(int), &d_atoms_0_rightatom);
    err |= clSetKernelArg(ko_conjcoh, 9, sizeof(int), &d_atoms_0_atomvalue);
    err |= clSetKernelArg(ko_conjcoh, 10, sizeof(int), &d_ktrilimit);
    err |= clSetKernelArg(ko_conjcoh, 11, sizeof(int), &d_katomcount);
    checkError(err, "Setting kernel arguments");

    // Execute the kernel over the entire range of our 1d input data set
    // letting the OpenCL runtime choose the work-group size
    global = d_ktrilimit; // d_katomcount; // normally use atomcount... or trilimit
    err = clEnqueueNDRangeKernel(commands, ko_conjcoh, 1, NULL, &global, NULL, 0, NULL, NULL);
    checkError(err, "Enqueueing kernel 1st time");

   // Read back the result from the compute device
    err = clEnqueueReadBuffer(commands, d_kconjugate, CL_TRUE, 0, sizeof(int) * d_ktrilimit * 5, conjugate, 0, NULL, NULL );
    checkError(err, "Reading back d_kconjugate");

   // Read back the result from the compute device
    err = clEnqueueReadBuffer(commands, d_kcohabitant, CL_TRUE, 0, sizeof(int) * d_ktrilimit * 5, cohabitant, 0, NULL, NULL );
    checkError(err, "Reading back d_kcohabitant");

   // Read back the result from the compute device
    err = clEnqueueReadBuffer(commands, d_katomeffects, CL_TRUE, 0, sizeof(int) * d_ktrilimit, atomeffects, 0, NULL, NULL );
    checkError(err, "Reading back d_katomeffects");

   // Read back the result from the compute device
    err = clEnqueueReadBuffer(commands, d_katomeffectssum, CL_TRUE, 0, sizeof(int) * d_ktrilimit, atomeffectssum, 0, NULL, NULL );
    checkError(err, "Reading back d_katomeffectssum");

   // Read back the result from the compute device
    err = clEnqueueReadBuffer(commands, d_kconjugateeffects, CL_TRUE, 0, sizeof(int) * d_ktrilimit, conjugateeffects, 0, NULL, NULL );
    checkError(err, "Reading back d_kconjugateeffects");

   // Read back the result from the compute device
    err = clEnqueueReadBuffer(commands, d_kcohabitanteffects, CL_TRUE, 0, sizeof(int) * d_ktrilimit, cohabitanteffects, 0, NULL, NULL );
    checkError(err, "Reading back d_kcohabitanteffects");

    // Test the results

/*
printf("\nSTAGE Z\n");
printf("translated atoms:\n");
for (i = 0; i < 10; i++) {
  printf("conjugate: %d, left: %d, right: %d, value: %d, position: %d, conjeffect: %d\n", conjugate[i][0], conjugate[i][1], conjugate[i][2], conjugate[i][3], conjugate[i][4], conjugateeffects[i]);
  printf("cohabitant: %d, left: %d, right: %d, value: %d, position: %d, coheffect: %d\n", cohabitant[i][0], cohabitant[i][1], cohabitant[i][2], cohabitant[i][3], cohabitant[i][4], cohabitanteffects[i]);
}
*/

clReleaseMemObject(d_katoms);
clReleaseMemObject(d_kconjugate);
clReleaseMemObject(d_kcohabitant);
clReleaseMemObject(d_katomeffects);
clReleaseMemObject(d_katomeffectssum);
clReleaseMemObject(d_kconjugateeffects);
clReleaseMemObject(d_kcohabitanteffects);

/* END OPENCL */

#pragma omp barrier

          
          /*
          printf("\n\nBEFORE AFFECTION:\n");
          for (i = 0; i < atomcount; i = i + 2) {
            printf("%d %d %d %d\t\t%d %d %d %d\n", atoms[i][0], atoms[i][1], atoms[i][2], atoms[i][3],
                   atoms[i + 1][0], atoms[i + 1][1], atoms[i + 1][2], atoms[i + 1][3]);
          }
          */
  
          /* AFFECT THE ORIGINAL RELATION WITH THE RESULTS OF TRIANGULATION */
          /* That is, transfer the values. */
          
          
          atomeffectssum[0] = 0;
          atomeffectsdivisor = 0;
          for (i = 0; i < maxtriangles; i++) {
          
            if ((!((conjugate[i][atomnumber] == 0) && (cohabitant[i][atomnumber] == 0))) &&
                (!((conjugate[i][leftatom] == 0) && (conjugate[i][rightatom] == 0))) &&
                (!((cohabitant[i][leftatom] == 0) && (cohabitant[i][rightatom] == 0)))) {
          
              atomeffectssum[0] = atomeffectssum[0] + atomeffects[i];
              atomeffectsdivisor++;
          
            }
          
          }
          
          if (atomeffectsdivisor > 0) {
          
            atomeffectssum[0] = atomeffectssum[0] / atomeffectsdivisor;
          
            atomeffectssum[0] = atomeffectssum[0] + atoms[0][atomvalue];
          
            if (atomeffectssum[0] < maxvic) {
          
              atomeffectssum[0] = maxvic;
          
            } else if (atomeffectssum[0] > maxana) {
          
              atomeffectssum[0] = maxana;
          
            }
          
            atoms[0][atomvalue] = atomeffectssum[0];
          
          }

#pragma omp barrier

          /* DIAGNOSTIC */

          printf("\nFOLDING INPUT ARRAY:\n");
          j = 0;
          for (i = 0; i < leninput; i++) {
            printf("%-7d ", inputarray[i]);
            if (j == 11) {
              printf("\n");
              j = 0;
            } else {
              j++;
            }
          }
          printf("\nSTATE OF EVALUATION:\n");

          /* Is the atom hypothetical or recognized? */
          if (hypotrue == 1) {
            printf("?");
          } else {
            printf("!");
          }

          if (atoms[0][atomvalue] < 0) {
            printf("   ATOM:%-7d L:%-7d R:%-7d VIC:%-+6d :: DAYDREAM:%-2d OF %-2d    ::  RECONS:%-2d OF %-2d :: HRCHY:%-3d OF %-3d :: INPOS    %-3d\n", atoms[0][atomnumber], atoms[0][leftatom], atoms[0][rightatom], atoms[0][atomvalue], h + 1, sledge, l + 1, reconsiderations, n + 1, leninput - 1, inputposition);

          } else {

            printf("   ATOM:%-7d L:%-7d R:%-7d ANA:%-+6d :: DAYDREAM:%-2d OF %-2d    ::  RECONS:%-2d OF %-2d :: HRCHY:%-3d OF %-3d :: INPOS    %-3d\n", atoms[0][atomnumber], atoms[0][leftatom], atoms[0][rightatom], atoms[0][atomvalue], h + 1, sledge, l + 1, reconsiderations, n + 1, leninput - 1, inputposition);

          }


          /* END OF DIAGNOSTIC */

          /* AFFECT EVERY CONJUGATE - THE CONJUGATES ARE ALL KNOWN! */
          #pragma omp parallel for
          for (i = 0; i < maxtriangles; i++) {
          
            /* consider using an array instead of a scalar in order to parallelize */
            atomeffectssum[i] = 0;
          
            if (!((conjugate[i][atomnumber] == 0) && (cohabitant[i][atomnumber] == 0))) {

              /* DIAGNOSTIC */

              if (i == 0) {
                maxshow = 0;
              }

              /* SHOW UP TO 20 TRIANGLES PER CONCLUSION SERIES */
              if (maxshow < 20) {
                maxshow++;


              if ((conjugate[i][3] < 0) && (cohabitant[i][3] < 0)) {

                printf(" __/CONJ:%-7d L:%-7d R:%-7d,VIC:%-+9d POS:%-7d EFCT:%-+6d\n   \\COHB:%-7d L:%-7d R:%-7d`VIC:%-+9d POS:%-7d EFCT:%-+6d",
                      conjugate[i][0], conjugate[i][1], conjugate[i][2], conjugate[i][3], conjugate[i][4], conjugateeffects[i],
                      cohabitant[i][0], cohabitant[i][1], cohabitant[i][2], cohabitant[i][3], cohabitant[i][4], cohabitanteffects[i]);


              } else
              if ((conjugate[i][3] >= 0) && (cohabitant[i][3] < 0)) {

                printf(" __/CONJ:%-7d L:%-7d R:%-7d,ANA:%-+9d POS:%-7d EFCT:%-+6d\n   \\COHB:%-7d L:%-7d R:%-7d`VIC:%-+9d POS:%-7d EFCT:%-+6d",
                      conjugate[i][0], conjugate[i][1], conjugate[i][2], conjugate[i][3], conjugate[i][4], conjugateeffects[i],
                      cohabitant[i][0], cohabitant[i][1], cohabitant[i][2], cohabitant[i][3], cohabitant[i][4], cohabitanteffects[i]);


              } else
              if ((conjugate[i][3] < 0) && (cohabitant[i][3] >= 0)) {

                printf(" __/CONJ:%-7d L:%-7d R:%-7d,VIC:%-+9d POS:%-7d EFCT:%-+6d\n   \\COHB:%-7d L:%-7d R:%-7d`ANA:%-+9d POS:%-7d EFCT:%-+6d",
                      conjugate[i][0], conjugate[i][1], conjugate[i][2], conjugate[i][3], conjugate[i][4], conjugateeffects[i],
                      cohabitant[i][0], cohabitant[i][1], cohabitant[i][2], cohabitant[i][3], cohabitant[i][4], cohabitanteffects[i]);


              } else {

                printf(" __/CONJ:%-7d L:%-7d R:%-7d,ANA:%-+9d POS:%-7d EFCT:%-+6d\n   \\COHB:%-7d L:%-7d R:%-7d`ANA:%-+9d POS:%-7d EFCT:%-+6d",
                      conjugate[i][0], conjugate[i][1], conjugate[i][2], conjugate[i][3], conjugate[i][4], conjugateeffects[i],
                      cohabitant[i][0], cohabitant[i][1], cohabitant[i][2], cohabitant[i][3], cohabitant[i][4], cohabitanteffects[i]);


              }

                if (cohabitant[i][0] != 0) {
                  printf("  [ADJUSTED] ");
                  if (((cohabitant[i][3]*conjugate[i][3] < 0) && (atoms[0][3] > 0)) ||
                      ((cohabitant[i][3]*conjugate[i][3] > 0) && (atoms[0][3] < 0))) {
                    printf("-\n");
                  } else
                  if ((cohabitant[i][3]*conjugate[i][3] == 0) || (atoms[0][3] == 0)) {
                    printf("0\n");
                  } else {
                    printf("+\n");
                  }
                } else {
                  printf("  [CONCLUDED] \n");
                }

              }
              /* END OF DIAGNOSTIC */
          
              atomeffectssum[i] = conjugateeffects[i] + atoms[conjugate[i][positioninatoms]][atomvalue];
          
              if (atomeffectssum[i] < maxvic) {
          
                atomeffectssum[i] = maxvic;
          
              } else if (atomeffectssum[i] > maxana) {
          
                atomeffectssum[i] = maxana;
          
              }
          
              atoms[conjugate[i][positioninatoms]][atomvalue] = atomeffectssum[i];
          
            }
          
          }
          
#pragma omp barrier
          
          /* AFFECT EVERY COHABITANT - THE COHABITANTS MAY BE HYPOTHETICAL! */
          
          /* AFFECT ALL KNOWN COHABITANTS, COLLECT ALL HYPOTHESES */
          hypothesescount = 0;
          /* SIMPLY NOT LOOKING LIKE A GOOD IDEA: #pragma omp parallel for */
          for (i = 0; i < maxtriangles; i++) {
          
            /* consider using an array instead of a scalar in order to parallelize */
            atomeffectssum[i] = 0;
          
            if (conjugate[i][atomnumber] != 0) {
          
              /* if the atom is actually known */
              if (cohabitant[i][atomnumber] != 0) {
          
                atomeffectssum[i] = cohabitanteffects[i] + atoms[cohabitant[i][positioninatoms]][atomvalue];
          
                if (atomeffectssum[i] < maxvic) {
          
                  atomeffectssum[i] = maxvic;
          
                } else if (atomeffectssum[i] > maxana) {
          
                  atomeffectssum[i] = maxana;
          
                }
          
                atoms[cohabitant[i][positioninatoms]][atomvalue] = atomeffectssum[i];
          
              /* otherwise, add the atom to the hypotheses */
              } else {
          
                hypotheses[hypothesescount][leftatom]  = cohabitant[i][leftatom];
                hypotheses[hypothesescount][rightatom] = cohabitant[i][rightatom];
                
                /* however, I CANNOT take the "cohabitant[i][atomvalue]" as value,
                   as that is ZERO per definitionem. */
                
                atomeffectssum[i] = cohabitanteffects[i];
          
                if (atomeffectssum[i] < maxvic) {
          
                  atomeffectssum[i] = maxvic;
          
                } else if (atomeffectssum[i] > maxana) {
          
                  atomeffectssum[i] = maxana;
          
                }
                
                hypotheses[hypothesescount][atomvalue] = atomeffectssum[i];
          
                hypothesescount++;
          
              }
          
            }
          
          }
          
#pragma omp barrier
          
          /* You now have transferred all values to all KNOWN atoms -
          but what happens when atoms are NOT KNOWN, however CONCLUDED
          by means of logical triangulation? Well - you generate hypotheses.
          For this, you will need to "forget" atoms to make space for new
          hypotheses. You "recycle" their atom numbers. That is, if X meant G=H,
          but X is "forgotten" and the new hypothesis is A-B, then X will now
          be made to mean A-B. */
          
          /* GENERATE HYPOTHESES - "FORGET" AND "RECYCLE" ATOMS AS NECESSARY */
          
          /* START OF HYPOTHESES SCRUBBING */
          
          /* test */
          /*
          printf("\n\nHYPOTHESES BEFORE SCRUBBING:\n");
          for (i = 0; i < maxtriangles; i++) {
            printf("%d %d %d %d\n", hypotheses[i][0], hypotheses[i][1], hypotheses[i][2], hypotheses[i][3]);
          }
          */
          /* end test */
          
          /* The hypotheses may contain duplicates. Remove the duplicates from a "scrub" array. */
          
          /* Transfer all hypotheses to the scrub array. */
          #pragma omp parallel for
          for (i = 0; i < hypothesescount; i++) {
            scrubhypotheses[i][atomnumber] = hypotheses[i][atomnumber];
            scrubhypotheses[i][leftatom]   = hypotheses[i][leftatom];
            scrubhypotheses[i][rightatom]  = hypotheses[i][rightatom];
            scrubhypotheses[i][atomvalue]  = hypotheses[i][atomvalue];
            hypotheses[i][atomnumber] = 0;
            hypotheses[i][leftatom] = 0;
            hypotheses[i][rightatom] = 0;
            hypotheses[i][atomvalue] = 0;
          }
          
#pragma omp barrier
          
          /* Remove multiple scrubhypotheses relating to the SAME pair: */
          scrubhypothesescount = hypothesescount;
          if (scrubhypothesescount > 0) {
            for (i = 0; i < scrubhypothesescount - 1; i++) {
  
            /*
            printf("scrubhypothesis: %d %d %d %d\n", scrubhypotheses[i][0], scrubhypotheses[i][1], scrubhypotheses[i][2], scrubhypotheses[i][3]);
            */
  
              if (!((scrubhypotheses[i][leftatom] == 0) && (scrubhypotheses[i][rightatom] == 0))) {
                /* AFFECTED RESULTS, BUT WITH BARRIERS NOW BETTER - STILL, CONSIDER TURNING OFF:  */ 
                #pragma omp parallel for
                for (j = i + 1; j < scrubhypothesescount; j++) {
                  if (((scrubhypotheses[i][leftatom] == scrubhypotheses[j][leftatom]) &&
                     (scrubhypotheses[i][rightatom] == scrubhypotheses[j][rightatom])) ||
          
                     ((scrubhypotheses[i][leftatom]  == scrubhypotheses[j][rightatom]) &&
                     (scrubhypotheses[i][rightatom] == scrubhypotheses[j][leftatom]))) {
          
                    scrubhypotheses[i][atomvalue] = scrubhypotheses[j][atomvalue] + scrubhypotheses[i][atomvalue];
          
                    if (scrubhypotheses[i][atomvalue] < maxvic) {
          
                      scrubhypotheses[i][atomvalue] = maxvic;
          
                    } else if (scrubhypotheses[i][atomvalue] > maxana) {
          
                      scrubhypotheses[i][atomvalue] = maxana;
          
                    }
          
                    scrubhypotheses[j][atomnumber] = 0;
                    scrubhypotheses[j][leftatom] = 0;
                    scrubhypotheses[j][rightatom] = 0;
                    scrubhypotheses[j][atomvalue] = 0;
                  }
                }
              }
            }
          
            j = 0;
            for (i = 0; i < scrubhypothesescount; i++) {
              if (!((scrubhypotheses[i][atomnumber] == 0) &&
                    (scrubhypotheses[i][leftatom] == 0) &&
                    (scrubhypotheses[i][rightatom] == 0))) {
                hypotheses[j][atomnumber] = scrubhypotheses[i][atomnumber];
                hypotheses[j][leftatom]   = scrubhypotheses[i][leftatom];
                hypotheses[j][rightatom]  = scrubhypotheses[i][rightatom];
                hypotheses[j][atomvalue]  = scrubhypotheses[i][atomvalue];
                j++;
                scrubhypotheses[i][atomnumber] = 0;
                scrubhypotheses[i][leftatom] = 0;
                scrubhypotheses[i][rightatom] = 0;
                scrubhypotheses[i][atomvalue] = 0;
              }
            }
          
            hypothesescount = j;
          
          }
          
          /* test */
          /*
          printf("\n\nHYPOTHESES AFTER SCRUBBING:\n");
          for (i = 0; i < maxtriangles; i++) {
            printf("%d %d %d %d\n", hypotheses[i][0], hypotheses[i][1], hypotheses[i][2], hypotheses[i][3]);
          }
          */
          /* end test */
          
          /* END OF HYPOTHESES SCRUBBING */
          
          /* Transfer atom numbers to hypotheses. */
          
          j = atomcount;
          for (i = 0; i < hypothesescount; i++) {
            j--;
            hypotheses[i][atomnumber] = atoms[j][atomnumber];
          }
          
#pragma omp barrier

          /* "Forget" final atoms. */
          #pragma omp parallel for
          for (k = atomcount - 1; k > hypothesescount - 1; k--) {
            atomscopy[k][atomnumber] = atoms[k - hypothesescount][atomnumber];
            atomscopy[k][leftatom]   = atoms[k - hypothesescount][leftatom];
            atomscopy[k][rightatom]  = atoms[k - hypothesescount][rightatom];
            atomscopy[k][atomvalue]  = atoms[k - hypothesescount][atomvalue];
          }

#pragma omp barrier
  
          #pragma omp parallel for
          for (k = atomcount - 1; k >= 0 ; k--) {
            atoms[k][atomnumber] = atomscopy[k][atomnumber];
            atoms[k][leftatom]   = atomscopy[k][leftatom];
            atoms[k][rightatom]  = atomscopy[k][rightatom];
            atoms[k][atomvalue]  = atomscopy[k][atomvalue];
          }

#pragma omp barrier
  
          /* Transfer hypotheses to atoms array as "normal" atoms. */
          #pragma omp parallel for        
          for (k = 0; k < hypothesescount; k++) {
            atoms[k][atomnumber] = hypotheses[k][atomnumber];
            atoms[k][leftatom]   = hypotheses[k][leftatom];
            atoms[k][rightatom]  = hypotheses[k][rightatom];
            atoms[k][atomvalue]  = hypotheses[k][atomvalue];
          }
          

          /* bubble up atoms */
          /* cohabitant hypotheses were pushed into the beginning anyway */
          /* so you may focus on bubbling up these atoms that are actually known */

#pragma omp barrier

          for (i = 0; i < maxtriangles; i++) {
            if (conjugate[i][atomnumber] != 0) {
            
              bubbleatom = conjugate[i][atomnumber];
          
              // find the atom in the main array:
              
              for (j = 0; j < atomcount; j++) {
              
                if (atoms[j][atomnumber] == bubbleatom) {
                  bubbleposition = j;
                  bubbleleft     = conjugate[i][leftatom];
                  bubbleright    = conjugate[i][rightatom];
                  bubblevalue    = atoms[j][atomvalue];
                  break;
                }
              }
              
              // create space for "bubbling up" a known atom

              for (k = bubbleposition; k > 0 ; k--) {
                atoms[k][atomnumber] = atoms[k - 1][atomnumber];
                atoms[k][leftatom]   = atoms[k - 1][leftatom];
                atoms[k][rightatom]  = atoms[k - 1][rightatom];
                atoms[k][atomvalue]  = atoms[k - 1][atomvalue];
              }
                
              atoms[0][atomnumber] = bubbleatom;
              atoms[0][leftatom]   = bubbleleft;
              atoms[0][rightatom]  = bubbleright;
              atoms[0][atomvalue]  = bubblevalue;
          
              bubbleatom     = 0;
              bubbleleft     = 0;
              bubbleright    = 0;
              bubblevalue    = 0;
              bubbleposition = 0;
          
              if (cohabitant[i][atomnumber] != 0) {
              
                bubbleatom = cohabitant[i][atomnumber];
            
                // find the atom in the main array:
                
                for (j = 0; j < atomcount; j++) {
                
                  if (atoms[j][atomnumber] == bubbleatom) {
                    bubbleposition = j;
                    bubbleleft     = cohabitant[i][leftatom];
                    bubbleright    = cohabitant[i][rightatom];
                    bubblevalue    = atoms[j][atomvalue];
                    break;
                  }
                }
                
                // create space for "bubbling up" a known atom 
                for (k = bubbleposition; k > 0 ; k--) {
                  atoms[k][atomnumber] = atoms[k - 1][atomnumber];
                  atoms[k][leftatom]   = atoms[k - 1][leftatom];
                  atoms[k][rightatom]  = atoms[k - 1][rightatom];
                  atoms[k][atomvalue]  = atoms[k - 1][atomvalue];
                }
                  
                atoms[0][atomnumber] = bubbleatom;
                atoms[0][leftatom]   = bubbleleft;
                atoms[0][rightatom]  = bubbleright;
                atoms[0][atomvalue]  = bubblevalue;
          
                bubbleatom     = 0;
                bubbleleft     = 0;
                bubbleright    = 0;
                bubblevalue    = 0;
                bubbleposition = 0;
              }
            }
          } /* */
        }

        /* Done. That's it with Logical Triangulation, at least in
        this iteration. All that COULD be found out following a given
        original atom - HAS been found out. */
        
      }

    /* end of reconsiderations */
    }
    
    
    
    
    
    
    /* SEEK A POTENTIAL PLAN */
    
    /* That is, if now you hierarchised the input up to "Z",
    you are now seeking the atom S that contains Z-R, so that
    your "plan" is R. */
    
    solution = 0;
    plan = 0;
    lengthofoutput = 0;
    
    /* That is worse than the worst plan
    - plans should be "vic", that is, the
    "best" plan is that plan which is most
    tightly coupled to the "present": */
    
    valueplan = maxana + 1;
    
    /* find the "solution" of the input, that is "the one" or "the last" atom of hierarchisation: */
    for (k = leninput - 1; k >= 0; k--) {
      if (inputarray[k] != 0) {
        solution = inputarray[k];
        break;
      }
    }
    
    /* TEST PLANNING: - assuming the solutions were these,
       does the system find the plan correctly: */
    /* solution = 5; */
    /* solution = 128; */
    
    
    /* test */
    /*
    printf("____INPUT ARRAY: ");
    for (i = 0; i < leninput; i++) {
      printf("%d ", inputarray[i]);
    }
    printf("\n");
    */
    /* end test */
    /*
    printf("____SOLUTION FOUND: %d\n", solution);
    */
    /* end test planning */
    
    /* in case NO input was found, there is NOTHING to say */
    if (solution == 0) {
      plan = 0;
    } else {
    
    /* your hierarchy of concepts CANNOT be HIGHER than the LENGTH of the input
       (or input window), take that into account when looking for a plan: */
      for (k = 0; k < leninput; k++) {
    
        for (i = 0; i < atomcount; i++) {
          if ((atoms[i][leftatom] == solution) && (atoms[i][atomvalue] <= valueplan)) {
            plan      = atoms[i][rightatom];
            valueplan = atoms[i][atomvalue];
    
            /* test */
            /*
            printf("solution: %d, plan for solution: %d, super-atom: %d, value: %d\n", solution, plan, atoms[i][atomnumber], valueplan);
            */
            /* end test */
    
          }
        }
    
    /* test */
    /*
    printf("____SOLUTION FOUND LATER: %d\n", solution);
    */
    /* end test */
    
        /* test */
        /*
        printf("plan___ was found: %d\n", plan);
        */
        /* end test */
    
        /* if no plan was found: */
        if (plan == 0) {
      
          /* if the solution was elementary, there really is no plan */
          if (solution < lowestatom) {
            break;
    
          /* otherwise, decompose the solution with the aim to find a plan: */
          } else {
            for (j = 0; j < atomcount; j++) {
            
              /* if the current solution is found, let us just try its right sub-atom, unless it is zero: */
              if (atoms[j][atomnumber] == solution) {
                if (atoms[j][rightatom] != 0) {
                  solution = atoms[j][rightatom];
                  break;
                } else {
                  solution = atoms[j][leftatom];
                  break;
                }
    
                /* test */
                /*
                printf("replacement solution: %d, replacement plan for solution: %d, super-atom: %d, value: %d\n",
                      solution, plan, atoms[i][atomnumber], valueplan);
                */
                /* end test */
    
              }
            }
          }
        }
      }
    }
    
    
    /* By now - you should have a plan. However, that "plan"
    will likely not be just one elementary atom, it will
    likely be some sort of super-atom. You will need to
    "decompose" it, that is, turn it into sub-atoms and
    sub-sub-atoms, etc., until you can finally output it. */

    /*
    printf("plan: %d\n", plan);
    */

    /* cleanse the output array in preparation for output */
    for (i = 0; i < leninput; i++) {
      outputarray[i] = 0;
    }
    outputarray[0] = plan;
    
    /* We really do not know how "high" the plan can be, so that is
    why we have an infinite loop to be left as required. There is
    ONE condition not well solved in the loop: what if the atom is
    recursive? That is, an atom of the form [A][A][rightatom][value].
    That will lead to infinite looping and hanging. Such an atom is
    "zeroed out" currently, though some other solution might be better. */
    
    i = 0;
    for (;;) {
    
      /* If the element of the plan is elementary, check the next plan element. */
      if ((outputarray[i] < lowestatom) || (outputarray[i] == 0)) {
    
        i++;
        
        /* If the plan is not yet fully decomposed, continue, else be done and present the plan. */
        if (i < leninput) {
          continue;
        } else {
          break;
        }
    
      /* If the element of the plan is NOT elementary, (i) shift the "tail" of the plan right,
                                                                          (ii) replace the element,
                                                                          (iii) DO NOT advance in the plan. */
      } else {
      
        /* shift the output array right */
        for (j = leninput - 1; j > i; j--) {
          outputarray[j] = outputarray[j - 1];
        }
    
        /* replace the element in the output array with its two sub-atoms */
        foundatom = 0;
        for (k = 0; k < atomcount; k++) {
          if (atoms[k][atomnumber] == outputarray[i]) {
            if ((atoms[k][atomnumber] == atoms[k][leftatom]) || (atoms[k][atomnumber] == atoms[k][rightatom])) {
              outputarray[i] = 0;
              i++;
            } else {
              foundatom = 1;
              outputarray[i]     = atoms[k][leftatom];
              outputarray[i + 1] = atoms[k][rightatom];
              break;
            }
          }
        }
    
        if (foundatom == 0) {
          outputarray[i] = 0;
          i++;
        }
    
      }
    }
    
    
    /* By now, the output array contains the system's answer to the outer world. */
    
    
    /* test */
    /*
    printf("PLAN FOR SOLUTION FOUND: ");
    for (i = 0; i < leninput; i++) {
      printf("%d ", outputarray[i]);
    }
    printf("\n");
    */
    /* end test */
    
    /* IN FUTURE: IT WILL BE BEST TO _IGNORE_ 0s IN THE OUTPUT
    AND HAVE _ANOTHER_ SIGN FOR TERMINATION!
    So far, however, I have not determined such other sign. */
    
    
    /* test */
    /*
    printf("\n\nFINAL ATOMS:\n");
    for (i = 0; i < atomcount; i++) {
      printf("%d %d %d %d\n", atoms[i][0], atoms[i][1], atoms[i][2], atoms[i][3]);
    }
    /**/
    /* end test */
    
    
    /* In hindsight: this entire section could operate on
       extensioninput instead of inputarray. */
    
#pragma omp barrier

    /* restore input for shifting */
    #pragma omp parallel for
    for (i = 0; i < leninput; i++) {
      inputarray[i] = extensioninput[i];
    }
    
    /* test */
    
    /* Mark one atom of the input so you can see it shifting: */
    /* inputarray[leninput - 1] = 1337; */
    
    /*
    for (i = 0; i < leninput; i++) {
      printf("%d ", inputarray[i]);
    }
    printf("\n");
    */
    /* end test */
    
    /* Now - you HAVE OUTPUT - but do you also observe your
    own actions? That is, "do you know what you said?" - It
    is assumed, yes, you do. That, however, means that the
    output array must also be transferred into the input
    array again, as that is the system's only venue of
    perception. */
    
    
    /* If a plan was found, transfer it to the input array;
       otherwise, the input array stays the same. */
    if (plan != 0 ) {
    
      /* Find out how long the reply is that is not zero,
         so we know what to transfer: */
      for (i = leninput - 1; i >= 0; i--) {
        if (outputarray[i] != 0) {
          lengthofoutput = i + 1;
          break;
        }
      }

      /*
      printf("length of output was: %d\n", lengthofoutput);
      */

      /* shift left the input array elements */
      for (i = 0; i < leninput - lengthofoutput; i++)  {
        inputarray[i] = inputarray[i + lengthofoutput];
      }
    
      /* implant the output array elements at the end of the shifted input */
      j = 0;
      for (i = leninput - lengthofoutput; i < leninput; i++)  {
        inputarray[i] = outputarray[j];
        j++;
      }
    
    }
    
    /* test */
    /*
    printf("Shifted input array: \n");
    for (i = 0; i < leninput; i++) {
      printf("%d ", inputarray[i]);
    }
    printf("\n");
    */
    /* end test */

#pragma omp barrier    
    /* COPY THE INPUT FOR RECONSIDERATION: */
    #pragma omp parallel for
    for (i = 0; i < leninput; i++) {
      extensioninput[i] = inputarray[i];
    }
    
#pragma omp barrier
    /* end of sledge */
  }


  /* TRANSFER TRIANGULATION RESULTS TO OUTPUT */

  for (chargevector = 0; chargevector < MAXHISTORY; chargevector++) {
    wordnumber[chargevector] = outputarray[chargevector];
  }

/* ____END____OF____LOGICAL____TRIANGULATION____ */

  /* The "upper part" of response ends here. Usually, you could now just
     continue to forward the found answer to the human as the next machine
     challenge, but in this version, the perceptions first must be replaced
     by "actions" - the part where the machine does not "convey only its own
     experiences", but undertakes actual "deeds". */

  /* END OF UPPER PART */

  /* Replace "perceptions" with appropriate "actions" instinctively. */

  for (wordindex = 0; wordindex < MAXHISTORY; wordindex++) {
    continueflag = 0;

    if (wordnumber[wordindex] == 0) {
      break;
    }

    if (!(strncmp("I", knownwords[wordnumber[wordindex]], LETTERSPERWORD))) {
      for (findaction = 0; findaction < WORDSKNOWN; findaction++) {
        if (!(strncmp("YOU", knownwords[findaction], LETTERSPERWORD))) {
          wordnumber[wordindex] = findaction;
          continueflag = 1;
          break;
        }
      }
    }

    if (continueflag == 1) {
      continue;
    }

    if (!(strncmp("ME", knownwords[wordnumber[wordindex]], LETTERSPERWORD))) {
      for (findaction = 0; findaction < WORDSKNOWN; findaction++) {
        if (!(strncmp("YOU", knownwords[findaction], LETTERSPERWORD))) {
          wordnumber[wordindex] = findaction;
          continueflag = 1;
          break;
        }
      }
    }

    if (continueflag == 1) {
      continue;
    }

    if (!(strncmp("YOU", knownwords[wordnumber[wordindex]], LETTERSPERWORD))) {
      for (findaction = 0; findaction < WORDSKNOWN; findaction++) {
        if (!(strncmp("ME", knownwords[findaction], LETTERSPERWORD))) {
          wordnumber[wordindex] = findaction;
          continueflag = 1;
          break;
        }
      }
    }

    if (continueflag == 1) {
      continue;
    }

    if (!(strncmp("MYSELF", knownwords[wordnumber[wordindex]], LETTERSPERWORD))) {
      for (findaction = 0; findaction < WORDSKNOWN; findaction++) {
        if (!(strncmp("YOURSELF", knownwords[findaction], LETTERSPERWORD))) {
          wordnumber[wordindex] = findaction;
          continueflag = 1;
          break;
        }
      }
    }

    if (continueflag == 1) {
      continue;
    }

    if (!(strncmp("YOURSELF", knownwords[wordnumber[wordindex]], LETTERSPERWORD))) {
      for (findaction = 0; findaction < WORDSKNOWN; findaction++) {
        if (!(strncmp("MYSELF", knownwords[findaction], LETTERSPERWORD))) {
          wordnumber[wordindex] = findaction;
          continueflag = 1;
          break;
        }
      }
    }

    if (continueflag == 1) {
      continue;
    }

    if (!(strncmp("MY", knownwords[wordnumber[wordindex]], LETTERSPERWORD))) {
      for (findaction = 0; findaction < WORDSKNOWN; findaction++) {
        if (!(strncmp("YOUR", knownwords[findaction], LETTERSPERWORD))) {
          wordnumber[wordindex] = findaction;
          continueflag = 1;
          break;
        }
      }
    }

    if (continueflag == 1) {
      continue;
    }

    if (!(strncmp("YOUR", knownwords[wordnumber[wordindex]], LETTERSPERWORD))) {
      for (findaction = 0; findaction < WORDSKNOWN; findaction++) {
        if (!(strncmp("MY", knownwords[findaction], LETTERSPERWORD))) {
          wordnumber[wordindex] = findaction;
          continueflag = 1;
          break;
        }
      }
    }

    if (continueflag == 1) {
      continue;
    }

    if (!(strncmp("MINE", knownwords[wordnumber[wordindex]], LETTERSPERWORD))) {
      for (findaction = 0; findaction < WORDSKNOWN; findaction++) {
        if (!(strncmp("YOURS", knownwords[findaction], LETTERSPERWORD))) {
          wordnumber[wordindex] = findaction;
          continueflag = 1;
          break;
        }
      }
    }

    if (continueflag == 1) {
      continue;
    }

    if (!(strncmp("YOURS", knownwords[wordnumber[wordindex]], LETTERSPERWORD))) {
      for (findaction = 0; findaction < WORDSKNOWN; findaction++) {
        if (!(strncmp("MINE", knownwords[findaction], LETTERSPERWORD))) {
          wordnumber[wordindex] = findaction;
          continueflag = 1;
          break;
        }
      }
    }

    if (continueflag == 1) {
      continue;
    }

    if (!(strncmp("I'M", knownwords[wordnumber[wordindex]], LETTERSPERWORD))) {
      for (findaction = 0; findaction < WORDSKNOWN; findaction++) {
        if (!(strncmp("YOU'RE", knownwords[findaction], LETTERSPERWORD))) {
          wordnumber[wordindex] = findaction;
          continueflag = 1;
          break;
        }
      }
    }

    if (continueflag == 1) {
      continue;
    }

    if (!(strncmp("YOU'RE", knownwords[wordnumber[wordindex]], LETTERSPERWORD))) {
      for (findaction = 0; findaction < WORDSKNOWN; findaction++) {
        if (!(strncmp("I'M", knownwords[findaction], LETTERSPERWORD))) {
          wordnumber[wordindex] = findaction;
          continueflag = 1;
          break;
        }
      }
    }

    if (continueflag == 1) {
      continue;
    }

    if (!(strncmp("AM", knownwords[wordnumber[wordindex]], LETTERSPERWORD))) {
      for (findaction = 0; findaction < WORDSKNOWN; findaction++) {
        if (!(strncmp("ARE", knownwords[findaction], LETTERSPERWORD))) {
          wordnumber[wordindex] = findaction;
          continueflag = 1;
          break;
        }
      }
    }

    if (continueflag == 1) {
      continue;
    }

    if (!(strncmp("ARE", knownwords[wordnumber[wordindex]], LETTERSPERWORD))) {
      for (findaction = 0; findaction < WORDSKNOWN; findaction++) {
        if (!(strncmp("AM'ARE", knownwords[findaction], LETTERSPERWORD))) {
          wordnumber[wordindex] = findaction;
          continueflag = 1;
          break;
        }
      }
    }

    if (continueflag == 1) {
      continue;
    }

    if (!(strncmp("WERE", knownwords[wordnumber[wordindex]], LETTERSPERWORD))) {
      for (findaction = 0; findaction < WORDSKNOWN; findaction++) {
        if (!(strncmp("WAS'WERE", knownwords[findaction], LETTERSPERWORD))) {
          wordnumber[wordindex] = findaction;
          continueflag = 1;
          break;
        }
      }
    }

    if (continueflag == 1) {
      continue;
    }

    if (!(strncmp("WAS", knownwords[wordnumber[wordindex]], LETTERSPERWORD))) {
      for (findaction = 0; findaction < WORDSKNOWN; findaction++) {
        if (!(strncmp("WAS'WERE", knownwords[findaction], LETTERSPERWORD))) {
          wordnumber[wordindex] = findaction;
          continueflag = 1;
          break;
        }
      }
    }
  }

  /* Here, the actions have replaced the experiences and the wordnumber array
     is has now been mutated from a "perception" into an "action" array. */

  /* TRANSFER WORDNUMBERS TO TEXTREPLY */
  /* I.e. that is the part where the numeric preparations are over and the
     user shall receive text. */

  /* initialise the reply to the user */
  for (chargearray = 0; chargearray < INPUTLENGTH; chargearray++) {
    textreply[chargearray] = 0;
  }

  terminationcounter = 0;
  escapeflag = 0;
  /* Termination safety switch for maximum output length - testoutput itself is not used. */
  /* This loop's task is the creation of output by triggering the net repeatedly, word for word. */
  for (testoutput = 0; testoutput < INPUTLENGTH; testoutput++) {

    bin2int = wordnumber[testoutput];

    /* terminate output upon HASH SIGN which should be represented by binary zero: */
    if (bin2int == 0) {
      break;
    }

    /* HERE, YOU NOW HAVE THE FIRST RESULT OF TRIGGERING AS DECIMAL IN bin2int. */
    /* NOW CREATE THE USER RESPONSE FROM knownwords[bin2int][:] */

    nextletter = 0;

    /* This is to make certain sentence signs stick to their previous word */
    /* for highlight reasons: ([{ */
    if ((knownwords[bin2int][0] == '.') || (knownwords[bin2int][0] == ',') || (knownwords[bin2int][0] == ':') || (knownwords[bin2int][0] == ';') || (knownwords[bin2int][0] == '?') || (knownwords[bin2int][0] == '!') || (knownwords[bin2int][0] == ')') || (knownwords[bin2int][0] == ']') || (knownwords[bin2int][0] == '}')) {
      terminationcounter--;
    }

    /* Basically, the output is extended as long as the word has not
       ended and as long as the text reply is not too long. */
    while((knownwords[bin2int][nextletter] != 0) && (escapeflag == 0)) {

      textreply[terminationcounter] = knownwords[bin2int][nextletter];
      nextletter++;
      terminationcounter++;

      if (terminationcounter == INPUTLENGTH - 3) {
        escapeflag = 1;
        textreply[INPUTLENGTH - 3] = '#';
        textreply[INPUTLENGTH - 2] = '\n';
        textreply[INPUTLENGTH - 1] = 0;
      }

    }

    /* If possible, add a space after the next word.
       Neither may the text be too long, nor may the
       "word" have been certain characters which are
       best not followed by space. */
    if ((escapeflag == 0) && (!((knownwords[bin2int][0] == '(') || (knownwords[bin2int][0] == '[') || (knownwords[bin2int][0] == '{')))) {
      /* for highlighting reasons: }]) */
      textreply[terminationcounter] = ' ';
      terminationcounter++;

      if (terminationcounter == INPUTLENGTH - 3) {
        escapeflag = 1;
        textreply[INPUTLENGTH - 3] = '#';
        textreply[INPUTLENGTH - 2] = '\n';
        textreply[INPUTLENGTH - 1] = 0;
      }
    }

    /* terminate if the output is becoming too long */
    if (escapeflag == 1) {
      break;
    }

    /* terminate output upon HASH SIGN: */
    if (knownwords[bin2int][0] == '#') {
      break;
    }

  }
  /* End of the testoutput loop. All output should have been created above. */


  /* OUTPUT TEXTREPLY */

  /* The dehash loop makes the output appear without hash-sign. */
  for (dehash = 0; dehash < INPUTLENGTH; dehash++) {
    if(textreply[dehash] == 0) {
      break;
    } else {
      if(textreply[dehash] == '#') {
        textreply[dehash] = ' ';
      }
    }
  }
  printf("SYSTEM: %s\n", textreply);


  /* LOWER PART */
  /* Previously, this was attached to the "upper part".
     It serves the purpose to incorporate the machine challenge, which
     the human already has seen above, into the longer history, to
     which the next human response is expected. */

  endofinput = MAXHISTORY - 1;
  for (findendofinput = 0; findendofinput < MAXHISTORY; findendofinput++) {
    if (wordnumber[findendofinput] == 0) {
      endofinput = findendofinput;
      break;
    }
  }

  if ((endofinput < MAXHISTORY - 1) && (endofinput != 0)) {
    endofinput++;

    for (loadhistory = 0; loadhistory < MAXHISTORY - endofinput; loadhistory++) {
      lasthistory[loadhistory] = lasthistory[loadhistory + endofinput];
    }

    for (loadhistory = 0; loadhistory < endofinput; loadhistory++) {
      lasthistory[MAXHISTORY - endofinput + loadhistory] = wordnumber[loadhistory];
    }
  } else if (endofinput == MAXHISTORY - 1) {

    for (loadhistory = 0; loadhistory < MAXHISTORY; loadhistory++) {
      lasthistory[loadhistory] = wordnumber[loadhistory];
    }
  } /* else there was no input - but then there is no history to update */

  /* END OF LOWER PART */

}

/* GREAT GENERAL LOOP HAS ENDED */

/* This label is jumped to in case of empty input above - it ensures
   a more "civilized" exit than just exit(0) and lets the system record
   what it has learned. */
exitnow:

/* WRITE KNOWN WORDS MATRIX TO FILE */

worddata = fopen("worddata.txt","w");
for (wordcounter = 0; wordcounter < WORDSKNOWN; wordcounter++) {
  fprintf(worddata, "%s\n", knownwords[wordcounter]);
}
fflush(worddata);
fclose(worddata);



/* OUTPUT IMPORTANCE TO FILE */

importancedata = fopen("importancedata.txt", "w");
for (imptrac = 0; imptrac < WORDSKNOWN; imptrac++) {
  fprintf(importancedata, "%d\n", importantwords[imptrac]);
}
fflush(importancedata);
fclose(importancedata);



/* WRITE HISTORY ARRAY TO FILE */

historydata = fopen("historydata.txt","w");
for (chargehistory = 0; chargehistory < MAXHISTORY; chargehistory++) {
  fprintf(historydata, "%d\n", lasthistory[chargehistory]);
}
fflush(historydata);
fclose(historydata);


/* OUTPUT ATOM ARRAY TO FILE */

atomdata = fopen("atomdata.txt", "w");
for (i = 0; i < atomcount; i++) {
  fprintf(atomdata, "%d\n", atoms[i][atomnumber]);
  fprintf(atomdata, "%d\n", atoms[i][leftatom]);
  fprintf(atomdata, "%d\n", atoms[i][rightatom]);
  fprintf(atomdata, "%d\n", atoms[i][atomvalue]);
}
fflush(atomdata);
fclose(atomdata);

/* Clean up and terminate OpenCL kernel usage */
if (neverrun != 1) {
  clReleaseProgram(program);
  clReleaseKernel(ko_conjcoh);
  clReleaseCommandQueue(commands);
  clReleaseContext(context);
}

}


