/*
/usr/local/cuda/bin/nvcc -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -o ocl ocl.c -lm -lOpenCL -g

gcc -I/opt/AMDAPPSDK-2.9-1/include -L/opt/AMDAPPSDK-2.9-1/lib/x86_64 -o tric tric_parser_ocl20170507.c -lm -lOpenCL -g

/usr/local/cuda/bin/nvcc -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -o tric tric_parser_ocl20170507.c -lm -lOpenCL -g

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
#endif
/* END OPENCL */

/* DEFS FROM LOGICAL TRIANGULATION */

#define lowestatom 10001
/* 1000000 seems to be the slowest still tolerable thing, 100k is pretty OK */
#define atomcount 3000000
#define leninput 60
#define atomnumber 0
#define leftatom 1
#define rightatom 2
#define atomvalue 3
#define positioninatoms 4
#define vicvalue -1000
#define maxvic -30000
#define maxana 30000
#define reconsiderations 3
#define sledge 3

/* avoid generating more than n triangles */
# define maxtri 2000

/* THIS IS WITH DOUBLETTS - that is, make it double as large as desired: */
# define trilimit 2000

/* absolute range limit for triangulation - disadviseable; but faster */
/* atoms behind that range will not even be considered as possible triangles sides */
# define absolutetrilimit 2000

/* END OF LOGICAL TRIANGULATION DEFS */

/* WORDSKNOWN is how many words the system may know - the count of columns of worddata.txt */
#define WORDSKNOWN 10000
/* LETTERSPERWORD determines how many letters a word may have - more is better, the increase in memory requirements is pretty unimportant */
#define LETTERSPERWORD 20
/* INPUTLENGTH defines the maximum size of input and output in chars. */
#define INPUTLENGTH 1000

/* BY PLACING THE LARGE ARRAYS OUTSIDE THE MAIN FUNCTION, I WEASEL MY WAY AROUND STACK LIMITATIONS - ALLOCATION THEN IS IN MAIN MEMORY. */

/* MAXHISTORY defines the maximum size of input in words. */
/* If maxhistory is inter, the system may understand more differentiated input - but is harder to teach in general. */
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
"   int j = get_global_id(1);             \n" \
"   if (i < ktrilimit) {                        \n" \
"                                                                 \n" \
"                                                                 \n" \
"                                                                 \n" \
"           if ((i > 0) &&                                                     \n" \
"                 (atoms_0_leftatom == katoms[j*4 + leftatom])    &&           \n" \
"                 (atoms_0_rightatom != katoms[j*4 + rightatom]) &&            \n" \
"                 (atoms_0_leftatom != 0)) {                                   \n" \
"                                                                              \n" \
"               kconjugate[i*5 + atomnumber]      = katoms[j*4 + atomnumber];  \n" \
"               kconjugate[i*5 + leftatom]        = katoms[j*4 + leftatom];    \n" \
"               kconjugate[i*5 + rightatom]       = katoms[j*4 + rightatom];   \n" \
"               kconjugate[i*5 + atomvalue]       = katoms[j*4 + atomvalue];   \n" \
"               kconjugate[i*5 + positioninatoms] = i;                         \n" \
"                                                                              \n" \
"               kcohabitant[i*5 + leftatom]  = atoms_0_rightatom;              \n" \
"               kcohabitant[i*5 + rightatom] = katoms[j*4 + rightatom];        \n" \
"                                                                              \n" \
"               kcohabitant[i*5 + atomnumber] = 0;                             \n" \
"               kcohabitant[i*5 + atomvalue] = 0;                              \n" \
"               kcohabitant[i*5 + positioninatoms] = 0;                        \n" \
"                                                                              \n" \
"             } else                                                           \n" \
"             if ((i > 0) &&                                                   \n" \
"                 (atoms_0_rightatom == katoms[j*4 + leftatom])   &&           \n" \
"                 (atoms_0_leftatom != katoms[j*4 + rightatom]) &&             \n" \
"                 (atoms_0_rightatom != 0)) {                                  \n" \
"                                                                              \n" \
"               kconjugate[i*5 + atomnumber]      = katoms[j*4 + atomnumber];  \n" \
"               kconjugate[i*5 + leftatom]        = katoms[j*4 + leftatom];    \n" \
"               kconjugate[i*5 + rightatom]       = katoms[j*4 + rightatom];   \n" \
"               kconjugate[i*5 + atomvalue]       = katoms[j*4 + atomvalue];   \n" \
"               kconjugate[i*5 + positioninatoms] = i;                         \n" \
"                                                                              \n" \
"               kcohabitant[i*5 + leftatom]  = atoms_0_leftatom;               \n" \
"               kcohabitant[i*5 + rightatom] = katoms[j*4 + rightatom];        \n" \
"                                                                              \n" \
"               kcohabitant[i*5 + atomnumber] = 0;                             \n" \
"               kcohabitant[i*5 + atomvalue] = 0;                              \n" \
"               kcohabitant[i*5 + positioninatoms] = 0;                        \n" \
"                                                                              \n" \
"             } else                                                           \n" \
"             if ((i > 0) &&                                                   \n" \
"                 (atoms_0_leftatom != katoms[j*4 + leftatom])    &&           \n" \
"                 (atoms_0_rightatom == katoms[j*4 + rightatom]) &&            \n" \
"                 (atoms_0_rightatom != 0)) {                                  \n" \
"                                                                              \n" \
"               kconjugate[i*5 + atomnumber]      = katoms[j*4 + atomnumber];  \n" \
"               kconjugate[i*5 + leftatom]        = katoms[j*4 + leftatom];    \n" \
"               kconjugate[i*5 + rightatom]       = katoms[j*4 + rightatom];   \n" \
"               kconjugate[i*5 + atomvalue]       = katoms[j*4 + atomvalue];   \n" \
"               kconjugate[i*5 + positioninatoms] = i;                         \n" \
"                                                                              \n" \
"               kcohabitant[i*5 + leftatom]  = atoms_0_leftatom;               \n" \
"               kcohabitant[i*5 + rightatom] = katoms[j*4 + leftatom];         \n" \
"                                                                              \n" \
"               kcohabitant[i*5 + atomnumber] = 0;                             \n" \
"               kcohabitant[i*5 + atomvalue] = 0;                              \n" \
"               kcohabitant[i*5 + positioninatoms] = 0;                        \n" \
"                                                                              \n" \
"             } else                                                           \n" \
"             if ((i > 0) &&                                                   \n" \
"                 (atoms_0_rightatom != katoms[j*4 + leftatom])   &&           \n" \
"                 (atoms_0_leftatom == katoms[j*4 + rightatom]) &&             \n" \
"                 (atoms_0_leftatom != 0))  {                                  \n" \
"                                                                              \n" \
"               kconjugate[i*5 + atomnumber]      = katoms[j*4 + atomnumber];  \n" \
"               kconjugate[i*5 + leftatom]        = katoms[j*4 + leftatom];    \n" \
"               kconjugate[i*5 + rightatom]       = katoms[j*4 + rightatom];   \n" \
"               kconjugate[i*5 + atomvalue]       = katoms[j*4 + atomvalue];   \n" \
"               kconjugate[i*5 + positioninatoms] = i;                         \n" \
"                                                                              \n" \
"               kcohabitant[i*5 + leftatom]  = atoms_0_rightatom;              \n" \
"               kcohabitant[i*5 + rightatom] = katoms[j*4 + leftatom];         \n" \
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
" barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);                                   \n" \
"                                                                                    \n" \
"                                                                                    \n" \
"                                                                                    \n" \
"                                                                                    \n" \
"                                                                                    \n" \
" if (i < ktrilimit) {                                                           \n" \
"                                                                                    \n" \
"           /* Find /cohabitant/ atoms - eliminate doubletts. */                     \n" \
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
" if ((i < ktrilimit) && (j < katomcount)) {                                                           \n" \
"                                                                                    \n" \
"             /* not i = 0, because 0 is the original atom */                        \n" \
"             /* why does it work to get rid of this loop? */                        \n" \
"             //for (j = 1; j < katomcount; j++) {                                   \n" \
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
"             //    break;                                                             \n" \
"                                                                                    \n" \
"               }                                                                    \n" \
"             //}                                                                      \n" \
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


int main() {


/* BEGIN OPENCL */

    cl_int          err;               // error code returned from OpenCL calls

    size_t dataSizeConjCoh = sizeof(int) * absolutetrilimit * 5;
    size_t dataSizeAtoms = sizeof(int) * atomcount * 4;
    size_t dataSizeEffects = sizeof(int) * absolutetrilimit;

    size_t global[2]; // = {absolutetrilimit, atomcount};                  // global domain size
    global[0] = absolutetrilimit;
    global[1] = atomcount;

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

/* END OPENCL */


/* zero out all array for test purposes & set atom numbers */
// DELETE MOST OF THIS ONCE YOU HANDLE STUFF OVER OPENCL
// ACTUALLY, KEEP IT!!!! IT TURNED OUT TO ME SOMEHOW IMPORTANT!
for (i = 0; i < maxtri; i++) {
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

for (i = 0; i < atomcount; i++) {
  for (k = 1; k < 4; k++) {
    atoms[i][k] = 0;
  }

  atoms[i][0] = i + lowestatom;

}

/*
atoms[][leftatom]   = ;
atoms[][rightatom]  = ;
atoms[][atomvalue]  = ;
*/

atoms[0][leftatom]   = 1;
atoms[0][rightatom]  = 2;
atoms[0][atomvalue]  = -120;

atoms[1][leftatom]   = 3;
atoms[1][rightatom]  = 4;
atoms[1][atomvalue]  = 100;

atoms[2][leftatom]   = 1;
atoms[2][rightatom]  = 5;
atoms[2][atomvalue]  = 200;

atoms[3][leftatom]   = 1;
atoms[3][rightatom]  = 6;
atoms[3][atomvalue]  = -300;

atoms[4][leftatom]   = 2;
atoms[4][rightatom]  = 6;
atoms[4][atomvalue]  = 50;

atoms[5][leftatom]   = 7;
atoms[5][rightatom]  = 8;
atoms[5][atomvalue]  = -10;

atoms[6][leftatom]   = 2;
atoms[6][rightatom]  = 9;
atoms[6][atomvalue]  = -40;

atoms[7][leftatom]   = 1;
atoms[7][rightatom]  = 9;
atoms[7][atomvalue]  = -40;

printf("present atoms:\n");
for (i = 0; i < 10; i++) {
  printf("atom: %d, left: %d, right: %d, value: %d\n", atoms[i][0], atoms[i][1], atoms[i][2], atoms[i][3]);
}


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
          // DELETE MOST OF THIS ONCE YOU HANDLE STUFF OVER OPENCL
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


printf("\nSTAGE X\n");

    // Create the input (a, b, e, g) arrays in device memory
    // NB: we copy the host pointers here too
    d_katoms  = clCreateBuffer(context,  CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,  dataSizeAtoms, atoms, &err);
    checkError(err, "Creating buffer d_katoms");

    // Create the output arrays in device memory
    d_kconjugate  = clCreateBuffer(context,  CL_MEM_READ_WRITE, dataSizeConjCoh, NULL, &err);
    checkError(err, "Creating buffer d_kconjugate");

    // Create the output arrays in device memory
    d_kcohabitant  = clCreateBuffer(context,  CL_MEM_READ_WRITE, dataSizeConjCoh, NULL, &err);
    checkError(err, "Creating buffer d_kcohabitant");

    // Create the output arrays in device memory
    d_katomeffects  = clCreateBuffer(context,  CL_MEM_READ_WRITE, dataSizeEffects, NULL, &err);
    checkError(err, "Creating buffer d_katomeffects");

    // Create the output arrays in device memory
    d_katomeffectssum  = clCreateBuffer(context,  CL_MEM_READ_WRITE, dataSizeEffects, NULL, &err);
    checkError(err, "Creating buffer d_katomeffectssum");

    // Create the output arrays in device memory
    d_kconjugateeffects  = clCreateBuffer(context,  CL_MEM_READ_WRITE, dataSizeEffects, NULL, &err);
    checkError(err, "Creating buffer d_kconjugateeffects");

    // Create the output arrays in device memory
    d_kcohabitanteffects  = clCreateBuffer(context,  CL_MEM_READ_WRITE, dataSizeEffects, NULL, &err);
    checkError(err, "Creating buffer d_kcohabitanteffects");

    // THIS SHOULD BE MOVED TO THE DELCLARATIONS; HERE, IT SHOULD ONLY BE POPULATED.
    int d_atoms_0_leftatom = atoms[0][leftatom];
    int d_atoms_0_rightatom = atoms[0][rightatom];
    int d_atoms_0_atomvalue = atoms[0][atomvalue];
    const int d_ktrilimit = absolutetrilimit; 
    const int d_katomcount = atomcount; // Just experimenting here; in reality, omit that & use atomcount or trilimit.

printf("\nSTAGE Y\n");


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
    //global = d_ktrilimit; // d_katomcount; // normally use atomcount... or trilimit
    err = clEnqueueNDRangeKernel(commands, ko_conjcoh, 2, NULL, global, NULL, 0, NULL, NULL);
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

printf("\nSTAGE Z\n");

printf("translated atoms:\n");
for (i = 0; i < 10; i++) {
  printf("conjugate: %d, left: %d, right: %d, value: %d, position: %d, conjeffect: %d\n", conjugate[i][0], conjugate[i][1], conjugate[i][2], conjugate[i][3], conjugate[i][4], conjugateeffects[i]);
  printf("cohabitant: %d, left: %d, right: %d, value: %d, position: %d, coheffect: %d\n", cohabitant[i][0], cohabitant[i][1], cohabitant[i][2], cohabitant[i][3], cohabitant[i][4], cohabitanteffects[i]);
}



    // cleanup then shutdown
    clReleaseMemObject(d_katoms);
    clReleaseMemObject(d_kconjugate);
    clReleaseMemObject(d_kcohabitant);
    clReleaseMemObject(d_katomeffects);
    clReleaseMemObject(d_katomeffectssum);
    clReleaseMemObject(d_kconjugateeffects);
    clReleaseMemObject(d_kcohabitanteffects);
    clReleaseProgram(program);
    clReleaseKernel(ko_conjcoh);
    clReleaseCommandQueue(commands);
    clReleaseContext(context);


/* END OPENCL */

















          /* Prepare for triangulation. */
          
          /* Find "conjugate" atoms that have ONLY ONE atom in
             common with the original atom. This already allows you
             to estimate "cohabitant" atoms, which have those two
             atoms as sub-atoms that are NOT in common. That is, if
             you are having A-B, now seek all atoms where e.g.
             A-C or C-A or B-C or C-B (without regard to type of
             vic or ana). */
          

/* 
printf("\nPRIOR POSITIONS:\n");
for (i = 0; i < 15; i++) {
  printf("conjugate: %d, left: %d, right: %d, value: %d, position: %d,\ncohabitant: %d, left: %d, right: %d, value: %d, position: %d\n\n",
         conjugate[i][0], conjugate[i][1], conjugate[i][2], conjugate[i][3], conjugate[i][4],
         cohabitant[i][0], cohabitant[i][1], cohabitant[i][2], cohabitant[i][3], cohabitant[i][4]);
}

printf("1. maxtriangles = %d    ", maxtriangles);
*/

/*
printf("2. maxtriangles = %d    ", maxtriangles);
*/
          /* absolute triangulation limit */
// RATHER, MAKE THIS PERMANENT:
          if (maxtriangles > absolutetrilimit) {
            maxtriangles = absolutetrilimit;
          }

printf("3. maxtriangles = %d, CRASHING IF HIGH ENOUGH\n", maxtriangles);
//maxtriangles = 10;
         
/*
printf("\nprior to affection:\n");
for (i = 0; i < maxtriangles; i++) {
  printf("conjugate: %d, left: %d, right: %d, value: %d, position: %d,\ncohabitant: %d, left: %d, right: %d, value: %d, position: %d\n\n",
         conjugate[i][0], conjugate[i][1], conjugate[i][2], conjugate[i][3], conjugate[i][4],
         cohabitant[i][0], cohabitant[i][1], cohabitant[i][2], cohabitant[i][3], cohabitant[i][4]);
}

printf("\ntriangulation effects to apply:\n");
for (i = 0; i < maxtriangles; i++) {
  printf("conjugateeffects: %d, left: %d, right: %d, position: %d,   cohabitanteffects: %d, left: %d, right: %d, position: %d\n",
         conjugateeffects[i], conjugate[i][leftatom], conjugate[i][rightatom], conjugate[i][positioninatoms],
         cohabitanteffects[i], cohabitant[i][leftatom], cohabitant[i][rightatom], cohabitant[i][positioninatoms]);
}

printf("corresponding atoms:\n");
for (i = 0; i < maxtriangles; i++) {
  printf("position: %d, atom: %d, left: %d, right: %d, value: %d\n", i, atoms[i][0], atoms[i][1], atoms[i][2], atoms[i][3]);
}
*/

/* LOOKS OK SO FAR. */

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
          
          /* AFFECT EVERY CONJUGATE - THE CONJUGATES ARE ALL KNOWN! */
          #pragma omp parallel for
          for (i = 0; i < maxtriangles; i++) {
          
            /* consider using an array instead of a scalar in order to parallelize */
            atomeffectssum[i] = 0;
          
            if (!((conjugate[i][atomnumber] == 0) && (cohabitant[i][atomnumber] == 0))) {
          
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
          

printf("\nAFTER TRANSFER:\n");
for (i = 0; i < 20; i++) {

  printf("atom: %d, left: %d, right: %d, value: %d\n", atoms[i][0], atoms[i][1], atoms[i][2], atoms[i][3]);

}

/*
#pragma omp barrier

          #pragma omp parallel for
          for (k = atomcount - 1; k >=0 ; k--) {
            atomscopy[k][atomnumber] = atoms[k][atomnumber];
            atomscopy[k][leftatom]   = atoms[k][leftatom];
            atomscopy[k][rightatom]  = atoms[k][rightatom];
            atomscopy[k][atomvalue]  = atoms[k][atomvalue];
          }
*/

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

/*
printf("\nBEFORE BUBBLING:\n");
for (i = 0; i < 20; i++) {

  printf("atom: %d, left: %d, right: %d, value: %d\n", atoms[i][0], atoms[i][1], atoms[i][2], atoms[i][3]);

}
*/
        
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
                  bubblevalue    = atoms[j][atomvalue]; /* ERROR: WAS TAKING THAT FROM CONJUGATE */
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
                    bubblevalue    = atoms[j][atomvalue]; /* ERROR: WAS TAKING THAT FROM COHABITANT */
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

/*
printf("\nEND RESULT:\n");
for (i = 0; i < 20; i++) {

  printf("atom: %d, left: %d, right: %d, value: %d\n", atoms[i][0], atoms[i][1], atoms[i][2], atoms[i][3]);

}
*/


}

