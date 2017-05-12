#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* DEFS FROM LOGICAL TRIANGULATION */

#define lowestatom 10001
#define atomcount 3000000
#define leninput 100
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
# define maxtriangles 100

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

/* triangulation arrays */

/* CONJUGATE: second triangle side;
5 atoms, not 4, because you
ALSO want to know WHERE the atom
was found. The first triangle
side is regarded as GIVEN. */
int conjugate[maxtriangles][5];

/* The third triangle side is
found in the COHABITANT array. */
int cohabitant[maxtriangles][5];

/* Cohabitants can be CONCLUSIONS,
that is SO FAR UNKNOWN ATOMS, or -
HYPOTHESES. */
int hypotheses[maxtriangles][4];
/* auxiliary arrays: */
int scrubhypotheses[maxtriangles][4];
int hypothesescount;
int scrubhypothesescount;
int alreadydone;

/* The values of triangulation -
as this can be "value * value",
the capacity should be large. */
long atomeffects[maxtriangles];
long atomeffectssum;
long atomeffectsdivisor;
long conjugateeffects[maxtriangles];
long cohabitanteffects[maxtriangles];

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

int main(void) {




/* END OF LOGICAL TRIANGULATION VARIABLES */

FILE *atomdata;

for (i = 0; i < atomcount; i++) {
  atoms[i][0] = lowestatom + i;
  for (j = 1; j < 4; j++) {
    atoms[i][j] = 0;
  }
  

//  printf("%d %d %d %d\n", atoms[i][0], atoms[i][1], atoms[i][2], atoms[i][3]);

  
}

atomdata = fopen("atomdata.txt", "w");
for (i = 0; i < atomcount; i++) {
  fprintf(atomdata, "%d\n", atoms[i][atomnumber]);
  fprintf(atomdata, "%d\n", atoms[i][leftatom]);
  fprintf(atomdata, "%d\n", atoms[i][rightatom]);
  fprintf(atomdata, "%d\n", atoms[i][atomvalue]);
}
fclose(atomdata);

}
