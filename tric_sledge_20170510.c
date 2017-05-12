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

#include<stdio.h>
#include<stdlib.h>
#include<math.h>

#define lowestatom 100
#define atomcount 30
#define leninput 10
#define atomnumber 0
#define leftatom 1
#define rightatom 2
#define atomvalue 3
#define positioninatoms 4
#define vicvalue -100
#define maxvic -1000
#define maxana 1000
#define reconsiderations 3
#define sledge 3

/* avoid generating more than n triangles */
# define maxtriangles 10

int main(void) {

/* INPUTARRAY IS THE SLIDING INPUT WINDOW */
/* int inputarray[leninput] = {11, 21, 31, 41, 51, 61, 71, 81, 91, 0}; */
/* int inputarray[leninput] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 0}; */
int inputarray[leninput] = {36, 16, 26, 36, 16, 26, 36, 16, 26, 36};

/* NOW FOLLOW ARRAYS FOR TEMPORARY COPOES OF THE INPUT */
int copyinputarray[leninput];
int extensioninput[leninput];

/* THIS IS THE ARRAY FOR PRESENTING OUTPUT OF THE SYSTEM */
int outputarray[leninput] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

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

/* diagnostic display of what triangles are being considered: */
int maxshow = 0;
int hypotrue = 0;

/* test */
/* load from file - here, generate experimentally */
for (i = 0; i < atomcount; i++) {
  atoms[i][0] = lowestatom + i;
  for (j = 1; j < 4; j++) {
    atoms[i][j] = 0;
  }
  
/*
  printf("%d %d %d %d\n", atoms[i][0], atoms[i][1], atoms[i][2], atoms[i][3]);
*/
  
}

atoms[0][leftatom] = 2;
atoms[0][rightatom] = 3;
atoms[0][atomvalue] = -vicvalue;

atoms[1][leftatom] = 1;
atoms[1][rightatom] = 2;
atoms[1][atomvalue] = -vicvalue;

atoms[2][leftatom] = 3;
atoms[2][rightatom] = 4;
atoms[2][atomvalue] = vicvalue;

atoms[3][leftatom] = 4;
atoms[3][rightatom] = 5;
atoms[3][atomvalue] = 2*vicvalue;

atoms[4][leftatom] = 5;
atoms[4][rightatom] = 6;
atoms[4][atomvalue] = vicvalue;


atoms[5][leftatom] = 5;
atoms[5][rightatom] = 3;
atoms[5][atomvalue] = -vicvalue;

atoms[6][leftatom] = 11;
atoms[6][rightatom] = 12;
atoms[6][atomvalue] = 13;

/*

atoms[7][leftatom] = ;
atoms[7][rightatom] = ;
atoms[7][atomvalue] = ;

atoms[8][leftatom] = ;
atoms[8][rightatom] = ;
atoms[8][atomvalue] = ;

atoms[9][leftatom] = ;
atoms[9][rightatom] = ;
atoms[9][atomvalue] = ;

*/

/* end test */

/* The input hierarchisation is attempted several times on each step
and for this sake, the input is copied prior to hierarchisation in
order to make it possible to be "restored" later for the "real"
hierarchisation. */

for (i = 0; i < leninput; i++) {
/* ACTIVATE BELOW AS SOON AS YOU PROGRAM READING IN REAL INPUT! */
/*      inputarray[i] = 0; */
  copyinputarray[i] = 0;
  extensioninput[i] = 0;
}
/* THIS IS ONLY FOR STARTUP - NEVER ZERO OUT THE INPUT DURING OPERATION!
   IT WILL LIKELY ALWAYS BE A "FULL" WINDOW!


/* main loop */


/* ---------------- HERE, IMPLANT THE TAKING OF USER AND THE IMPLANTATION OF THE USERINPUT INTO REAL INPUT ---------------- */

/* COPY THE INPUT FOR RECONSIDERATION: */

/* Create a copy of the input array in order to be
able to "reconsider" the input later so as to find
further (and indirekt) conclusions which were not
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
             (atoms[j][atomvalue] <= candidatevalue)) {
      
            candidateatom     = atoms[j][atomnumber];
            candidatevalue    = atoms[j][atomvalue];
            candidateleft     = inputarray[i];
            candidateright    = inputarray[i + 1];
            candidateposition = j;
            inputposition     = i;
      
          }
      
        }
      
      }
      
      /* if no atom was found, create a "hypothesis" - the tail pair */
      
      if (candidatevalue > maxana) {
      /* hypothesis creation */
      
        /* for the next hypothesis creation: */
        atomnumbertracer = atoms[atomcount - 1][atomnumber];
      
        /* create space for a new atom by "forgetting" an old atom */
        for (k = atomcount - 1; k > 0 ; k--) {
          atoms[k][atomnumber] = atoms[k - 1][atomnumber];
          atoms[k][leftatom]   = atoms[k - 1][leftatom];
          atoms[k][rightatom]  = atoms[k - 1][rightatom];
          atoms[k][atomvalue]  = atoms[k - 1][atomvalue];
        }
      
        inputposition = 0;
        /* find the inputposition */
        for (k = leninput - 1; k > 0; k--) {
          if (inputarray[k] != 0) {
            inputposition = k - 1;
            break;
          }
        }
        
        if ((inputarray[inputposition] == 0) && (inputarray[inputposition + 1] == 0)) {
          printf("input all empty, terminating\n");
          exit(0);
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
      printf("\n\nBEFORE TRIANGULATION:\n");
      for (i = 0; i < atomcount; i = i + 2) {
        printf("%d %d %d %d\t\t%d %d %d %d\n", atoms[i][0], atoms[i][1], atoms[i][2], atoms[i][3],
               atoms[i + 1][0], atoms[i + 1][1], atoms[i + 1][2], atoms[i + 1][3]);
      }
      /* end test */
      
      /* DO TRIANGULATION */
      
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
      
      /* Prepare for triangulation. */
      
      /* Find "conjugate" atoms that have ONLY ONE atom in
         common with the original atom. This already allows you
         to estimate "cohabitant" atoms, which have those two
         atoms as sub-atoms that are NOT in common. That is, if
         you are having A-B, now seek all atoms where e.g.
         A-C or C-A or B-C or C-B (without regard to type of
         vic or ana). */
      
      j = 0;
      /* not i = 0, because 0 is the original atom */
      for (i = 1; i < atomcount; i++) {
      
        /* stop generating triangles if over maximum range */
        if (j == maxtriangles) {
          break;
        }
      
        if ((atoms[0][leftatom] == atoms[i][leftatom])    &&
            (atoms[0][rightatom] != atoms[i][rightatom]) &&
            (atoms[0][leftatom] != 0)) {
      
          conjugate[j][atomnumber]      = atoms[i][atomnumber];
          conjugate[j][leftatom]        = atoms[i][leftatom];
          conjugate[j][rightatom]       = atoms[i][rightatom];
          conjugate[j][atomvalue]       = atoms[i][atomvalue];
          conjugate[j][positioninatoms] = i;
      
          cohabitant[j][leftatom]  = atoms[0][rightatom];
          cohabitant[j][rightatom] = atoms[i][rightatom];
      
          j++;
      
        } else
        if ((atoms[0][rightatom] == atoms[i][leftatom])   &&
            (atoms[0][leftatom] != atoms[i][rightatom]) &&
            (atoms[0][rightatom] != 0)) {
      
          conjugate[j][atomnumber]      = atoms[i][atomnumber];
          conjugate[j][leftatom]        = atoms[i][leftatom];
          conjugate[j][rightatom]       = atoms[i][rightatom];
          conjugate[j][atomvalue]       = atoms[i][atomvalue];
          conjugate[j][positioninatoms] = i;
      
          cohabitant[j][leftatom]  = atoms[0][leftatom];
          cohabitant[j][rightatom] = atoms[i][rightatom];
      
          j++;
      
        } else
        if ((atoms[0][leftatom] != atoms[i][leftatom])    &&
            (atoms[0][rightatom] == atoms[i][rightatom]) &&
            (atoms[0][rightatom] != 0)) {
      
          conjugate[j][atomnumber]      = atoms[i][atomnumber];
          conjugate[j][leftatom]        = atoms[i][leftatom];
          conjugate[j][rightatom]       = atoms[i][rightatom];
          conjugate[j][atomvalue]       = atoms[i][atomvalue];
          conjugate[j][positioninatoms] = i;
      
          cohabitant[j][leftatom]  = atoms[0][leftatom];
          cohabitant[j][rightatom] = atoms[i][leftatom];
      
          j++;
      
        } else
        if ((atoms[0][rightatom] != atoms[i][leftatom])   &&
            (atoms[0][leftatom] == atoms[i][rightatom]) &&
            (atoms[0][leftatom] != 0))  {
      
          conjugate[j][atomnumber]      = atoms[i][atomnumber];
          conjugate[j][leftatom]        = atoms[i][leftatom];
          conjugate[j][rightatom]       = atoms[i][rightatom];
          conjugate[j][atomvalue]       = atoms[i][atomvalue];
          conjugate[j][positioninatoms] = i;
      
          cohabitant[j][leftatom]  = atoms[0][rightatom];
          cohabitant[j][rightatom] = atoms[i][leftatom];
      
          j++;
      
        }
      
      }
      
      /* Find "cohabitant" atoms - eliminate doubletts. */
      
      /* That is, here you e.g. have the original connection
      A-B, you found a conjugate e.g. C-A, and now you are
      looking to find B=C (again, without regard to vic or
      ana). */
      
      for (j = 0; j < maxtriangles; j++) {
        alreadydone = 0;

        /* the triangle must not contain a "same side" as the
           given original side, or else you get repeated atoms
           in the list of all atoms: */

        if ((((cohabitant[j][leftatom] == atoms[0][leftatom]) &&
           (cohabitant[j][rightatom] == atoms[0][rightatom])) ||
           ((cohabitant[j][rightatom] == atoms[0][leftatom]) &&
           (cohabitant[j][leftatom] == atoms[0][rightatom]))) ||
           (((conjugate[j][leftatom] == atoms[0][leftatom]) &&
           (conjugate[j][rightatom] == atoms[0][rightatom])) ||
           ((conjugate[j][rightatom] == atoms[0][leftatom]) &&
           (conjugate[j][leftatom] == atoms[0][rightatom])))) {

          cohabitant[j][atomnumber]      = 0;
          cohabitant[j][leftatom]        = 0;
          cohabitant[j][rightatom]       = 0;
          cohabitant[j][atomvalue]       = 0;
          cohabitant[j][positioninatoms] = 0;
        
          conjugate[j][atomnumber]       = 0;
          conjugate[j][leftatom]         = 0;
          conjugate[j][rightatom]        = 0;
          conjugate[j][atomvalue]        = 0;
          conjugate[j][positioninatoms]  = 0;
        }
      
        /* not i = 0, because 0 is the original atom */
        for (i = 1; i < atomcount; i++) {
          if ((((cohabitant[j][leftatom]  == atoms[i][leftatom]) &&
                (cohabitant[j][rightatom] == atoms[i][rightatom])) ||
      
               ((cohabitant[j][rightatom] == atoms[i][leftatom]) &&
                (cohabitant[j][leftatom]  == atoms[i][rightatom]))) &&
      
                (!((cohabitant[j][rightatom] == 0) &&
                   (cohabitant[j][leftatom]  == 0)))) {
      
            cohabitant[j][atomnumber]      = atoms[i][atomnumber];
            cohabitant[j][leftatom]        = atoms[i][leftatom];
            cohabitant[j][rightatom]       = atoms[i][rightatom];
            cohabitant[j][atomvalue]       = atoms[i][atomvalue];
            cohabitant[j][positioninatoms] = i;
      
            /* KILL DOUBLETTS */
      
            for (k = 0; k < j; k++) {
              if (cohabitant[j][atomnumber] == conjugate[k][atomnumber]) {
      
                cohabitant[j][atomnumber]      = 0;
                cohabitant[j][leftatom]        = 0;
                cohabitant[j][rightatom]       = 0;
                cohabitant[j][atomvalue]       = 0;
                cohabitant[j][positioninatoms] = 0;
      
                conjugate[j][atomnumber]       = 0;
                conjugate[j][leftatom]         = 0;
                conjugate[j][rightatom]        = 0;
                conjugate[j][atomvalue]        = 0;
                conjugate[j][positioninatoms]  = 0;
      
                break;
              }
            }
      
            /* END KILLING DOUBLETTS */
      
            break;
          }
        }
      }
      
      /* TRIANGULATE */
      
      /* That is, you HAVE the atoms, now ADJUST THEIR VALUES.
      After having adjusted the values, you still need to TRANSFER
      the adjustments to the main knowledge array. */
      
      for (i = 0; i < maxtriangles; i++) {
      
        if (!((conjugate[i][atomnumber] == 0) && (cohabitant[i][atomnumber] == 0))) {
      
          if (((conjugate[i][atomvalue] >= 0) && (cohabitant[i][atomvalue] >= 0)) ||
              ((conjugate[i][atomvalue] < 0) && (cohabitant[i][atomvalue] < 0))) {
      
            atomeffects[i] = sqrt(1.0 * conjugate[i][atomvalue] * cohabitant[i][atomvalue]);
            if (atomeffects[i] > maxana) {
              atomeffects[i] = maxana;
            }
      
          } else {
      
            atomeffects[i] = -1 * sqrt(-1.0 * conjugate[i][atomvalue] * cohabitant[i][atomvalue]);
            if (atomeffects[i] < maxvic) {
              atomeffects[i] = maxvic;
            }
      
          }
      
          if (((atoms[0][atomvalue] >= 0) && (conjugate[i][atomvalue] >= 0)) ||
              ((atoms[0][atomvalue] < 0) && (conjugate[i][atomvalue] < 0))) {
      
            cohabitanteffects[i] = sqrt(1.0 * atoms[0][atomvalue] * conjugate[i][atomvalue]);
            if (cohabitanteffects[i] > maxana) {
              cohabitanteffects[i] = maxana;
            }
      
          } else {
      
            cohabitanteffects[i] = -1 * sqrt(-1.0 * atoms[0][atomvalue] * conjugate[i][atomvalue]);
            if (cohabitanteffects[i] < maxvic) {
              cohabitanteffects[i] = maxvic;
            }
      
          }
      
          if (((atoms[0][atomvalue] >= 0) && (cohabitant[i][atomvalue] >= 0)) ||
              ((atoms[0][atomvalue] < 0) && (cohabitant[i][atomvalue] < 0))) {
      
            conjugateeffects[i] = sqrt(1.0 * atoms[0][atomvalue] * cohabitant[i][atomvalue]);
            if (conjugateeffects[i] > maxana) {
              conjugateeffects[i] = maxana;
            }
      
          } else {
      
            conjugateeffects[i] = -1 * sqrt(-1.0 * atoms[0][atomvalue] * cohabitant[i][atomvalue]);
            if (conjugateeffects[i] < maxvic) {
              conjugateeffects[i] = maxvic;
            }
      
          }
      
        }
      
      }
      
      printf("\n\nBEFORE AFFECTION:\n");
      for (i = 0; i < atomcount; i = i + 2) {
        printf("%d %d %d %d\t\t%d %d %d %d\n", atoms[i][0], atoms[i][1], atoms[i][2], atoms[i][3],
               atoms[i + 1][0], atoms[i + 1][1], atoms[i + 1][2], atoms[i + 1][3]);
      }
      
      /* AFFECT THE ORIGINAL RELATION WITH THE RESULTS OF TRIANGULATION */
      /* That is, transfer the values. */
      
      
      atomeffectssum = 0;
      atomeffectsdivisor = 0;
      for (i = 0; i < maxtriangles; i++) {
      
        if (!((conjugate[i][atomnumber] == 0) && (cohabitant[i][atomnumber] == 0))) {
      
          atomeffectssum = atomeffectssum + atomeffects[i];
          atomeffectsdivisor++;
      
        }
      
      }
      
      if (atomeffectsdivisor > 0) {
      
        atomeffectssum = atomeffectssum / atomeffectsdivisor;
      
        atomeffectssum = atomeffectssum + atoms[0][atomvalue];
      
        if (atomeffectssum < maxvic) {
      
          atomeffectssum = maxvic;
      
        } else if (atomeffectssum > maxana) {
      
          atomeffectssum = maxana;
      
        }
      
        atoms[0][atomvalue] = atomeffectssum;
      
      }

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
      
      for (i = 0; i < maxtriangles; i++) {
      
        /* consider using an array instead of a scalar in order to parallelize */
        atomeffectssum = 0;
      
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
                    printf("-\n"); // printf("-\n");
                  } else
                  if ((cohabitant[i][3]*conjugate[i][3] == 0) || (atoms[0][3] == 0)) {
                    printf("0\n"); // printf("0\n");
                  } else {
                    printf("+\n"); // printf("+\n");
                  }
                } else {
                  printf("  [CONCLUDED] \n");
                }

              }
              /* END OF DIAGNOSTIC */


          atomeffectssum = conjugateeffects[i] + atoms[conjugate[i][positioninatoms]][atomvalue];
      
          if (atomeffectssum < maxvic) {
      
            atomeffectssum = maxvic;
      
          } else if (atomeffectssum > maxana) {
      
            atomeffectssum = maxana;
      
          }
      
          atoms[conjugate[i][positioninatoms]][atomvalue] = atomeffectssum;
      
        }
      
      }
      
      
      /* AFFECT EVERY COHABITANT - THE COHABITANTS MAY BE HYPOTHETICAL! */
      
      /* AFFECT ALL KNOWN COHABITANTS, COLLECT ALL HYPOTHESES */
      
      hypothesescount = 0;
      for (i = 0; i < maxtriangles; i++) {
      
        /* consider using an array instead of a scalar in order to parallelize */
        atomeffectssum = 0;
      
        if (conjugate[i][atomnumber] != 0) {
      
          /* if the atom is actually known */
          if (cohabitant[i][atomnumber] != 0) {
      
            atomeffectssum = cohabitanteffects[i] + atoms[cohabitant[i][positioninatoms]][atomvalue];
      
            if (atomeffectssum < maxvic) {
      
              atomeffectssum = maxvic;
      
            } else if (atomeffectssum > maxana) {
      
              atomeffectssum = maxana;
      
            }
      
            atoms[cohabitant[i][positioninatoms]][atomvalue] = atomeffectssum;
      
          /* otherwise, add the atom to the hypotheses */
          } else {
      
            hypotheses[hypothesescount][leftatom]  = cohabitant[i][leftatom];
            hypotheses[hypothesescount][rightatom] = cohabitant[i][rightatom];
            
            /* however, I CANNOT take the "cohabitant[i][atomvalue]" as value,
               as that is ZERO per definitionem. */
            
            atomeffectssum = cohabitanteffects[i];
      
            if (atomeffectssum < maxvic) {
      
              atomeffectssum = maxvic;
      
            } else if (atomeffectssum > maxana) {
      
              atomeffectssum = maxana;
      
            }
            
            hypotheses[hypothesescount][atomvalue] = atomeffectssum;
      
            hypothesescount++;
      
          }
      
        }
      
      }
      
      
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
      printf("\n\nHYPOTHESES BEFORE SCRUBBING:\n");
      for (i = 0; i < maxtriangles; i++) {
        printf("%d %d %d %d\n", hypotheses[i][0], hypotheses[i][1], hypotheses[i][2], hypotheses[i][3]);
      }
      /* end test */
      
      /* The hypotheses may contain duplicates. Remove the duplicates from a "scrub" array. */
      
      /* Transfer all hypotheses to the scrub array. */
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
      
      
      /* Remove multiple scrubhypotheses relating to the SAME pair: */
      scrubhypothesescount = hypothesescount;
      if (scrubhypothesescount > 0) {
        for (i = 0; i < scrubhypothesescount - 1; i++) {
      
        printf("scrubhypothesis: %d %d %d %d\n", scrubhypotheses[i][0], scrubhypotheses[i][1], scrubhypotheses[i][2], scrubhypotheses[i][3]);
      
          if (!((scrubhypotheses[i][leftatom] == 0) && (scrubhypotheses[i][rightatom] == 0))) {
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
      printf("\n\nHYPOTHESES AFTER SCRUBBING:\n");
      for (i = 0; i < maxtriangles; i++) {
        printf("%d %d %d %d\n", hypotheses[i][0], hypotheses[i][1], hypotheses[i][2], hypotheses[i][3]);
      }
      /* end test */
      
      /* END OF HYPOTHESES SCRUBBING */
      
      /* Transfer atom numbers to hypotheses. */
      
      j = atomcount;
      for (i = 0; i < hypothesescount; i++) {
        j--;
        hypotheses[i][atomnumber] = atoms[j][atomnumber];
      }
      
      /* "Forget" final atoms. */
      
      for (k = atomcount - 1; k > hypothesescount - 1; k--) {
        atoms[k][atomnumber] = atoms[k - hypothesescount][atomnumber];
        atoms[k][leftatom]   = atoms[k - hypothesescount][leftatom];
        atoms[k][rightatom]  = atoms[k - hypothesescount][rightatom];
        atoms[k][atomvalue]  = atoms[k - hypothesescount][atomvalue];
      }
      
      /* Transfer hypotheses to atoms array as "normal" atoms. */
      
      for (k = 0; k < hypothesescount; k++) {
        atoms[k][atomnumber] = hypotheses[k][atomnumber];
        atoms[k][leftatom]   = hypotheses[k][leftatom];
        atoms[k][rightatom]  = hypotheses[k][rightatom];
        atoms[k][atomvalue]  = hypotheses[k][atomvalue];
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
  printf("____INPUT ARRAY: ");
  for (i = 0; i < leninput; i++) {
    printf("%d ", inputarray[i]);
  }
  printf("\n");
  /* end test */
  printf("____SOLUTION FOUND: %d\n", solution);
  
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
  
          printf("solution: %d, plan for solution: %d, super-atom: %d, value: %d\n", solution, plan, atoms[i][atomnumber], valueplan);
  
          /* end test */
  
        }
      }
  
  /* test */
  printf("____SOLUTION FOUND LATER: %d\n", solution);
  /* end test */
  
      /* test */
      printf("plan___ was found: %d\n", plan);
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
  
              printf("replacement solution: %d, replacement plan for solution: %d, super-atom: %d, value: %d\n", solution, plan, atoms[i][atomnumber], valueplan);
  
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
  
  printf("plan: %d\n", plan);
  
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
  
  printf("PLAN FOR SOLUTION FOUND: ");
  for (i = 0; i < leninput; i++) {
    printf("%d ", outputarray[i]);
  }
  printf("\n");
  
  /* end test */
  
  /* IN FUTURE: IT WILL BE BEST TO _IGNORE_ 0s IN THE OUTPUT
  AND HAVE _ANOTHER_ SIGN FOR TERMINATION!
  So far, however, I have not determined such other sign. */
  
  
  /* test */
  printf("\n\nFINAL ATOMS:\n");
  for (i = 0; i < atomcount; i++) {
    printf("%d %d %d %d\n", atoms[i][0], atoms[i][1], atoms[i][2], atoms[i][3]);
  }
  /* end test */
  
  
  /* In hindsight: this entire section could operate on
     extensioninput instead of inputarray. */
  
  /* restore input for shifting */
  for (i = 0; i < leninput; i++) {
    inputarray[i] = extensioninput[i];
  }
  
  /* test */
  
  /* Mark one atom of the input so you can see it shifting: */
  /* inputarray[leninput - 1] = 1337; */
  
  
  for (i = 0; i < leninput; i++) {
    printf("%d ", inputarray[i]);
  }
  printf("\n");
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
  
    printf("length of output was: %d\n", lengthofoutput);
  
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
  printf("Shifted input array: \n");
  for (i = 0; i < leninput; i++) {
    printf("%d ", inputarray[i]);
  }
  printf("\n");
  /* end test */
  
  /* COPY THE INPUT FOR RECONSIDERATION: */
  for (i = 0; i < leninput; i++) {
    extensioninput[i] = inputarray[i];
  }
  
  /* end of sledge */
}


}
