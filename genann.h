/*
 * GENANN - Minimal C Artificial Neural Network
 *
 * Copyright (c) 2015, 2016 Lewis Van Winkle
 *
 * http://CodePlea.com
 *
 * This software is provided 'as-is', without any express or implied
 * warranty. In no event will the authors be held liable for any damages
 * arising from the use of this software.
 *
 * Permission is granted to anyone to use this software for any purpose,
 * including commercial applications, and to alter it and redistribute it
 * freely, subject to the following restrictions:
 *
 * 1. The origin of this software must not be misrepresented; you must not
 *    claim that you wrote the original software. If you use this software
 *    in a product, an acknowledgement in the product documentation would be
 *    appreciated but is not required.
 * 2. Altered source versions must be plainly marked as such, and must not be
 *    misrepresented as being the original software.
 * 3. This notice may not be removed or altered from any source distribution.
 *
 */


#ifndef __GENANN_H__
#define __GENANN_H__

#define GENANN_NUMBER_TYPE_IS_FLOAT
#define GENANN_NUMBER_TYPE float

#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifndef GENANN_RANDOM
/* We use the following for uniform random numbers between 0 and 1.
 * If you have a better function, redefine this macro. */
#define GENANN_RANDOM() (((GENANN_NUMBER_TYPE)rand())/RAND_MAX)
#endif


typedef GENANN_NUMBER_TYPE (*genann_actfun)(GENANN_NUMBER_TYPE a);

typedef struct genann_node {
    GENANN_NUMBER_TYPE output;
    GENANN_NUMBER_TYPE error; // delta between desired output and real output
    GENANN_NUMBER_TYPE gradient; // above, in terms of input, not output
    GENANN_NUMBER_TYPE * weight; // pointer to after self
    // extended by array of input weights
} genann_node;

typedef struct genann_nodelist {
    int inputs; // number of input weights on each node in this layer -- that includes the weight for the hidden/constant input
    int nodes; // number of nodes
    genann_actfun act;
    genann_node ** node; // pointer to after self
    // extended by list of pointers to nodes
    // extended by list of nodes
} genann_nodelist;

typedef struct genann {
    /* How many inputs, how many layers (including hidden and output but not input) */
    int inputs, layers;
    
    genann_nodelist ** layer; // pointer to after self
    // extended by list of pointers to layers
    // extended by list of layers
} genann;

/* Creates and returns a new ann. */
genann * genann_init(int inputs, int layers, int * nodelist, char * functions);

/* Creates ANN from file saved with genann_write. */
genann *genann_read(FILE *in);

/* Sets weights randomly. Called by init. */
void genann_randomize(genann *ann);

/* Returns a new copy of ann. */
genann *genann_copy(genann const *ann);

/* Frees the memory used by an ann. */
void genann_free(genann *ann);

/* Returns a particular output value from the network */
GENANN_NUMBER_TYPE genann_output(genann const * ann, int n);

/* Runs the feedforward algorithm to calculate the ann's output. */
void genann_run(genann const * ann, GENANN_NUMBER_TYPE const * inputs);

/* Does a single backprop update. */
void genann_train(genann const *ann, GENANN_NUMBER_TYPE const *inputs, GENANN_NUMBER_TYPE const *desired_outputs, GENANN_NUMBER_TYPE learning_rate);

/* Saves the ann. */
void genann_write(genann const *ann, FILE *out);


GENANN_NUMBER_TYPE genann_act_tanh(GENANN_NUMBER_TYPE a);
GENANN_NUMBER_TYPE genann_act_tanh_cached(GENANN_NUMBER_TYPE a);
GENANN_NUMBER_TYPE genann_act_threshold(GENANN_NUMBER_TYPE a);
GENANN_NUMBER_TYPE genann_act_linear(GENANN_NUMBER_TYPE a);


#ifdef __cplusplus
}
#endif

#endif /*__GENANN_H__*/
