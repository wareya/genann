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
 */
 

// overhaul/hard fork to support different numbers of neurons per layer, 2017, wareya

#include "genann.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <stdio.h>

#define LOOKUP_SIZE 4096
#define GENANN_MAX_WEIGHT_VALUE 3

GENANN_NUMBER_TYPE genann_act_tanh(GENANN_NUMBER_TYPE a)
{
    #ifdef GENANN_NUMBER_TYPE_IS_FLOAT
        return tanhf(a);
    #else
        return tanh(a);
    #endif
}

GENANN_NUMBER_TYPE genann_act_tanh_cached(GENANN_NUMBER_TYPE a)
{
    // If you're optimizing for memory usage, just
    // delete this entire function and replace references
    // of genann_act_tanh_cached to genann_act_tanh
     
    const GENANN_NUMBER_TYPE min = -15.0;
    const GENANN_NUMBER_TYPE max = 15.0;
    static GENANN_NUMBER_TYPE interval;
    static int initialized = 0;
    static GENANN_NUMBER_TYPE lookup[LOOKUP_SIZE];

    // Calculate entire lookup table on first run. 
    if (!initialized)
    {
        interval = (max - min) / LOOKUP_SIZE;
        int i;
        for (i = 0; i < LOOKUP_SIZE; ++i)
            lookup[i] = genann_act_tanh(min + interval * i);
        // This is down here to make this thread safe. // ???
        initialized = 1;
    }

    int i;
    i = (int)((a-min)/interval+0.5);
    if (i <= 0) return lookup[0];
    if (i >= LOOKUP_SIZE) return lookup[LOOKUP_SIZE-1];
    return lookup[i];
}

GENANN_NUMBER_TYPE genann_act_threshold(GENANN_NUMBER_TYPE a)
{
    return a > 0;
}

GENANN_NUMBER_TYPE genann_act_linear(GENANN_NUMBER_TYPE a)
{
    return a;
}

GENANN_NUMBER_TYPE genann_act_linear_notrain(GENANN_NUMBER_TYPE a)
{
    return a;
}

genann * genann_init(int inputs, int layers, int * nodelist, char * functions)
{
    if (inputs < 1) return 0;
    if (layers < 1) return 0;
    
    // Make sure all layers have at least one output
    for (int i = 0; i < layers; i++)
        if (nodelist[i] == 0) return 0;
    for (int i = 0; i < layers; i++)
        if (functions[i] != 'l' && functions[i] != 't' && functions[i] != 'c') return 0;
    
    int size = sizeof(genann);
    
    size += sizeof(genann_nodelist*)*layers; // size of pointers to layers
    size += sizeof(genann_nodelist)*layers; // base size of each layer
    
    for(int i = 0; i < layers; i++)
    {
        int layer_inputs = (i==0) ? (inputs+1) : (nodelist[i-1]+1); // number of inputs to each node (previous layer's nodes plus one)
        int nodelength = sizeof(genann_node)+sizeof(GENANN_NUMBER_TYPE)*layer_inputs; // base size of node plus size of all weights
        
        size += sizeof(genann_node*)*nodelist[i]; // size of pointers to nodes in this layer
        size += nodelength*nodelist[i]; // size of all nodes in this layer
    }
    
    genann * ret = (genann*)malloc(size);
    if (!ret) return 0;
    
    ret->inputs = inputs;
    ret->layers = layers;
    ret->layer = (genann_nodelist**)(((char*)ret)+sizeof(genann)); // pointer to after self
    
    char * index = (char*)(&ret->layer[layers]); // address past end of list of pointers to layers
    for(int i = 0; i < layers; i++)
    {
        ret->layer[i] = (genann_nodelist*)index;
        
        genann_nodelist & layer = *ret->layer[i];
        
        layer.act = (functions[i]=='t')?genann_act_tanh:(functions[i]=='l')?genann_act_linear:genann_act_linear_notrain;
        layer.inputs = (i==0) ? (inputs+1) : (nodelist[i-1]+1);
        int nodelength = sizeof(genann_node)+sizeof(GENANN_NUMBER_TYPE)*layer.inputs;
        layer.nodes = nodelist[i];
        layer.node = (genann_node**)&(ret->layer[i][1]); // address past end of "real" struct
        char * index2 = (char*)(&layer.node[layer.nodes]); // address past end of list of pointers to nodes
        for(int n = 0; n < layer.nodes; n++)
        {
            layer.node[n] = (genann_node*)index2;
            layer.node[n]->output = 1.0; // dummy value
            layer.node[n]->error = 3.14159; // dummy value
            layer.node[n]->weight = (GENANN_NUMBER_TYPE*)&(layer.node[n][1]); // address past end of "real" struct
            index2 += nodelength;
        }
        
        index += sizeof(genann_nodelist) + sizeof(genann_node*)*layer.nodes + nodelength*layer.nodes;
    }

    genann_randomize(ret);
    
    return ret;
}

void genann_randomize(genann *ann)
{
    for (int l = 0; l < ann->layers; l++)
    {
        for (int n = 0; n < ann->layer[l]->nodes; n++)
        {
            for (int w = 0; w < ann->layer[l]->inputs; w++)
            {
                ann->layer[l]->node[n]->weight[w] = (GENANN_RANDOM()) - 0.5;
            }
        }
    }
}

void genann_free(genann *ann)
{
    // Everything is stored in a single buffer.
    free(ann);
}

// get a particular output from the neural network
GENANN_NUMBER_TYPE genann_output(genann const * ann, int n)
{
    return ann->layer[ann->layers-1]->node[n]->output;
}

GENANN_NUMBER_TYPE get_input(genann const * ann, const GENANN_NUMBER_TYPE * inputs, int l, int w)
{
    genann_nodelist & layer = *ann->layer[l];
    //genann_nodelist * lastlayer = ;

    GENANN_NUMBER_TYPE input;
    if(w != layer.inputs-1)
    {
        if(l == 0) // first layer, take from inputs instead of another layer
            input = inputs[w];
        else
            input = ann->layer[l-1]->node[w]->output;
            
    }
    else // dummy/constant node, no corresponding input
        input = -1.0;
    
    return input;
}

void genann_run(genann const * ann, GENANN_NUMBER_TYPE const * inputs) {
    for (int l = 0; l < ann->layers; l++)
    {
        genann_nodelist & layer = *ann->layer[l];
        for (int n = 0; n < layer.nodes; n++)
        {
            genann_node & node = *layer.node[n];
            GENANN_NUMBER_TYPE value = 0;
            for (int w = 0; w < layer.inputs; w++)
                value += node.weight[w] * get_input(ann, inputs, l, w);
            
            value = layer.act(value);
            
            node.output = value;
        }
    }
}

void genann_train(genann const * ann, GENANN_NUMBER_TYPE const * inputs, GENANN_NUMBER_TYPE const * desired_output, GENANN_NUMBER_TYPE learning_rate)
{
    // Get what the network thinks of our current input
    genann_run(ann, inputs);

    genann_nodelist & output_layer = *ann->layer[ann->layers-1];
    for (int n = 0; n < output_layer.nodes; n++)
    {
        genann_node & node = *output_layer.node[n];
        node.error = desired_output[n] - node.output;
        if(output_layer.act != genann_act_linear)
            node.error *= 1.0 - node.output * node.output;
    }
    
    // start at layer before output layer, work our way back towards input
    for (int l = ann->layers-2; l >= 0; l--)
    {
        genann_nodelist & layer = *ann->layer[l];
        genann_nodelist & nextlayer = *ann->layer[l+1];
        for (int n = 0; n < layer.nodes; n++)
        {
            genann_node & node = *layer.node[n];
            
            node.error = 0;
            for(int n2 = 0; n2 < nextlayer.nodes; n2++) // loop over nodes in the next layer
            {
                genann_node & node2 = *nextlayer.node[n2];
                node.error += node2.error * node2.weight[n]; // its error times the strength of our connection to it
            }
            
            if(layer.act != genann_act_linear)
                node.error *= 1.0 - node.output * node.output;
        }
    }
    
    // train all the weights
    for (int l = 0; l < ann->layers; l++)
    {
        genann_nodelist & layer = *ann->layer[l];
        if(layer.act == genann_act_linear_notrain)
        {
            continue;
        }
        for (int n = 0; n < layer.nodes; n++)
        {
            genann_node & node = *layer.node[n];
            
            for (int w = 0; w < layer.inputs; w++)
            {
                // Apply error (against input that caused the error) to the weight
                node.weight[w] += get_input(ann, inputs, l, w) * node.error * learning_rate; 
                if(node.weight[w] > GENANN_MAX_WEIGHT_VALUE) node.weight[w] = GENANN_MAX_WEIGHT_VALUE;
                if(node.weight[w] < -GENANN_MAX_WEIGHT_VALUE) node.weight[w] = -GENANN_MAX_WEIGHT_VALUE;
            }
        }
    }
}

/*

genann *genann_read(FILE *in) {
    int inputs, hidden_layers, hidden, outputs;
    
    fscanf(in, "%d %d", &inputs, &hidden_layers);
    
    int * nodelist = (int*)malloc(sizeof(int)*hidden_layers);
    for(int i = 0; i < hidden_layers; i++)
        fscanf(in, " %d", nodelist[i]);
    
    fscanf(in, "%d", &outputs);

    genann *ann = genann_init(inputs, hidden_layers, nodelist, outputs);
    free(nodelist);
    
    int i;
    for (i = 0; i < ann->total_weights; ++i) {
        fscanf(in, " %le", ann->weight + i);
    }

    return ann;
}


genann *genann_copy(genann const *ann) {
    const int size = sizeof(genann) + sizeof(GENANN_NUMBER_TYPE) * (ann->total_weights + ann->total_neurons + (ann->total_neurons - ann->inputs));
    genann *ret = (genann*)malloc(size);
    if (!ret) return 0;
    
    memcpy(ret, ann, size);
    
    int * nodelist = (int*)malloc(sizeof(int)*ret->hidden_layers);
    memcpy(nodelist, ret->hidden_nodes, sizeof(int)*ret->hidden_layers);
    ret->hidden_nodes = nodelist;
    
    fscanf(in, "%d", &outputs);

    // Set pointers.
    ret->weight = (GENANN_NUMBER_TYPE*)((char*)ret + sizeof(genann));
    ret->output = ret->weight + ret->total_weights;
    ret->delta = ret->output + ret->total_neurons;

    return ret;
}


void genann_write(genann const *ann, FILE *out) {
    fprintf(out, "%d %d %d %d", ann->inputs, ann->hidden_layers);
    for(int i = 0; i < ann->hidden_layers; i++)
        fprintf(out, " %d", ann->hidden_nodes[i]);
    fprintf(out, " %d", ann->outputs);
    
    int i;
    for (i = 0; i < ann->total_weights; ++i) {
        fprintf(out, " %.20e", ann->weight[i]);
    }
}

*/
