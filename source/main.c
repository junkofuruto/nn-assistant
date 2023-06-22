#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>

#include "data.h"

#define EPS  (1e-1f)
#define RATE (1e-3f)

float rand_float() {
    return (float) rand() / (float) RAND_MAX;
}

float sigmoid(float input) {
    return 1.0f / (1.0f + exp(-input));
}

float cost(float w1, float w2 , float b) {
    float result = 0.0f;
    for (size_t i = 0; i < train_count; ++i) {
        float y = sigmoid(train_x[i][0] * w1 + train_x[i][1] * w2 + b);
        float d = y - train_y[i][0];
        result += d * d;
    }
    return result / train_count;
}

void forward(float *w1, float *w2, float *b) {
    float c = cost(*w1, *w2, *b); 
    float dw1 = (cost(*w1 + EPS, *w2, *b) - c) / EPS;
    float dw2 = (cost(*w1, *w2 + EPS, *b) - c) / EPS;
    float db = (cost(*w1, *w2, *b + EPS) - c) / EPS;
    *w1 -= RATE * dw1;
    *w2 -= RATE * dw2;
    *b -= RATE * db;
}

int main(void) {
    srand(time(0));
    float w1 = rand_float() * 10.0f - 5.0f;
    float w2 = rand_float() * 10.0f - 5.0f;
    float b = rand_float() * 5.0f - 2.5f;

    for (size_t i = 0; i <= 10000000; ++i) {
        forward(&w1, &w2, &b);
        if (i % 1000000 == 0) {
            printf("| W1: %6.3f | W2: %6.3f | B: %6.3f >> (%5.3f)\n", w1, w2, b, cost(w1, w2, b));
        }
    }
    for (size_t i = 0; i < train_count; ++i) {
        printf("\n%.0f & %.0f = %f", train_x[i][0], train_x[i][1], sigmoid(train_x[i][0] * w1 + train_x[i][1] * w2 + b));
    }

    return 0;
}