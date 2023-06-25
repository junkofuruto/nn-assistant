#define __NFRAMEWORK_IMPL__

#include "framework/framework.h"

int main(void) {
    nf_matrix mat = nf_mat_alloc(1, 2);
    nf_mat_print(mat);

    return 0;
}