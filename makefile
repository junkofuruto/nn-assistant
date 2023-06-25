build/assistant.exe: source/main.c source/nnf/nf_matrix.h
	gcc -Wall -Wextra -lm -o build/assistant.exe source/main.c