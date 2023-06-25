build/assistant.exe: source/main.c source/framework/framework.h
	gcc -Wall -Wextra -lm -o build/assistant.exe source/main.c