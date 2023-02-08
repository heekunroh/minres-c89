CC :=gcc
CFLAGS := -std=c89 -lm
SOURCES :=$(wildcard *.c)
EXECUTABLE := minrestest.out
OUT_DIR := build
TARGETS=$(OUT_DIR)/$(EXECUTABLE)
OBJECTS=$(patsubst %.c, %.o, $(SOURCES))

all:$(TARGETS)

$(TARGETS):$(OBJECTS) | $(OUT_DIR)
		$(CC) $(OBJECTS) -o $@ $(CFLAGS) 
		rm $(OBJECTS)

%.o : %.c 
		$(CC) $< -c -o $@ $(CFLAGS) 

$(OUT_DIR):
		mkdir -p $(OUT_DIR)

clean:
		rm $(OBJECTS) $(TARGETS)