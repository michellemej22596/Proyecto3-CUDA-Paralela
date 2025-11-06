NVCC = nvcc

# Source files
SRCS = ../src/main.cu ../src/hough.cu ../src/image_utils.cpp

# Output binary
TARGET = hough

# Compiler flags
NVCC_FLAGS = -O2 -I../src

all: $(TARGET)

$(TARGET): $(SRCS)
	$(NVCC) $(SRCS) -o $(TARGET) $(NVCC_FLAGS)

clean:
	rm -f $(TARGET)
