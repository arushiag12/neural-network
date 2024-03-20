NVCC = nvcc
NVCCFLAGS = -O3 -arch=sm_70 -lm -lz -lcublas

all: nn

nn: nn.cu
	$(NVCC) $(NVCCFLAGS) -o nn.out nn.cu

.PHONY: clean

clean:
	rm -f nn.out