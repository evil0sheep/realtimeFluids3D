CC=nvcc
LD=nvcc
CFLAGS= -O3 -c  -lGL -lglut -DGL_GLEXT_PROTOTYPES -lGLU 
LDFLAGS= -O3   -lGL -lglut -DGL_GLEXT_PROTOTYPES -lGLU  
CUDAFLAGS= -O3 -c --profile -arch=sm_21 -Xptxas -dlcm=ca 

ALL= callbacksPBO.o kernelPBO.o simpleGLmain.o simplePBO.o

all= $(ALL) fluids

RT:	$(ALL)
	$(CC) $(LDFLAGS) $(ALL) -o fluids

callbacksPBO.o:	callbacksPBO.cpp
	$(CC) $(CFLAGS) -o $@ $<

kernelPBO.o:	kernelPBO.cu
	$(CC) $(CUDAFLAGS) -o $@ $<

simpleGLmain.o:	simpleGLmain.cpp
	$(CC) $(CFLAGS) -o $@ $<

simplePBO.o: simplePBO.cpp
	$(CC) $(CFLAGS) -o $@ $<

clean:
	rm -rf core* *.o *.gch $(ALL) junk*

