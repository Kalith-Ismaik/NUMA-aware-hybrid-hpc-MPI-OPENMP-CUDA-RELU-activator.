# Define compilers and flags
FC = mpif90           # MPI Fortran compiler
CUFC = nvfortran      # NVIDIA Fortran compiler for CUDA

# Flags for OpenMP, CUDA, and MPI
FFLAGS = -mp -cuda -g -O0
CUFLAGS = -Mcuda -Minfo -g -O0

# Source and Object Files
SRCS = TypeX.f90 RandGen.f90 Main.f90
CUFSRCS = RandGenCUDA.cuf
OBJS = $(SRCS:.f90=.o) $(CUFSRCS:.cuf=.o)

TARGET = NeuroFort.x

# Rule to compile CUDA Fortran files
%.o: %.cuf
	$(CUFC) $(CUFLAGS) -c $< -o $@

# Rule to link the final executable
$(TARGET): $(OBJS)
	$(FC) $(FFLAGS) -o $@ $(OBJS) -lcudart

# Rule to compile Fortran files
%.o: %.f90
	$(FC) $(FFLAGS) -c $< -o $@

# Clean the build
.PHONY: clean
clean:
	rm -f $(OBJS) $(TARGET)
