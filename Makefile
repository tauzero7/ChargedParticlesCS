
CUDA_DIR     = /usr/local/cuda-8.0
FREETYPE_DIR = /usr/include/freetype2

COMPUTE_PRECISION  = float
COMPUTE_CAPABILITY = compute_20


DEBUGGING = yes
#DEBUGGING = no

## CUDA works only with gcc version <= 4.4
CC        = gcc
CPP       = g++
NVCCFLAGS = --compiler-bindir /usr/bin

#CFLAGS += -DFBO_TEXTURE_SIZE=512
CFLAGS += -DFBO_TEXTURE_SIZE=2048


## ----------- Do not change ----------------

TOP_DIR      = $(PWD)
SRC_DIR      = $(TOP_DIR)/src
CU_DIR       = $(TOP_DIR)/cuda
GL3W_DIR     = $(TOP_DIR)/gl3w

EXEC  =  ChargedParticlesCS
MAIN  =  main.cpp

ifeq ($(shell uname -m), x86_64)
        BITS := 64
else
        BITS := 
endif


CUDA_LIB_DIR = $(CUDA_DIR)/lib$(BITS)
SCENES_DIR = scenes/
BUILD_DIR  = compiled/

Q = #@

NVCC      = $(CUDA_DIR)/bin/nvcc  --ptxas-options="-v" -arch=$(COMPUTE_CAPABILITY)

CFLAGS   += -Wall -Wno-comment 
LFLAGS   += -lglut -lGL -ldl \
            -L$(CUDA_LIB_DIR) -lcudart -Wl,-rpath $(CUDA_LIB_DIR) \
            -L$(FREETYP_DIR)/lib -lfreetype

ifeq ($(COMPUTE_PRECISION),double) 
         CFLAGS  += -DUSE_DOUBLE
      NVCCFLAGS  += -DUSE_DOUBLE
endif

ifeq ($(DEBUGGING),yes)
         CFLAGS  += -g
endif	 

INCLUDES += -I. -I$(TOP_DIR) -I$(SRC_DIR) -I$(CU_DIR) \
            -I$(TOP_DIR)/glm -I$(TOP_DIR)/gl3w \
            -I$(CUDA_DIR)/include \
            -I$(FREETYPE_DIR)

C_SOURCES   +=  $(GL3W_DIR)/gl3w.c

CPP_SOURCES += Camera.cpp \
               GLShader.cpp \
               RenderText.cpp \
               utils.cpp
               
CU_SOURCES  += curvedSurfaceCode.cu     

OBJS     = $(addprefix $(BUILD_DIR), $(notdir $(patsubst %.cpp, %.o, $(CPP_SOURCES))))
C_OBJS   = $(addprefix $(BUILD_DIR), $(notdir $(patsubst %.c, %.o, $(C_SOURCES))))
CU_OBJS  = $(addprefix $(BUILD_DIR), $(notdir $(patsubst %.cu, %.cu_o, $(CU_SOURCES))))
M_OBJS   = $(addprefix $(BUILD_DIR), $(notdir $(patsubst %.cpp, %.o, $(MAIN))))
SC_FILES = $(wildcard $(SCENES_DIR)/*.inl)
O_FILES  = $(wildcard $(CU_DIR)/*.inl)


$(EXEC) : $(OBJS) $(C_OBJS) $(CU_OBJS) $(M_OBJS) 
	@echo 'Link ' $(EXEC)
	$(Q)$(CPP) -o $(EXEC) $(OBJS) $(C_OBJS) $(M_OBJS) $(CU_OBJS) $(CFLAGS) $(LIBS) $(LFLAGS)


.PHONY: all clean extra show

all: $(OBJS) $(C_OBJS) $(CU_OBJS) $(M_OBJS)

$(M_OBJS): $(BUILD_DIR)%.o: %.cpp  $(SC_FILES)  $(O_FILES)  main.inl
	@echo 'Compile '$(@F)
	$(Q)$(CPP)  -c $(CFLAGS) $(INCLUDES) $< -o $@

$(OBJS): $(BUILD_DIR)%.o: $(SRC_DIR)/%.cpp $(SC_FILES) | $(BUILD_DIR)
	@echo 'Compile '$(@F)
	$(Q)$(CPP) -c $(CFLAGS) $(INCLUDES) $< -o $@

$(C_OBJS): $(BUILD_DIR)%.o: $(GL3W_DIR)/%.c
	@echo 'Compile '$(@F)
	$(Q)$(CC) -c $(CFLAGS) $(INCLUDES) $< -o $@

$(CU_OBJS): $(BUILD_DIR)%.cu_o: $(CU_DIR)/%.cu  $(O_FILES) | $(BUILD_DIR)
	@echo 'Compile '$(@F)
	$(Q)$(NVCC) -c $(NVCCFLAGS) $(INCLUDES) $< -o $@

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

clean:
	rm -f $(EXEC) $(OBJS) $(C_OBJS) $(CU_OBJS) $(M_OBJS) *.o *~ core tags

extra:
	@echo $(PWD)
	
show:
	@echo $(OBJS)

