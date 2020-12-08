
# Disable default rebuild Makefile rule to avoid Makeile.o match
MAKEFLAGS += -r

CXX ?= aarch64-xilinx-linux-g++
#CFLAGS += -O3 -g -Wall -Wpointer-arith -std=c++14 -ffast-math -mcpu=cortex-a53
CFLAGS += -g -Wall -std=c++14 -mcpu=cortex-a53
LDFLAGS += -L./ -ln2cube -lpthread -lopencv_core -lopencv_imgproc -lopencv_videoio -lopencv_imgcodecs -lopencv_highgui
CFLAGS += -fdiagnostics-color=always

SRC = ./src
COMPILE_RESULTS = compile_results
BUILD = build

VPATH = $(SRC)
CPP_FILES = $(wildcard $(SRC)/*.cpp)
OBJ = $(patsubst $(SRC)/%.cpp, $(BUILD)/%.o, $(CPP_FILES))

TARGETS = zcu104.tsm_resnet50_kinetics_8f ultra96v2.tsm_resnet50_kinetics_8f
.PHONY: all clean copy $(TARGETS)

all : $(TARGETS)

$(TARGETS) : SUBDIR = $(patsubst %.tsm_resnet50_kinetics_8f,%,$@)
$(TARGETS) : ELF = $(shell find $(SUBDIR)/$(COMPILE_RESULTS) -name *.elf)
$(TARGETS) : %.tsm_resnet50_kinetics_8f : $(OBJ)
	mkdir -p $(SUBDIR)/$(BUILD)
	$(CXX) $(CFLAGS) $^ $(ELF) -o $*/tsm_resnet50_kinetics_8f $(LDFLAGS)

%.copy : tsm_resnet50_kinetics_8f
	scp ./tsm_resnet50_kinetics_8f %(dir %@):~/tsm_resnet50_kinetics_8f/


$(BUILD)/%.o : %.cpp
	$(CXX) -c $(CFLAGS) $< -o $@

clean : 
	$(RM) -rf $(BUILD)
	$(RM) tsm_resnet50_kinetics_8f
