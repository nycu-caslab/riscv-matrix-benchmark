SHELL=/bin/bash

top_dir = ../../..
inc_dir = $(top_dir)/include
src_dir = $(top_dir)/src
includes = -I$(inc_dir)/env -I$(inc_dir)/common -I$(src_dir)
defines = $(DEFS)

# simulators
# supported: spike gem5 vcs
SIM ?= spike
# toolchains
# supported: llvm gnu
TC ?= llvm
TC_DIR:=matrix-riscv-llvm

MAIN?=

SPIKE := spike
SPIKE_ARGS :=

GEM5 ?= /home/anonymouself/matrix-gem5
GEM5_OPTS :=
GEM5_ARGS :=
GEM5_OUTLOG := 
# GEM5_DEBUG_FLAGS :=--debug-flags=MinorMatrixExecute,MinorExecute,MinorCPU,Decode,MDecode

# GEM5_DEBUG_FLAGS :=--debug-flags=Minor,Activity,RiscvMisc#,Quiesce,Branch,Fetch,Decode

# GEM5_DEBUG_FLAGS ?=--debug-flags=MinorStreamTable,MinorExecute,MinorScoreboard,MinorMem,Activity,RiscvMisc
# GEM5_DEBUG_FLAGS ?=--debug-flags=Exec,RiscvMisc,Registers,MinorMatrixExecute,MinorExecute,MinorMem
# GEM5_DEBUG_FLAGS ?=--debug-flags=MinorMem,MinorExecute,RiscvMisc
GEM5_DEBUG_FLAGS :=--debug-flags=Exec
# GEM5_DEBUG_FLAGS :=--debug-flags=MinorTrace
# GEM5_DEBUG_FLAGS ?=--debug-flags=MinorStreamTable,MinorExecute,MinorScoreboard,MinorMem,Activity,RiscvMisc,MinorMatrixExecute
# GEM5_DEBUG_FLAGS :=
# GEM5_DEBUG_FLAGS :=--debug-flags=MinorExecute,MinorScoreboard,MinorMem,RiscvMisc
# GEM5_DEBUG_FLAGS :=--debug-flags=Exec,RiscvMisc,Registers,MinorMatrixExecute,MinorExecute
# GEM5_DEBUG_FLAGS :=
# MINOR_FU_TYPE:=MinorMatrixTimingFUPool
# MINOR_FU_TYPE:=MinorMatrix2MU1MemTimingFUPool
# MINOR_FU_TYPE:=MinorMatrix2MU2MemTimingFUPool
# MINOR_FU_TYPE:=MinorMatrix2MU8MemTimingFUPool
MINOR_OPTS:=
SIMV ?= $(top_dir)/../chipyard/sims/vcs/simv-chipyard-StcBoomConfig-debug
SIMV_ARGS := +fsdbfile=test.fsdb
SIMV_POST := 

SYSROOT:= --sysroot=/opt/gcc-riscv-linux-rvv/riscv64-unknown-elf --gcc-toolchain=/opt/gcc-riscv-linux-rvv/

PK := pk

ifeq (x$(SIM), xspike)
	SIM_CMD ?= \
		$(SPIKE) --isa=rv64gcv_zfh --varch=vlen:1024,elen:64,slen:1024,mlen:65536 \
			+signature=spike.sig +signature-granularity=32 
	defines += -D__SPIKE__
else ifeq (x$(SIM), xgem5)
	SIM_CMD ?= $(top_dir)/scripts/gem5.sh \
		$(GEM5)/build/RISCV/gem5.debug $(GEM5_DEBUG_FLAGS) --debug-file=trace.out --listener-mode=off $(GEM5_OPTS) \
		   ${GEM5}/configs/example/se.py --num-cpus 1 \
        --cpu-type MinorCPU \
				$(MINOR_OPTS) \
				--caches \
				--l1d_size 128kB \
				--cmd $(MAIN)/test.elf \
					--output gem5.sig \
					2>&1 | tee $(GEM5_OUTLOG)
	defines += -D__GEM5__
else ifeq (x$(SIM), xvcs)
	SIM_CMD ?= $(top_dir)/scripts/vcs.sh $(SIMV) +signature=vcs.sig +signature-granularity=32 +permissive +loadmem=test.hex +loadmem_addr=80000000 $(SIMV_ARGS) +permissive-off 
	SIMV_POST := </dev/null 2> >(spike-dasm > vcs.out) | tee vcs.log
else ifeq (x$(SIM), xgem5gdb)
	SIM_CMD ?= $(top_dir)/scripts/gem5.sh \
		gdb $(GEM5)/build/RISCV/gem5.debug -x $(MAIN)/gem5.gdb
	defines += -D__GEM5__
endif


# toolchain

PREFIX ?= /opt/gcc-riscv-linux-rvv/bin/riscv64-unknown-elf-
OBJDUMP := $(PREFIX)objdump

ifneq (x$(TC), xllvm)
	CC := $(PREFIX)gcc
	LINK := $(PREFIX)gcc
	CFLAGS := -g -march=rv64gv0p10zfh0p1  -DPREALLOCATE=1 -mcmodel=medany -static -ffast-math -fno-common -fno-builtin-printf -fno-tree-loop-distribute-patterns $(includes) $(defines)
else
	CC := /opt/$(TC_DIR)/bin/clang $(SYSROOT) --target=riscv64-unknown-elf  -march=rv64gv0p10zfh0p1 -menable-experimental-extensions 
	LINK :=/opt/$(TC_DIR)/bin/clang $(SYSROOT) --target=riscv64-unknown-elf  -march=rv64gv0p10zfh0p1 -menable-experimental-extensions
	CFLAGS := -g -mcmodel=medany -mllvm -ffast-math -fno-common -fno-builtin-printf $(includes) $(defines)
endif

LDFLAGS :=-static

target_elf = test.elf
target_dump = test.dump
target_map = test.map

objects += test.o

all: $(target_elf)

syscalls.o: $(inc_dir)/common/syscalls.c
	$(CC) $(CFLAGS) -c -o $@ $<

crt.o: $(inc_dir)/common/crt.S
	$(CC) $(CFLAGS) -c -o $@ $<

sgemm.o: $(src_dir)/sgemm.S
	$(CC) $(CFLAGS) -c -o $@ $<

sgemm_4.o: $(src_dir)/sgemm_4.S
	$(CC) $(CFLAGS) -c -o $@ $<

sgemm_1.o: $(src_dir)/sgemm_1.S
	$(CC) $(CFLAGS) -c -o $@ $<

sgemm_%.o: $(src_dir)/sgemm_%.S
	$(CC) $(CFLAGS) -c -o $@ $<

.c.o:
	$(CC) $(CFLAGS) -c -o $@ $<
	
$(target_elf): $(objects)
	$(LINK) -o $(target_elf) $^ $(LDFLAGS)

$(target_map): $(target_elf)
	$(PREFIX)readelf -s -W $< > $@

run: $(target_elf) $(target_map)
	$(SIM_CMD) $(SIMV_POST)

dump: $(target_dump)

$(target_dump): $(target_elf)
	$(OBJDUMP) -S $(target_elf) > $(target_dump)
	
clean:
	rm -f $(target_elf) $(objects) $(target_dump) $(target_map) *.sig

