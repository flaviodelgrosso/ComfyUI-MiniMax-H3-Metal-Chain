# Builds the persistent ComfyUI worker against a local h3.c checkout:
#   make -C native H3_DIR=/path/to/h3.c
# libh3.a is (re)built in that checkout by h3.c's own Makefile.
ifndef H3_DIR
$(error set H3_DIR to your h3.c checkout, e.g. make H3_DIR=$$HOME/h3.c)
endif

# zsh passes H3_DIR=~/... through unexpanded, so expand ~ here. Relative paths are relative to native/.
H3 := $(abspath $(patsubst ~%,$(HOME)%,$(H3_DIR)))
ifeq ($(wildcard $(H3)/h3.h),)
$(error no h3.c checkout at $(H3): H3_DIR must point to the folder containing h3.h)
endif
CC := clang
CFLAGS := -std=c11 -O2 -Wall -Wextra -Wpedantic -Wshadow -Wconversion \
	-Wno-sign-conversion -D_DARWIN_C_SOURCE
LDLIBS := -framework Foundation -framework Metal \
	-framework MetalPerformanceShaders -framework MetalPerformanceShadersGraph \
	-framework Accelerate -licucore -lm -lz

h3_comfy_worker: h3_comfy_worker.c $(H3)/libh3.a $(H3)/h3.h
	$(CC) $(CFLAGS) -I$(H3) -DH3_SOURCE_DIR='"$(H3)"' -o $@ $< $(H3)/libh3.a $(LDLIBS)

$(H3)/libh3.a: FORCE
	$(MAKE) -C $(H3) libh3.a

.PHONY: clean FORCE
FORCE:

clean:
	rm -f h3_comfy_worker
