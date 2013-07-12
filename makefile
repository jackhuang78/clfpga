.PHONY: amd altera nvidia

amd:
	make -f makefile.amd

altera:
	make -f makefile.altera

nvidia:
	make -f makefile.nvidia
