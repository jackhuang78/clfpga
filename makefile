.PHONY: amd altera nvidia clean

amd:
	make -f makefile.amd

altera:
	make -f makefile.altera

nvidia:
	make -f makefile.nvidia

clean:
	make -f makefile.amd clean
	make -f makefile.altera clean
	make -f makefile.nvidia clean
