.PHONY: amd altera nvidia clean

amd: 
	make -f makefile.sub ODIR="amd" CLINC="-I$(AMDAPPSDKROOT)/include/" CLLIB="-L$(AMDAPPSDKROOT)/lib/x86_64 -lOpenCL"

altera:
	make -f makefile.sub ODIR="altera" CLINC="-I/opt/altera/13.0/AOCL/host/include" CLLIB="-L/opt/altera/13.0/AOCL/linux64/lib -lalterahalmmd -lalterammdpcie -lpkg_editor -lalteracl -lrt -lstdc++ -L/opt/altera/13.0/AOCL/host/linux64/lib -lelf -Wl,-rpath=/opt/altera/13.0/AOCL/linux64/lib -Wl,-rpath=/opt/altera/13.0/AOCL/host/linux64/lib -DLINUX -DALTERA"

nvidia:
	make -f makefile.nvidia

clean:
	make -f makefile.amd clean
	make -f makefile.altera clean
	make -f makefile.nvidia clean
