#makefile

start:
	rm bin/*.*

serial:
	echo "serial"
#	${LLVM}/bin/clang -mepi -Wall --target=riscv64-unknown-elf -march=rv64g -O2 -c -o src/axpy_serial.o src/axpy.c
#	${LLVM}/bin/clang -mepi -Wall --target=riscv64-unknown-elf -march=rv64g -O2 -c -o src/main_serial.o src/main.c
#	${LLVM}/bin/clang -mepi -Wall --target=riscv64-unknown-elf -march=rv64g -O2 -c -o src/utils_serial.o src/utils.c
#	${LLVM}/bin/clang -mepi -Wall --target=riscv64-unknown-elf -march=rv64g -O2 -o bin/axpy_serial.exe src/*.o -lm
#	rm src/*.o
#	${LLVM}/bin/llvm-objdump  -mattr=+m,+f,+d,+a,+c,+epi -ds  bin/axpy_serial.exe > bin/axpy_serial.dump

vector:
	echo "vector"
	${GCC_RISCV}/bin/riscv64-unknown-elf-g++ -std=c++17 -mcpu=c920 -mabi=lp64d -O3 -mtune=c920  src/main.c -o bin/axpy
	# ${GCC_RISCV}/bin/riscv64-unknown-elf-g++ -std=c++17 -mcpu=c920 -mabi=lp64d -O3 -mtune=c920  src/main.c -S -o bin/axpy.s

clean:
	rm bin/*.dump; \
	rm bin/*.exe;