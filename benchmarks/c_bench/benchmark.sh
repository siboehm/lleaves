#!/usr/bin/env bash

model_file='"../../tests/models/single_tree/model.txt"'

# clean up cache
if [[ -f "llvm.o" ]]; then
  rm "llvm.o"
fi

python -c "import lleaves; lleaves.Model(model_file=${model_file}).compile(\"llvm.o\")"

g++ c_bench.cpp -c -o c_bench.o

# fails with:
# /usr/bin/ld: llvm.o: warning: relocation in read-only section `.text'
# /usr/bin/ld: c_bench.o: in function `main':
# c_bench.cpp:(.text+0x19c): undefined reference to `forest_root(double*, double*, int, int)'
# /usr/bin/ld: warning: creating DT_TEXTREL in a PIE
# collect2: error: ld returned 1 exit status
g++ c_bench.o llvm.o -o c_bench -L/usr/local/lib/libcnpy.so -lcnpy -lz --std=c++11 -lstdc++
