Developed a loop interchange pass in MLIR driven by an analytical cost model that
optimizes for locality (spatial, temporal - both self and group) and parallelism
for multicores (so as to minimize the frequence of synchronization).

References:
[1] Improving Data Locality with Loop Transformations, https://scholarworks.umass.edu/cgi/viewcontent.cgi?article=1139&context=cs_faculty_pubs
[2] Improving locality and parallelism in nested loops, http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.79.9865&rep=rep1&type=pdf
[3] Optimizations for Parallelism, Locality and More, https://www.csa.iisc.ac.in/~udayb/slides/uday-polyhedral-opt.pdf
[4] Chapter 11, "Optimizing for Parallelism and Locality", Aho,Lam,Sethi,Ullman, "Compilers Principles, Techniques, & Tools", Second edition.

I have taken help from above references to come up with this algorithm.
Algorithm:
    1. For each loop l, compute groups of array references.
        Two references ref1 and ref2 belong to same group with respect to loop l if:
            a. they refer to the same array and has exactly same access function.
            b. or, they refer to the same array and differ only in lth dimension by atmost cacheLineSize.
            c. or, they refer to the same array and differ by at most cacheLineSize in the last dimension.
        a and b corresponds to group temporal reuse, c corresponds to group spatial reuse.
        After this step, we have a set of reference groups for each loop l.
    2. Find valid loop permutations.
    3. For each loop l, compute number of memory accesses made when l is the innermost loop.
        For innermost Loop, number of memory accesses = {
            1 , when reference is loop invariant;
            (tripCount/cacheLineSize), when reference has spatial reuse for loop l;
            (tripCount), otherwise
        }
        a. For each reference group, choose a reference R.
            i. if R has spatial reuse on loop l, add (tripCount/cacheLineSize) to number of memory accesses.
            ii. else, if R has temporal reuse on loop l, add 1 to number of memory accesses.
            iii. else, (add tripCount) to number of memory accesses.
        b. Multiple the result of number of memory accesses by the tripCount of all the remaining loops.
    4. Choose the loop with least number of memory accesses as the innermost loop, say it is L.
    5. Find the valid loop permutation which has loop L as the innermost loop.
    6. Find the loops which are parallel, does not carry any loop dependence.
    7. For each loop permutation found in step 5, calculate the cost of synchronization.
        Cost of synchronization is calculated for each parallel loop.
        For a loop, synchronization cost = product of tripCounts of all loops which are at outer positions to this loop.
    8. Choose the permutation with the least synchronization cost as the best permutation.

Assumptions and intution behind the algorithm:
1. Chosen only the best innermost loop because in general when innermost loop has large number of iterations and accesses
large amount of data, only reuse within innermost loop can be exploited. Or only innermost loop is included in the localized vector space.
2. Reuse in multiple directions cannot be exploited by loop interchange.
3. Cost Model includes number of memory accesses (loopCost) and synchronization cost. loopCost is calculated for every 
possible innermost loop. Best loop is chosen as the innermost loop. loopCost calculation needs only cacheLineSize information.
4. LoopCost is least for that innermost loop which converts most reuses into locality.
5. Synchronization cost is least for the loop permutation which has the most coarse-grained parallelism.

Test cases not passed:


How to build and run:
1. git clone https://github.com/llvm/llvm-project
2. git checkout 4ba7ae85da314e3f14a5bf26bf80dc29410b3e71
3. git apply --check patch.file
3. git apply patch (see correct command)
4. Build and run (https://mlir.llvm.org/getting_started/)
	mkdir llvm-project/build
	cd llvm-project/build
	cmake -G Ninja ../llvm \
	   -DLLVM_ENABLE_PROJECTS=mlir \
	   -DLLVM_BUILD_EXAMPLES=ON \
	   -DLLVM_TARGETS_TO_BUILD="X86;NVPTX;AMDGPU" \
	   -DCMAKE_BUILD_TYPE=Release \
	   -DLLVM_ENABLE_ASSERTIONS=ON \
	#  -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DLLVM_ENABLE_LLD=ON

	cmake --build . --target check-mlir
	
	bin/mlir-opt interchange.mlir -affine-interchange-loop
 	bin/mlir-opt ../../interchange4.mlir --affine-loop-interchange | bin/FileCheck ../../interchange4.mlir


How to create patch:

- git add .
- git commit -m "msg"
- git format-patch HEAD~
