# Schorle
Schorle is an implementation of the [Ozaki I scheme](https://www.r-ccs.riken.jp/assets/uploads/2025/12/FugakuNEXT_4th_ApplicationSeminar.pdf).
It leverages fast int8 x int8 -> int32 multiplication available in the new Mac devices (starting with M5).


# Description
Ozaki I scheme is based on the split of float64 matrices of certain size into `S` matrices of the same size but in 
int8 data type.
Later, utilizing fast int8 multiplication they can be multiplied and summed up in `S(S-1)/2` operations. 
On the modern accelerator frameworks this type of multiplication can be faster (while sacrificing some levels of accuracy).

# Building locally & benchmarking
In order to locally build the code cmake can be used (requires xcode):

"""bash
cmake -S . -B build
cmake --build build
"""
Two binaries will be created, `bench` and `profile-kernels`. First one measures the performance 
and compares with native accelerate framework for baseline float64 x float64 multiplication. The second one 
profiles each individual kernel for different values of `S`.


# Roadmap
[ ] Ozaki II scheme
