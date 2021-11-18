# gcc-bucketer

## What's this?
This repository contains a slightly modified implementation of the [AVX2 Bucketer](https://github.com/lducas/AVX2-BDGL-bucketer) that uses
GCC's vector intrinsics as opposed to Intel's vector intrinsics. This allows the bucketer to be used on platforms that do not support AVX2.

This repository belongs to a much larger research project into lattice sieving across multiple platforms.

## Acknowledgements
The code in this repository was initially taken from [AVX2 Bucketer](https://github.com/lducas/AVX2-BDGL-bucketer), and all of the really good ideas from this work come from that repository. We are deeply grateful to the authors of that project for sharing their code with us at an early stage.

## How to use:
We refer the reader to the excellent readme at [AVX2 Bucketer](https://github.com/lducas/AVX2-BDGL-bucketer) for a description of the interface to the bucketer itself. Here we simply describe the added interface as part of this project.

The main namespace for this project is known as ```Simd```. This namespace contains a variety of both low-level intrinsic functions for operations on vectors and for higher-level manipulations on these vectors. 

As GCC is likely to generate poor object code (compared to hand-written intrinsics) we provide the ability to switch between using AVX2 instructions and the GCC vector code. To use AVX2 natively, define ```HAVE_AVX2``` using the C preprocessor. 

## How to test:
This repository contains a series of test suites. Note that in order to run these tests you will need both ```CMAKE``` and ```googleTest```.


### Building tests:
```
mkdir build
cd build
cmake ../
make
```

This will produce a series of files that can be used for running tests.

- Unit tests are provided in src/simd.t.cpp. 
- If you want to test the GCC vectors, run SimdGCCTests. If you want to use AVX2, run SimdIntelTests.
- Minimal microbenchmarks are provided in ```src/simd.b.cpp```. Running these requires the use of ```google benchmark```. Assuming you have ```google benchmark``` you can run the code by executing the ```SimdBench``` program.
- You can also use the ```test_lsh``` files for various statistics and dimensions. Again, the original code comes from the AVX2 Bucketer.










