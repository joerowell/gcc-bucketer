// Force use of GCC vectors
#ifdef HAVE_AVX2
#undef HAVE_AVX2
#endif

#include "simd.t.cpp"

