// We'll only enable AVX2 if the __AVX2__
// macro is set.
#if defined(__AVX2__)
#define HAVE_AVX2 1
#else
#undef HAVE_AVX2
#endif
#include "simd.t.cpp"
