#include "fht_lsh_old.h"
#include "simd.hpp"
#include <benchmark/benchmark.h>
#include <immintrin.h>

static void BM_hadd_epi16_native(benchmark::State &state)
{

  __m256i a;
  __m256i b;

  for (auto _ : state)
  {
    auto c = _mm256_hadd_epi16(a, b);
    benchmark::DoNotOptimize(c);
  }
}

static void BM_hadd_epi16_emu(benchmark::State &state)
{
  // Generate two random vectors
  Simd::Vec16s a;
  Simd::Vec16s b;

  for (auto _ : state)
  {
    auto c = Simd::m256_hadd_epi16(a, b);
    benchmark::DoNotOptimize(c);
  }
}

static void BM_permute_4x64_epi64_native(benchmark::State &state)
{
  __m256i a;
  for (auto _ : state)
  {
    auto c = _mm256_permute4x64_epi64(a, 0b01001110);
    benchmark::DoNotOptimize(c);
  }
}

static void BM_permute_4x64_epi64_emu(benchmark::State &state)
{
  Simd::Vec16s a;
  for (auto _ : state)
  {
    auto c = Simd::m256_permute4x64_epi64<0b01001110>(a);
    benchmark::DoNotOptimize(c);
  }
}

static void BM_permute_4x64_epi64_emu_hadamard(benchmark::State &state)
{
  Simd::Vec16s a;

  for (auto _ : state)
  {
    auto c = Simd::m256_permute4x64_epi64_for_hadamard(a);
    benchmark::DoNotOptimize(c);
  }
}

static void BM_sign_epi16_native(benchmark::State &state)
{
  __m256i a;

  for (auto _ : state)
  {
    // NB this is FastHadamardLsh's mask
    auto c = _mm256_sign_epi16(a, sign_mask_2);
    benchmark::DoNotOptimize(c);
  }
}

static void BM_sign_epi16_emu(benchmark::State &state)
{
  Simd::Vec16s a;

  for (auto _ : state)
  {
    auto c = Simd::m256_sign_epi16(a, Simd::sign_mask_2);
    benchmark::DoNotOptimize(c);
  }
}

static void BM_hadamard16_epi16_native(benchmark::State &state)
{
  __m256i a, res;

  for (auto _ : state)
  {
    FastHadamardLSH::m256_hadamard16_epi16(a, res);
    benchmark::DoNotOptimize(res);
  }
}

static void BM_hadamard16_epi16_emu(benchmark::State &state)
{
  Simd::Vec16s a, b;
  for (auto _ : state)
  {
    Simd::m256_hadamard16_epi16(a, b);
    benchmark::DoNotOptimize(b);
  }
}

static void BM_hadamard32_epi16_native(benchmark::State &state)
{
  __m256i a, b, res1, res2;

  for (auto _ : state)
  {
    FastHadamardLSH::m256_hadamard32_epi16(a, b, res1, res2);
    benchmark::DoNotOptimize(res1);
    benchmark::DoNotOptimize(res2);
  }
}

static void BM_hadamard32_epi16_emu(benchmark::State &state)
{
  Simd::Vec16s a, b, res1, res2;
  for (auto _ : state)
  {
    Simd::m256_hadamard32_epi16(a, b, res1, res2);
    benchmark::DoNotOptimize(res1);
    benchmark::DoNotOptimize(res2);
  }
}

static void BM_m256_mix_native(benchmark::State &state)
{
  __m256i a, b, c;
  for (auto _ : state)
  {
    FastHadamardLSH::m256_mix(a, b, c);
    benchmark::DoNotOptimize(a);
    benchmark::DoNotOptimize(b);
  }
}

static void BM_m256_mix_emu(benchmark::State &state)
{
  Simd::Vec16s a, b, res1;
  for (auto _ : state)
  {
    Simd::m256_mix(a, b, res1);
    benchmark::DoNotOptimize(a);
    benchmark::DoNotOptimize(b);
  }
}

static void BM_broadcastsi128_si256_native(benchmark::State &state)
{
  __m256i a;
  __m128i b;

  for (auto _ : state)
  {
    a = _mm256_broadcastsi128_si256(b);
    benchmark::DoNotOptimize(a);
  }
}

static void BM_broadcastsi128_si256_emu(benchmark::State &state)
{
  Simd::Vec16s a;
  Simd::Vec8s b;

  for (auto _ : state)
  {
    a = Simd::m256_broadcastsi128_si256(b);
    benchmark::DoNotOptimize(a);
  }
}

BENCHMARK(BM_hadd_epi16_native);
BENCHMARK(BM_hadd_epi16_emu);
BENCHMARK(BM_permute_4x64_epi64_native);
BENCHMARK(BM_permute_4x64_epi64_emu);
BENCHMARK(BM_permute_4x64_epi64_emu_hadamard);
BENCHMARK(BM_sign_epi16_native);
BENCHMARK(BM_sign_epi16_emu);
BENCHMARK(BM_hadamard16_epi16_native);
BENCHMARK(BM_hadamard16_epi16_emu);
BENCHMARK(BM_hadamard32_epi16_native);
BENCHMARK(BM_hadamard32_epi16_emu);
BENCHMARK(BM_m256_mix_native);
BENCHMARK(BM_m256_mix_emu);
BENCHMARK(BM_broadcastsi128_si256_native);
BENCHMARK(BM_broadcastsi128_si256_emu);
BENCHMARK_MAIN();
