#ifndef SIMD_INTEL_H
#error Do not include simd_intel.inl without simd_intel.h
#endif

#include <array>    // Needed for array
#include <cstring>  // Needed for memcpy/memcmp.
// If (for some unknown reason) this is prohibitive you can instead
// use __builtin_memcpy.

template <int pos> inline SimdIntel::SmallVecType SimdIntel::m128_slli_epi64(const SmallVecType a)
{
  return _mm_slli_epi64(a, pos);
}

inline SimdIntel::VecType SimdIntel::m256_loadu_si256(const int16_t *const a)
{
  return _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a));
}

inline void SimdIntel::m256_storeu_si256(int16_t *a, const VecType b)
{
  return _mm256_storeu_si256(reinterpret_cast<__m256i *>(a), b);
}

inline SimdIntel::SmallVecType SimdIntel::m128_set_epi64x(const int64_t e1, const int64_t e0)
{
  return _mm_set_epi64x(e1, e0);
}

inline SimdIntel::SmallVecType SimdIntel::m128_set_epi64x(const uint64_t e1, const uint64_t e0)
{
  return _mm_set_epi64x(e1, e0);
}

inline SimdIntel::VecType SimdIntel::m256_hadd_epi16(const SimdIntel::VecType a,
                                                     const SimdIntel::VecType b)
{
  return _mm256_hadd_epi16(a, b);
}

inline SimdIntel::SmallVecType SimdIntel::m128_add_epi64(const SmallVecType a, const SmallVecType b)
{
  return _mm_add_epi64(a, b);
}

inline SimdIntel::VecType SimdIntel::m256_add_epi16(const VecType a, const VecType b)
{
  return _mm256_add_epi16(a, b);
}

inline SimdIntel::VecType SimdIntel::m256_sub_epi16(const VecType a, const VecType b)
{
  return _mm256_sub_epi16(a, b);
}

inline SimdIntel::SmallVecType SimdIntel::m128_xor_si128(const SmallVecType a, const SmallVecType b)
{
  return _mm_xor_si128(a, b);
}

template <int pos> inline SimdIntel::SmallVecType SimdIntel::m128_srli_epi64(const SmallVecType a)
{
  return _mm_srli_si128(a, pos);
}

inline SimdIntel::VecType SimdIntel::m256_xor_si256(const VecType a, const VecType b)
{
  return _mm256_xor_si256(a, b);
}

inline SimdIntel::VecType SimdIntel::m256_and_si256(const VecType a, const VecType b)
{
  return _mm256_and_si256(a, b);
}

template <uint8_t mask> inline SimdIntel::VecType SimdIntel::m256_permute4x64_epi64(const VecType a)
{
  return _mm256_permute4x64_epi64(a, mask);
}

inline SimdIntel::VecType SimdIntel::m256_permute4x64_epi64_for_hadamard(const VecType a)
{
  return _mm256_permute4x64_epi64(a, 0b01001110);
}

inline int SimdIntel::m256_testz_si256(const VecType a, const VecType b)
{
  return _mm256_testz_si256(a, b);
}

inline SimdIntel::VecType SimdIntel::m256_abs_epi16(const VecType a) { return _mm256_abs_epi16(a); }

template <int pos> inline int64_t SimdIntel::m256_extract_epi64(const VecType a)
{
  static_assert(pos < 4, "Error: the requested index is too high.");
  return _mm256_extract_epi64(a, pos);
}

template <int pos> inline int64_t SimdIntel::m128_extract_epi64(const SmallVecType a)
{
  static_assert(pos < 2, "Error: the requested index is too high.");
  return _mm_extract_epi64(a, pos);
}

inline SimdIntel::VecType SimdIntel::m256_sign_epi16_ternary(const VecType a, const VecType mask)
{
  // Just use a regular sign operation here.
  return _mm256_sign_epi16(a, mask);
}

inline SimdIntel::VecType SimdIntel::m256_sign_epi16(const VecType a, const VecType mask)
{
  return _mm256_sign_epi16(a, mask);
}

inline SimdIntel::VecType SimdIntel::m256_slli_epi16(const VecType a, const int count)
{
  return _mm256_slli_epi16(a, count);
}

inline SimdIntel::VecType SimdIntel::m256_cmpgt_epi16(const VecType a, const VecType b)
{
  return _mm256_cmpgt_epi16(a, b);
}

inline SimdIntel::VecType SimdIntel::m256_broadcastsi128_si256(const SmallVecType in)
{
  return _mm256_broadcastsi128_si256(in);
}

inline SimdIntel::SmallVecType SimdIntel::m128_shuffle_epi8(const SmallVecType in,
                                                            const SmallVecType mask)
{
  return _mm_shuffle_epi8(in, mask);
}

inline SimdIntel::VecType SimdIntel::m256_shuffle_epi8(const VecType in, const VecType mask)
{
  return _mm256_shuffle_epi8(in, mask);
}

inline void SimdIntel::m256_hadamard16_epi16(const VecType x1, VecType &r1)
{
  // Apply a permutation 0123 -> 1032 to x1 (this operates on 64-bit words).
  auto a1 = m256_permute4x64_epi64_for_hadamard(x1);

  // From here we go back to treating x1 as a 16x16 vector.
  // Negate the first 8 of the elements in the vector
  auto t1 = m256_sign_epi16(x1, reinterpret_cast<VecType>(SimdIntel::sign_mask_8));

  // Add the permutation to the recently negated portion & apply the second sign mask.
  // (BTW the Wikipedia page for the Hadamard transform is really useful for understanding what's
  // going on here!)
  a1      = m256_add_epi16(a1, t1);
  auto b1 = m256_sign_epi16(a1, reinterpret_cast<VecType>(SimdIntel::sign_mask_2));
  a1      = m256_hadd_epi16(a1, b1);
  b1      = m256_sign_epi16(a1, reinterpret_cast<VecType>(SimdIntel::sign_mask_2));
  a1      = m256_hadd_epi16(a1, b1);
  b1      = m256_sign_epi16(a1, reinterpret_cast<VecType>(SimdIntel::sign_mask_2));
  r1      = m256_hadd_epi16(a1, b1);
}

inline void SimdIntel::m256_hadamard32_epi16(const VecType x1, const VecType x2, VecType &r1,
                                             VecType &r2)
{
  auto a1 = m256_permute4x64_epi64_for_hadamard(x1);
  auto a2 = m256_permute4x64_epi64_for_hadamard(x2);

  auto t1 = m256_sign_epi16(x1, reinterpret_cast<VecType>(SimdIntel::sign_mask_8));
  auto t2 = m256_sign_epi16(x2, reinterpret_cast<VecType>(SimdIntel::sign_mask_8));

  a1 = m256_add_epi16(a1, t1);
  a2 = m256_add_epi16(a2, t2);

  auto b1 = m256_sign_epi16(a1, reinterpret_cast<VecType>(SimdIntel::sign_mask_2));
  auto b2 = m256_sign_epi16(a2, reinterpret_cast<VecType>(SimdIntel::sign_mask_2));

  // Now apply the 16-bit Hadamard transforms and repeat the process
  a1 = m256_hadd_epi16(a1, b1);
  a2 = m256_hadd_epi16(a2, b2);
  b1 = m256_sign_epi16(a1, reinterpret_cast<VecType>(SimdIntel::sign_mask_2));
  b2 = m256_sign_epi16(a2, reinterpret_cast<VecType>(SimdIntel::sign_mask_2));
  a1 = m256_hadd_epi16(a1, b1);
  a2 = m256_hadd_epi16(a2, b2);
  b1 = m256_sign_epi16(a1, reinterpret_cast<VecType>(SimdIntel::sign_mask_2));
  b2 = m256_sign_epi16(a2, reinterpret_cast<VecType>(SimdIntel::sign_mask_2));
  a1 = m256_hadd_epi16(a1, b1);
  a2 = m256_hadd_epi16(a2, b2);

  r1 = m256_add_epi16(a1, a2);
  r2 = m256_sub_epi16(a1, a2);
}

inline void SimdIntel::m256_mix(VecType &v0, VecType &v1, const VecType &mask)
{
  VecType diff;
  diff = m256_xor_si256(v0, v1);
  diff = m256_and_si256(diff, mask);
  v0   = m256_xor_si256(v0, diff);
  v1   = m256_xor_si256(v1, diff);
}

inline SimdIntel::SmallVecType
SimdIntel::m128_random_state(SmallVecType prg_state, SmallVecType key, SmallVecType *extra_state)
{
  (void)extra_state;
  return _mm_aesenc_si128(prg_state, key);
}

template <>
inline void SimdIntel::m256_permute_epi16<2>(VecType *const v, SmallVecType &prg_state,
                                             const VecType tailmask, const SmallVecType &key,
                                             SmallVecType *extra_state)
{
  // double pack the prg state in rnd (has impact of doubly repeating the prg state in rnd)
  // Though we will use different threshold on each part decorrelating the permutation
  // on each halves

  auto rnd = m256_broadcastsi128_si256(prg_state);
  VecType mask;

  // With only 2 registers, we may not have enough room to randomize via m256_mix,
  // so we also choose at random among a few precomputed permutation to apply on
  // the first register

  uint32_t x  = m128_extract_epi64<0>(prg_state);
  uint32_t x1 = (x >> 16) & 0x03;
  uint32_t x2 = x & 0x03;

  // Apply the precomputed permutations to the input vector
  v[0] = m256_shuffle_epi8(v[0], reinterpret_cast<VecType>(SimdIntel::permutations_epi16[x1]));
  m256_mix(v[0], v[1], tailmask);
  v[0] = m256_permute4x64_epi64<0b10010011>(v[0]);
  v[0] = m256_shuffle_epi8(v[0], reinterpret_cast<VecType>(SimdIntel::permutations_epi16[x2]));

  mask = m256_cmpgt_epi16(rnd, reinterpret_cast<VecType>(SimdIntel::mixmask_threshold));
  mask = m256_and_si256(mask, tailmask);
  m256_mix(v[0], v[1], mask);

  // Update the randomness
  prg_state = m128_random_state(prg_state, key, extra_state);
}

template <int regs_>
inline void SimdIntel::m256_permute_epi16(VecType *const v, SmallVecType &prg_state,
                                          const VecType tailmask, const SmallVecType &key,
                                          SmallVecType *extra_state)
{

  // double pack the prg state in rnd (has impact of doubly repeating the prg state in rnd)
  // Though we will use different threshold on each part decorrelating the permutation
  // on each halves

  auto rnd = m256_broadcastsi128_si256(prg_state);
  VecType tmp;

  // We treat the even and the odd positions differently
  // This is for the goal of decorrelating the permutation on the
  // double packed prng state.
  for (int i = 0; i < (regs_ - 1) / 2; ++i)
  {
    // shuffle 8 bit parts in each 128 bit lane
    // Note - the exact semantics of what this function does are a bit confusing.
    // See the Intel intrinsics guide if you're curious
    v[2 * i] = m256_shuffle_epi8(v[2 * i],
                                 reinterpret_cast<VecType>(SimdIntel::permutations_epi16[i % 3]));
    // For the odd positions we permute each 64-bit chunk according to the mask.
    v[2 * i + 1] = m256_permute4x64_epi64<0b10010011>(v[2 * i + 1]);
  }

  // Now we negate the first two vectors according to the negation masks
  v[0] = m256_sign_epi16(v[0], reinterpret_cast<VecType>(SimdIntel::negation_masks_epi16[0]));
  v[1] = m256_sign_epi16(v[1], reinterpret_cast<VecType>(SimdIntel::negation_masks_epi16[1]));

  // swap int16 entries of v[0] and v[1] where rnd > threshold
  tmp = m256_cmpgt_epi16(rnd, reinterpret_cast<VecType>(SimdIntel::mixmask_threshold));
  m256_mix(v[0], v[1], tmp);
  // Shift the randomness around before extracting more (somewhat independent) mixing bits
  rnd = m256_slli_epi16(rnd, 1);

  // Now do random swaps between v[0] and v[last-1]
  m256_mix(v[0], v[regs_ - 2], tmp);
  rnd = m256_slli_epi16(rnd, 1);

  // Now do swaps between v[1] and v[last], avoiding padding data
  m256_mix(v[1], v[regs_ - 1], tailmask);

  // More permuting
  for (int i = 2; i + 2 < regs_; i += 2)
  {
    rnd = m256_slli_epi16(rnd, 1);
    tmp = m256_cmpgt_epi16(rnd, reinterpret_cast<VecType>(SimdIntel::mixmask_threshold));
    m256_mix(v[0], v[i], tmp);
    rnd = m256_slli_epi16(rnd, 1);
    tmp = m256_cmpgt_epi16(rnd, reinterpret_cast<VecType>(SimdIntel::mixmask_threshold));
    m256_mix(v[1], v[i + 1], tmp);
  }

  // Update the randomness.
  prg_state = m128_random_state(prg_state, key, extra_state);
}
