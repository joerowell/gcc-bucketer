#ifndef INCLUDED_SIMD_GENERIC
#error Do not include simd_generic.inl without simd_generic.hpp
#endif

#include <array>  // Needed for array
#include <cassert>
#include <cstring>  // Needed for memcpy.

inline void SimdGeneric::m256_storeu_si256(int16_t *a, const VecType b)
{
  memcpy(a, &b, sizeof(VecType));
}

inline SimdGeneric::VecType SimdGeneric::m256_loadu_si256(const int16_t *const a)
{
  SimdGeneric::VecType b;
  memcpy(&b, a, sizeof(b));
  return b;
}

template <int pos> inline int64_t SimdGeneric::m256_extract_epi64(const VecType a)
{
  static_assert(pos == 0 || pos == 1 || pos == 2 || pos == 3, "Error: pos can only be in {0,3}");
  std::array<int64_t, 4> b;
  memcpy(&b, &a, sizeof(a));
  return b[pos];
}

template <int pos> inline int64_t SimdGeneric::m128_extract_epi64(const SmallVecType a)
{
  static_assert(pos == 0 || pos == 1, "Error: pos can only be 0 or 1");
  std::array<int64_t, 2> b;
  memcpy(&b, &a, sizeof(a));
  return b[pos];
}

inline SimdGeneric::SmallVecType SimdGeneric::m128_set_epi64x(const int64_t e1, const int64_t e0)
{
  SimdGeneric::SmallVecType a;
  memcpy(&a, &e0, sizeof(e0));
  memcpy(&a[4], &e1, sizeof(e1));
  return a;
}

inline SimdGeneric::SmallVecType SimdGeneric::m128_set_epi64x(const uint64_t e1, const uint64_t e0)
{
  SimdGeneric::SmallVecType a;
  memcpy(&a, &e0, sizeof(e0));
  memcpy(&a[4], &e1, sizeof(e1));
  return a;
}

inline int SimdGeneric::m256_testz_si256(const SimdGeneric::VecType a, const SimdGeneric::VecType b)
{
  bool is_zero = false;
  for (unsigned i = 0; i < a.size(); i++)
  {
    is_zero |= a[i] & b[i];
  }

  return !is_zero;
}

inline SimdGeneric::VecType SimdGeneric::m256_abs_epi16(const SimdGeneric::VecType a)
{
  VecType out;
  for (unsigned i = 0; i < a.size(); i++)
  {
    out[i] = std::abs(a[i]);
  }
  return out;
}

inline SimdGeneric::VecType SimdGeneric::m256_cmpgt_epi16(const SimdGeneric::VecType a,
                                                          const SimdGeneric::VecType b)
{
  VecType c;
  for (unsigned i = 0; i < a.size(); i++)
  {
    c[i] = static_cast<int16_t>((a[i] > b[i]) * 0xFFFF);
  }
  return c;
}

inline SimdGeneric::VecType SimdGeneric::m256_slli_epi16(const SimdGeneric::VecType a,
                                                         const int count)
{
  VecType c;
  for (unsigned i = 0; i < a.size(); i++)
  {
    c[i] = a[i] << count;
  }
  return c;
}

inline SimdGeneric::VecType SimdGeneric::m256_hadd_epi16(const SimdGeneric::VecType a,
                                                         const SimdGeneric::VecType b)
{

  return VecType{static_cast<int16_t>(a[0] + a[1]),   static_cast<int16_t>(a[2] + a[3]),
                 static_cast<int16_t>(a[4] + a[5]),   static_cast<int16_t>(a[6] + a[7]),
                 static_cast<int16_t>(b[0] + b[1]),   static_cast<int16_t>(b[2] + b[3]),
                 static_cast<int16_t>(b[4] + b[5]),   static_cast<int16_t>(b[6] + b[7]),
                 static_cast<int16_t>(a[8] + a[9]),   static_cast<int16_t>(a[10] + a[11]),
                 static_cast<int16_t>(a[12] + a[13]), static_cast<int16_t>(a[14] + a[15]),
                 static_cast<int16_t>(b[8] + b[9]),   static_cast<int16_t>(b[10] + b[11]),
                 static_cast<int16_t>(b[12] + b[13]), static_cast<int16_t>(b[14] + b[15])};
}

inline SimdGeneric::VecType SimdGeneric::m256_add_epi16(const SimdGeneric::VecType a,
                                                        const SimdGeneric::VecType b)
{
  VecType c;
  for (unsigned i = 0; i < a.size(); i++)
  {
    c[i] = a[i] + b[i];
  }
  return c;
}

inline SimdGeneric::SmallVecType SimdGeneric::m128_add_epi64(const SimdGeneric::SmallVecType a,
                                                             const SimdGeneric::SmallVecType b)
{
  std::array<int64_t, 2> a_1, b_1;
  memcpy(&a_1, &a, sizeof(a));
  memcpy(&b_1, &b, sizeof(b));
  a_1[0] += b_1[0];
  a_1[1] += b_1[1];
  SmallVecType out;
  memcpy(&out, &a_1, sizeof(out));
  return out;
}

inline SimdGeneric::VecType SimdGeneric::m256_sub_epi16(const SimdGeneric::VecType a,
                                                        const SimdGeneric::VecType b)
{
  VecType c;
  for (unsigned i = 0; i < a.size(); i++)
  {
    c[i] = a[i] - b[i];
  }
  return c;
}

inline SimdGeneric::VecType SimdGeneric::m256_and_si256(const SimdGeneric::VecType a,
                                                        const SimdGeneric::VecType b)
{
  VecType c;
  for (unsigned i = 0; i < a.size(); i++)
  {
    c[i] = a[i] & b[i];
  }
  return c;
}

inline SimdGeneric::VecType SimdGeneric::m256_xor_si256(const SimdGeneric::VecType a,
                                                        const SimdGeneric::VecType b)
{
  VecType c;
  for (unsigned i = 0; i < a.size(); i++)
  {
    c[i] = a[i] ^ b[i];
  }
  return c;
}

inline SimdGeneric::SmallVecType SimdGeneric::m128_xor_si128(const SimdGeneric::SmallVecType a,
                                                             const SimdGeneric::SmallVecType b)
{

  SmallVecType c;
  for (unsigned i = 0; i < a.size(); i++)
  {
    c[i] = a[i] ^ b[i];
  }
  return c;
}

inline SimdGeneric::VecType
SimdGeneric::m256_permute4x64_epi64_for_hadamard(const SimdGeneric::VecType a)
{
  return SimdGeneric::m256_permute4x64_epi64<0b01001110>(a);
}

template <uint8_t mask>
inline SimdGeneric::VecType SimdGeneric::m256_permute4x64_epi64(const SimdGeneric::VecType a)
{
  // This intrinsic works by using each 2-bit pair in the `mask` variable as a selector in `a`
  constexpr auto zeroth = mask & 3;
  constexpr auto first  = (mask & 12) >> 2;
  constexpr auto second = (mask & 48) >> 4;
  constexpr auto third  = (mask & 192) >> 6;

  // NOTE: to prevent UB here we need to copy `a` into an array of 64-bit ints.
  // This is essentially to stop type punning related bugs. This should actually
  // be zero overhead, as the optimiser should optimise this away.
  const SimdGeneric::Vec4q a_as_4q{*reinterpret_cast<const int64_t *const>(&a[0]),
                                   *reinterpret_cast<const int64_t *const>(&a[4]),
                                   *reinterpret_cast<const int64_t *const>(&a[8]),
                                   *reinterpret_cast<const int64_t *const>(&a[12])};

  // And now we'll do the actual shuffling.
  const SimdGeneric::Vec4q temp{a_as_4q[zeroth], a_as_4q[first], a_as_4q[second], a_as_4q[third]};

  // And now convert back to a VecType
  SimdGeneric::VecType c;
  memcpy(&c, &temp, sizeof(c));
  return c;
}

inline SimdGeneric::VecType SimdGeneric::m256_sign_epi16(const SimdGeneric::VecType a,
                                                         const SimdGeneric::VecType mask)
{

  // The best thing to do here is to extract the sign of each element in `mask`
  // and then multiply through.

  SimdGeneric::VecType signs;
  for (unsigned i = 0; i < mask.size(); i++)
  {
    signs[i] = (mask[i] > 0) - (mask[i] < 0);
  }

  return m256_sign_epi16_ternary(a, signs);
}

inline SimdGeneric::VecType SimdGeneric::m256_sign_epi16_ternary(const SimdGeneric::VecType a,
                                                                 const SimdGeneric::VecType mask)
{
  SimdGeneric::VecType c;
  for (unsigned i = 0; i < a.size(); i++)
  {
    c[i] = a[i] * mask[i];
  }

  return c;
}

inline SimdGeneric::VecType
SimdGeneric::m256_broadcastsi128_si256(const SimdGeneric::SmallVecType in)
{
  SimdGeneric::VecType a;
  memcpy(&a[0], &in[0], sizeof(in));
  memcpy(&a[8], &in[0], sizeof(in));
  return a;
}

inline SimdGeneric::SmallVecType SimdGeneric::m128_shuffle_epi8(const SmallVecType in,
                                                                const SmallVecType mask)
{

  // We need to convert the input vectors into arrays of 8-bit ints.
  std::array<uint8_t, 16> in_arr, mask_arr;

  static_assert(sizeof(in_arr) == sizeof(in),
                "Error: mismatch between sizeof(in_arr) and sizeof(in)");
  static_assert(sizeof(mask_arr) == sizeof(mask),
                "Error: mismatch between sizeof(mask_arr) and sizeof(mask)");

  SimdGeneric::Vec16c c;

  memcpy(&in_arr, &in, sizeof(in_arr));
  memcpy(&mask_arr, &mask, sizeof(mask_arr));

  for (unsigned i = 0; i < in_arr.size(); i++)
  {
    c[i] = in_arr[mask_arr[i] & 15];
    c[i] = (mask_arr[i] & 0x80) ? 0 : c[i];
  }

  SimdGeneric::SmallVecType out;
  memcpy(&out, &c, sizeof(out));
  return out;
}

inline void SimdGeneric::m256_mix(VecType &v0, VecType &v1, const VecType &mask)
{
  VecType diff;
  diff = m256_xor_si256(v0, v1);
  diff = m256_and_si256(diff, mask);
  v0   = m256_xor_si256(v0, diff);
  v1   = m256_xor_si256(v1, diff);
}

inline SimdGeneric::VecType SimdGeneric::m256_shuffle_epi8(const VecType in, const VecType mask)
{
  Vec8s first_mask, last_mask, first, last;
  memcpy(&first, &in, sizeof(first));
  memcpy(&last, &in[8], sizeof(last));
  memcpy(&first_mask, &mask, sizeof(first_mask));
  memcpy(&last_mask, &mask[8], sizeof(last_mask));

  // Delegate to the 128-bit version.
  auto res_1 = SimdGeneric::m128_shuffle_epi8(first, first_mask);
  auto res_2 = SimdGeneric::m128_shuffle_epi8(last, last_mask);

  std::array<int16_t, 16> out_as_arr;
  memcpy(&out_as_arr, &res_1, sizeof(res_1));
  memcpy(&out_as_arr[8], &res_2, sizeof(res_2));

  Vec16s result;
  memcpy(&result, &out_as_arr, sizeof(result));
  return result;
}

inline void SimdGeneric::m256_hadamard16_epi16(const VecType x1, VecType &r1)
{
  // Apply a permutation 0123 -> 1032 to x1 (this operates on 64-bit words).
  auto a1 = m256_permute4x64_epi64_for_hadamard(x1);

  // From here we go back to treating x1 as a 16x16 vector.
  // Negate the first 8 of the elements in the vector
  auto t1 = m256_sign_epi16(x1, sign_mask_8);

  // Add the permutation to the recently negated portion & apply the second sign mask.
  // (BTW the Wikipedia page for the Hadamard transform is really useful for understanding what's
  // going on here!)
  a1      = m256_add_epi16(a1, t1);
  auto b1 = m256_sign_epi16(a1, sign_mask_2);
  a1      = m256_hadd_epi16(a1, b1);
  b1      = m256_sign_epi16(a1, sign_mask_2);
  a1      = m256_hadd_epi16(a1, b1);
  b1      = m256_sign_epi16(a1, sign_mask_2);
  r1      = m256_hadd_epi16(a1, b1);
}

template <>
inline void SimdGeneric::m256_permute_epi16<2>(VecType *const v, SmallVecType &prg_state,
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
  v[0] = m256_shuffle_epi8(v[0], permutations_epi16[x1]);
  m256_mix(v[0], v[1], tailmask);
  v[0] = m256_permute4x64_epi64<0b10010011>(v[0]);
  v[0] = m256_shuffle_epi8(v[0], permutations_epi16[x2]);

  mask = m256_cmpgt_epi16(rnd, mixmask_threshold);
  mask = m256_and_si256(mask, tailmask);
  m256_mix(v[0], v[1], mask);

  // Update the randomness
  prg_state = m128_random_state(prg_state, key, extra_state);
}

template <int regs_>
inline void SimdGeneric::m256_permute_epi16(VecType *const v, SmallVecType &prg_state,
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
    v[2 * i] = m256_shuffle_epi8(v[2 * i], permutations_epi16[i % 3]);
    // For the odd positions we permute each 64-bit chunk according to the mask.
    v[2 * i + 1] = m256_permute4x64_epi64<0b10010011>(v[2 * i + 1]);
  }

  // Now we negate the first two vectors according to the negation masks
  v[0] = m256_sign_epi16(v[0], negation_masks_epi16[0]);
  v[1] = m256_sign_epi16(v[1], negation_masks_epi16[1]);

  // swap int16 entries of v[0] and v[1] where rnd > threshold
  tmp = m256_cmpgt_epi16(rnd, mixmask_threshold);
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
    tmp = m256_cmpgt_epi16(rnd, mixmask_threshold);
    m256_mix(v[0], v[i], tmp);
    rnd = m256_slli_epi16(rnd, 1);
    tmp = m256_cmpgt_epi16(rnd, mixmask_threshold);
    m256_mix(v[1], v[i + 1], tmp);
  }

  // Update the randomness.
  prg_state = m128_random_state(prg_state, key, extra_state);
}

inline SimdGeneric::SmallVecType
SimdGeneric::m128_random_state(SmallVecType prg_state, SmallVecType key, SmallVecType *extra_state)
{
  // Silence the fact it isn't used.
  (void)key;

  SmallVecType s1       = prg_state;
  const SmallVecType s0 = *extra_state;

  s1           = m128_xor_si128(s1, m128_slli_epi64<23>(s1));
  *extra_state = m128_xor_si128(m128_xor_si128(m128_xor_si128(s1, s0), m128_srli_epi64<5>(s1)),
                                m128_srli_epi64<5>(s0));
  return m128_add_epi64(*extra_state, s0);
}

template <int pos>
inline SimdGeneric::SmallVecType SimdGeneric::m128_slli_epi64(const SmallVecType a)
{
  std::array<int64_t, 2> a_as_arr;
  memcpy(&a_as_arr, &a, sizeof(a));
  a_as_arr[0] = a_as_arr[0] << pos;
  a_as_arr[1] = a_as_arr[1] << pos;
  SmallVecType out;
  memcpy(&out, &a_as_arr, sizeof(a_as_arr));
  return out;
}

template <int pos>
inline SimdGeneric::SmallVecType SimdGeneric::m128_srli_epi64(const SmallVecType a)
{
  std::array<int64_t, 2> a_as_arr;
  memcpy(&a_as_arr, &a, sizeof(a));
  a_as_arr[0] = a_as_arr[0] >> pos;
  a_as_arr[1] = a_as_arr[1] >> pos;
  SmallVecType out;
  memcpy(&out, &a_as_arr, sizeof(a_as_arr));
  return out;
}

inline void SimdGeneric::m256_hadamard32_epi16(const VecType x1, const VecType x2, VecType &r1,
                                               VecType &r2)
{
  auto a1 = m256_permute4x64_epi64_for_hadamard(x1);
  auto a2 = m256_permute4x64_epi64_for_hadamard(x2);

  auto t1 = m256_sign_epi16(x1, sign_mask_8);
  auto t2 = m256_sign_epi16(x2, sign_mask_8);

  a1 = m256_add_epi16(a1, t1);
  a2 = m256_add_epi16(a2, t2);

  auto b1 = m256_sign_epi16(a1, sign_mask_2);
  auto b2 = m256_sign_epi16(a2, sign_mask_2);

  // Now apply the 16-bit Hadamard transforms and repeat the process
  a1 = m256_hadd_epi16(a1, b1);
  a2 = m256_hadd_epi16(a2, b2);
  b1 = m256_sign_epi16(a1, sign_mask_2);
  b2 = m256_sign_epi16(a2, sign_mask_2);
  a1 = m256_hadd_epi16(a1, b1);
  a2 = m256_hadd_epi16(a2, b2);
  b1 = m256_sign_epi16(a1, sign_mask_2);
  b2 = m256_sign_epi16(a2, sign_mask_2);
  a1 = m256_hadd_epi16(a1, b1);
  a2 = m256_hadd_epi16(a2, b2);

  r1 = m256_add_epi16(a1, a2);
  r2 = m256_sub_epi16(a1, a2);
}
