#ifndef INCLUDED_SIMD_GENERIC
#define INCLUDED_SIMD_GENERIC
#include <array>
#include <cstdint>
#include <iostream>

/**
 * SimdGeneric. This namespace provides generic implementations of the
 * simd functions provided in simd.hpp. This namespace should primarily be used
 * for comparing the simd functions against their intended behaviour on different platforms,
 * as the behaviour may vary.
 *
 * Please note that testing this namespace must be done on x86-64 machines, as we compare directly
 * against the Intel intrinsics.
 **/

namespace SimdGeneric
{
// Naming convention for vectors here is: Vec_(number_of_elements)_(first letter of type).
// So, for example, a vector with 8 shorts in it is written as Vec8s.
// Second note: technically all of the vector types we define here (with a handful of exceptions)
// are the same. The reason why we differentiate between different types is simply to make later
// debugging easier.
/**
   Vec16s. This vector contains 16 shorts (i.e 16 * 16 bits). It is 256-bits in size.
   Note that this type is 16-byte aligned.
 **/
using Vec16s = std::array<int16_t, 16>;
/**
   Vec4q. This vector contains 4 quadwords (i.e 4 * 64 bits). It is 256-bits in size and
   16-byte aligned.
**/
using Vec4q = std::array<int64_t, 4>;

/**
   Vec4uq. This vector contains 4 unsigned quadwords (i.e 4 * 64 bits). It is 256-bits in size and
   16-byte aligned.
**/
using Vec4uq = std::array<uint64_t, 16>;

/**
   Vec2q. This vector contains 2 quadwords (i.e 2 * 64 bits). It is 128-bits in size
   and 16-byte aligned.
**/
using Vec2q = std::array<int64_t, 2>;

/**
   Vec2uq. This vector contains 2 unsigned quadwords (i.e 2 * 64 bits). It is 128-bits in size
   and 16-byte aligned.
**/
using Vec2uq = std::array<int64_t, 2>;

/**
   Vec8s. This vector contains 8 shorts (i.e 8 * 16 bits). It is 128-bits in size and 16-byte
aligned.
**/
using Vec8s = std::array<int16_t, 8>;

/**
   Vec8d. This vector contains 8 doublewords (i.e 8 * 32bits). It is 256-bits in size and 16-byte
   aligned.
**/
using Vec8d = std::array<int32_t, 8>;

/**
   Vec32c. This vector contains 32 chars (i.e 32 * 8 bits). It is 256-bits in size and 16-byte
   aligned.
**/
using Vec32c = std::array<int8_t, 32>;

/**
   Vec16c. This vector contains 16 chars (i.e 16 * 8 bits). It is 128-bits in size and 16-byte
   aligned.
**/
using Vec16c = std::array<int8_t, 16>;

/**
   Vec16uc. This vector contains 16 unsigned chars (i.e 16 * 8 bits). It is 128-bits in size and
16-byte aligned.
**/
using Vec16uc = std::array<uint8_t, 16>;

using VecType      = Vec16s;
using SmallVecType = Vec8s;

// Generic constructors for the Vector Types.
// NOTE: Vec16s's constructor takes the input array and reverses it.
// This is solely to make the functions compatible between both Intel's format and
// the GCC format. That's the reason why the function is constexpr: ideally you shouldn't
// call it at runtime as it generates truly awful object code.
// Note: to allow these to be used elsewhere in this namespace we'll define these inline
constexpr Vec16s build_vec16s(const int16_t *const in)
{
  return Vec16s{in[0], in[1], in[2],  in[3],  in[4],  in[5],  in[6],  in[7],
                in[8], in[9], in[10], in[11], in[12], in[13], in[14], in[15]};
}

// And hey, why not
constexpr Vec16s build_vec16s(int16_t e15, int16_t e14, int16_t e13, int16_t e12, int16_t e11,
                              int16_t e10, int16_t e9, int16_t e8, int16_t e7, int16_t e6,
                              int16_t e5, int16_t e4, int16_t e3, int16_t e2, int16_t e1,
                              int16_t e0)
{
  return Vec16s{e0, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12, e13, e14, e15};
}

// Apparently this is useful too
constexpr Vec32c build_vec32c(int8_t e31, int8_t e30, int8_t e29, int8_t e28, int8_t e27,
                              int8_t e26, int8_t e25, int8_t e24, int8_t e23, int8_t e22,
                              int8_t e21, int8_t e20, int8_t e19, int8_t e18, int8_t e17,
                              int8_t e16, int8_t e15, int8_t e14, int8_t e13, int8_t e12,
                              int8_t e11, int8_t e10, int8_t e9, int8_t e8, int8_t e7, int8_t e6,
                              int8_t e5, int8_t e4, int8_t e3, int8_t e2, int8_t e1, int8_t e0)
{

  return Vec32c{e0,  e1,  e2,  e3,  e4,  e5,  e6,  e7,  e8,  e9,  e10, e11, e12, e13, e14, e15,
                e16, e17, e18, e19, e20, e21, e22, e23, e24, e25, e26, e27, e28, e29, e30, e31};
}

// And this
constexpr Vec8d build_vec8d(int32_t e7, int32_t e6, int32_t e5, int32_t e4, int32_t e3, int32_t e2,
                            int32_t e1, int32_t e0)
{

  return Vec8d{e0, e1, e2, e3, e4, e5, e6, e7};
}

inline VecType build_vec_type(const int16_t *const in)
{
  // Delegate to the dedicated routine.
  return build_vec16s(in);
}

inline VecType build_vec_type(const int16_t in)
{
#ifdef HAVE_AVX2
  return _mm256_set_epi16(in, in, in, in, in, in, in, in, in, in, in, in, in, in, in, in);
#else
  return Vec16s{in, in, in, in, in, in, in, in, in, in, in, in, in, in, in, in};
#endif
}

constexpr auto mixmask_threshold =
    build_vec16s(0x5555, 0x5555, 0x5555, 0x5555, 0x5555, 0x5555, 0x5555, 0x5555, 0xAAAA, 0xAAAA,
                 0xAAAA, 0xAAAA, 0xAAAA, 0xAAAA, 0xAAAA, 0xAAAA);

constexpr auto _7FFF_epi16 =
    build_vec16s(0x7FFF, 0x7FFF, 0x7FFF, 0x7FFF, 0x7FFF, 0x7FFF, 0x7FFF, 0x7FFF, 0x7FFF, 0x7FFF,
                 0x7FFF, 0x7FFF, 0x7FFF, 0x7FFF, 0x7FFF, 0x7FFF);

constexpr auto sign_mask_2 =
    build_vec16s(0xFFFF, 0x0001, 0xFFFF, 0x0001, 0xFFFF, 0x0001, 0xFFFF, 0x0001, 0xFFFF, 0x0001,
                 0xFFFF, 0x0001, 0xFFFF, 0x0001, 0xFFFF, 0x0001);

constexpr auto mask_even_epi16 =
    build_vec16s(0xFFFF, 0x0000, 0xFFFF, 0x0000, 0xFFFF, 0x0000, 0xFFFF, 0x0000, 0xFFFF, 0x0000,
                 0xFFFF, 0x0000, 0xFFFF, 0x0000, 0xFFFF, 0x0000);

constexpr auto mask_odd_epi16 =
    build_vec16s(0x0000, 0xFFFF, 0x0000, 0xFFFF, 0x0000, 0xFFFF, 0x0000, 0xFFFF, 0x0000, 0xFFFF,
                 0x0000, 0xFFFF, 0x0000, 0xFFFF, 0x0000, 0xFFFF);

constexpr auto regroup_for_max = build_vec32c(
    0x0F, 0x0E, 0x07, 0x06, 0x0D, 0x0C, 0x05, 0x04, 0x0B, 0x0A, 0x03, 0x02, 0x09, 0x08, 0x01, 0x00,
    0x1F, 0x1E, 0x17, 0x16, 0x1D, 0x1C, 0x15, 0x14, 0x1B, 0x1A, 0x13, 0x12, 0x19, 0x18, 0x11, 0x10);

constexpr auto sign_mask_8 =
    build_vec16s(0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0x0001, 0x0001,
                 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001);

constexpr auto sign_shuffle =
    build_vec16s(0xFFFF, 0xFFFF, 0xFFFF, 0x0001, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0x0001, 0x0001,
                 0x0001, 0x0001, 0x0001, 0x0001, 0xFFFF, 0xFFFF);

constexpr auto indices_epi8 = build_vec32c(
    0x0F, 0x0E, 0x0D, 0x0C, 0x0B, 0x0A, 0x09, 0x08, 0x07, 0x06, 0x05, 0x04, 0x03, 0x02, 0x01, 0x00,
    0x1F, 0x1E, 0x1D, 0x1C, 0x1B, 0x1A, 0x19, 0x18, 0x17, 0x16, 0x15, 0x14, 0x13, 0x12, 0x11, 0x10);

constexpr auto indices_epi16 =
    build_vec16s(0x000F, 0x000E, 0x000D, 0x000C, 0x000B, 0x000A, 0x0009, 0x0008, 0x0007, 0x0006,
                 0x0005, 0x0004, 0x0003, 0x0002, 0x0001, 0x0000);

constexpr auto indices_sa1_epi16 =
    build_vec16s(0x0010, 0x000F, 0x000E, 0x000D, 0x000C, 0x000B, 0x000A, 0x0009, 0x0008, 0x0007,
                 0x0006, 0x0005, 0x0004, 0x0003, 0x0002, 0x0001);

constexpr auto _0010_epi16 =
    build_vec16s(0x0010, 0x0010, 0x0010, 0x0010, 0x0010, 0x0010, 0x0010, 0x0010, 0x0010, 0x0010,
                 0x0010, 0x0010, 0x0010, 0x0010, 0x0010, 0x0010);

constexpr auto rnd_mult_epi32 = build_vec8d(0xF010A011, 0x70160011, 0x70162011, 0x00160411,
                                            0x0410F011, 0x02100011, 0xF0160011, 0x00107010);

// 0xFFFF = -1, 0x0001 = 1
constexpr Vec16s negation_masks_epi16[2] = {
    build_vec16s(0xFFFF, 0x0001, 0xFFFF, 0xFFFF, 0xFFFF, 0x0001, 0x0001, 0xFFFF, 0xFFFF, 0x0001,
                 0xFFFF, 0x0001, 0xFFFF, 0xFFFF, 0x0001, 0xFFFF),
    build_vec16s(0xFFFF, 0x0001, 0x0001, 0xFFFF, 0xFFFF, 0x0001, 0x0001, 0xFFFF, 0xFFFF, 0x0001,
                 0xFFFF, 0x0001, 0xFFFF, 0xFFFF, 0x0001, 0xFFFF)};

constexpr Vec16s permutations_epi16[4] = {
    build_vec16s(0x0F0E, 0x0706, 0x0100, 0x0908, 0x0B0A, 0x0D0C, 0x0504, 0x0302, 0x0706, 0x0F0E,
                 0x0504, 0x0302, 0x0B0A, 0x0908, 0x0D0C, 0x0100),
    build_vec16s(0x0D0C, 0x0504, 0x0302, 0x0B0A, 0x0F0E, 0x0908, 0x0706, 0x0100, 0x0B0A, 0x0908,
                 0x0706, 0x0F0E, 0x0302, 0x0100, 0x0504, 0x0D0C),
    build_vec16s(0x0D0C, 0x0B0A, 0x0706, 0x0100, 0x0F0E, 0x0908, 0x0504, 0x0302, 0x0B0A, 0x0908,
                 0x0302, 0x0100, 0x0504, 0x0D0C, 0x0706, 0x0F0E),
    build_vec16s(0x0D0C, 0x0F0E, 0x0908, 0x0706, 0x0100, 0x0504, 0x0302, 0x0B0A, 0x0302, 0x0100,
                 0x0504, 0x0B0A, 0x0908, 0x0706, 0x0F0E, 0x0D0C)};

constexpr Vec16s tailmasks[16] = {
    build_vec16s(0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF,
                 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF),
    build_vec16s(0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
                 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0xFFFF),
    build_vec16s(0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
                 0x0000, 0x0000, 0x0000, 0x0000, 0xFFFF, 0xFFFF),
    build_vec16s(0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
                 0x0000, 0x0000, 0x0000, 0xFFFF, 0xFFFF, 0xFFFF),
    build_vec16s(0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
                 0x0000, 0x0000, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF),
    build_vec16s(0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
                 0x0000, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF),
    build_vec16s(0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
                 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF),
    build_vec16s(0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0xFFFF,
                 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF),
    build_vec16s(0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0xFFFF, 0xFFFF,
                 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF),
    build_vec16s(0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0xFFFF, 0xFFFF, 0xFFFF,
                 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF),
    build_vec16s(0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF,
                 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF),
    build_vec16s(0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF,
                 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF),
    build_vec16s(0x0000, 0x0000, 0x0000, 0x0000, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF,
                 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF),
    build_vec16s(0x0000, 0x0000, 0x0000, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF,
                 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF),
    build_vec16s(0x0000, 0x0000, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF,
                 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF),
    build_vec16s(0x0000, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF,
                 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF)};

/**
   m256_storeu_si256. This function accepts a vector `b` and stores its entries in the passed array
   `a`. This function exaclty mimics the m256_storeu_si256 function. \param[in] a: the array to
store in. \param[in] b: the vector that is being stored.
**/
inline void m256_storeu_si256(int16_t *a, const VecType b);

/**
   m256_loadu_si256. This function accepts an array of 16-bit ints (`a`) and
   returns a vector containing the elements on `a`. This function exactly mimics
   the _mm256_loadu_si256 intrinsic.
   \param[in] a: the array to store.
   \return a vector containing the elements of `a`.
**/
inline VecType m256_loadu_si256(const int16_t *const a);

/**
   m256_extract_epi64. This function accepts a vector `a` and extracts the 64-bit entry
   held at position `pos`. This function exactly mimics the behaviour of _mm256_extract_epi64.
   \param[in] a: the vector to extract from.
   \tparam[in] pos: the position to extract from.
   \return a[pos];
 **/
template <int pos> inline int64_t m256_extract_epi64(const VecType a);

/**
   m128_extract_epi64. This function accepts a vector `a` and extracts the 64-bit entry
   held at position `pos`. This function exactly mimics the behaviour of _mm_extract_epi64.
   \param[in] a: the vector to extract from.
   \tparam[in] pos: the position to extract from.
   \return a[pos];
 **/
template <int pos> inline int64_t m128_extract_epi64(const SmallVecType a);

/**
 m128_set_epi64. This function creates a 128-bit vector with `e0` at position 0
 and `e1` at position 1. This function exactly mimics the behaviour of _mm_set_epi64x.
 \param[in] e1: the element to be placed at position 1.
 \param[in] e0: the element to be placed at position 0.
 \return a vector {e0, e1}.
**/
inline SmallVecType m128_set_epi64x(const int64_t e1, const int64_t e0);
/**
   m128_set_epi64. This function creates a 128-bit vector with `e0` at position 0
   and `e1` at position 1. This function exactly mimics the behaviour of _mm_set_epi64x: the main
difference is that it works on unsigned ints. \param[in] e1: the element to be placed at position 1.
   \param[in] e0: the element to be placed at position 0.
   \return a vector {e0, e1}.
**/
inline SmallVecType m128_set_epi64x(const uint64_t e1, const uint64_t e0);

/**
   m256_testz_si256. This function accepts two vectors `a` and `b` and computes their
   bitwise AND. If the bitwise AND is 0 then return 1. Otherwise, return 0.
   This function mimics the behaviour of _mm256_testz_si256.
   \param[in] a: the left operand.
   \param[in] b: the right operand.
   \return (a & b) == 0.
**/
inline int m256_testz_si256(const VecType a, const VecType b);

/**
   m256_abs_epi16. This function accepts a vector `a` and returns a vector
   containing the absolute value of all entries in `a`. In other words,
   this function returns {ABS(a[0]), ABS(a[1]),...,ABS(a[15]).
   This function mimics the behaviour provided by _mm256_abs_epi16.
   \param[in] a: the vector to be operated on.
   \return ABS(a).
**/
inline VecType m256_abs_epi16(const VecType a);

/**
   m256_cmpgt_epi16. This function accepts two vectors `a` and `b`, and returns
 * the comparison mask between `a` and `b`, denoted as `c`.
 *
 * This exactly mimics the behaviour of the _mm256_cmpgt_epi16
 * function. In particular, after this function is called, `c` has the following
 * layout: For all i = 0, ..., 15:
 *
 * c[i] = 0xFFFF if a[i] > b[i]
 * c[i] = 0 otherwise
 * \param[in] a: the left operand.
 * \param[in] b: the right operand.
 * \return a > b.
 */
inline VecType m256_cmpgt_epi16(const VecType a, const VecType b);

/**
 * m256_slli_epi16. This function accepts a vector `a` and shifts each word in `a` left by
 * `count` many bits. This function mimics exactly the behaviour of _mm256_slli_epi16.
 * \param[in] a: the vector to shift.
 * \param[in] count: the amount to shift by.
 * \return a << count.
 */
inline VecType m256_slli_epi16(const VecType a, const int count);

/**
 * m128_slli_epi64. This function accepts a vector `a` and shifts each quadword in `a` left by
 * `count` many bits. This function mimics exactly the behaviour of _mm_slli_epi64.
 * \param[in] a: the vector to shift.
 * \tparam[in] count: the amount to shift by.
 * \return a << count.
 */
template <int pos> inline SmallVecType m128_slli_epi64(const SmallVecType a);

/**
 * m128_srli_epi64. This function accepts a vector `a` and shifts each quadword in `a` right by
 * `count` many bits. This function mimics exactly the behaviour of _mm_srli_epi64.
 * \param[in] a: the vector to shift.
 * t\param[in] count: the amount to shift by.
 * \return a >> count.
 */
template <int pos> inline SmallVecType m128_srli_epi64(const SmallVecType a);

/**
 m256_hadd_epi16. Accepts two vectors `a` and `b` and emulates the _mm256i_hadd_epi16 instruction.
 This instruction pairwises adds the first 8 elements in `a` (i.e a[0] + a[1], a[2] + a[3], a[4] +
 a[5], a[6] + a[7]) and stores them in the first 64 bits of the output vector: the second 64 bits
are the pairwise sum of the first 8 elements of `b`, and the remaining 128 bits are the second 8
 elements of `a` and `b` respectively. This method works by applying a series of shuffles to `a` and
 `b` to produce the correct output. \param[in] a: the vector whose sum makes up the first and third
 sets of 4 in the output vector. \param[in] b: the vector whose sum makes up the second and fourth
 sets of 4 in the output vector. \return the hadd_epi16 of `a` and `b`
**/
inline VecType m256_hadd_epi16(const VecType a, const VecType b);

/**
   m256_add_epi16. This function accepts two vectors `a` and `b` and returns a vector containing
   `a + b`. This function adds `a` and `b` component-wise. Let `c` denote the output vector: then
c[i] = a[i] + b[i]. \param[in] a: the left operand. \param[in] b: the right operand. \return a + b.
**/

inline VecType m256_add_epi16(const VecType a, const VecType b);

/**
     m128_add_epi64. This function accepts two vectors `a` and `b` and adds them pairwise.
     Here the addition occurs across 64-bit entries. This function mimics _mm_add_epi64.
     \param[in] a: the left operand.
     \param[in] b: the right operand,
     \return a + b.
**/
inline SmallVecType m128_add_epi64(const SmallVecType a, const SmallVecType b);

/**
   m256_sub_epi16. This function accepts two vectors `a` and `b` and returns a vector containing
   `a - b`. This function subtracts `b` from `a` component-wise. Let `c` denote the output vector:
then c[i] = a[i] - b[i]. \param[in] a: the left operand. \param[in] b: the right operand. \return a
- b.
**/
inline VecType m256_sub_epi16(const VecType a, const VecType b);

/**
 m256_and_si256. This function accepts two vectors `a` and `b` and returns a vector containing
 their bitwise and. (i.e c[i] = a[i] & b[i] for 0 <= i < 255).
 \param[in] a: the left operand.
 \param[in] b: the right operand.
 \return a & b.
**/
inline VecType m256_and_si256(const VecType a, const VecType b);

/**
   m256_xor_si256. This function accepts two vectors `a` and `b` and returns a vector containing
   their bitwise xor. (i.e c[i] = a[i] ^ b[i] for 0 <= i < 255).
   \param[in] a: the left operand.
   \param[in] b: the right operand.
   \return a ^ b.
**/
inline VecType m256_xor_si256(const VecType a, const VecType b);

/**
   m128_xor_si128. This function accepts two vectors `a` and `b` and returns a vector containing
   their bitwise xor. (i.e c[i] = a[i] ^ b[i] for 0 <= i < 128).
   \param[in] a: the left operand.
   \param[in] b: the right operand.
   \return a ^ b.
**/
inline SmallVecType m128_xor_si128(const SmallVecType a, const SmallVecType b);

/**
   m256_permute4x64_epi64. This function accepts a vector `a`, a compile-time known 8-bit int `mask`
and shuffles the elements in `a` according to `mask`. This function emulates
the_mm256_permute_4x4_epi64 function in AVX2. \tparam[in] mask: the shuffle mask. \param[in] a: the
vector to be permuted. \param[in] mask: the control mask \return `a` shuffled according to `mask`.
**/
template <uint8_t mask> inline VecType m256_permute4x64_epi64(const VecType a);

/**
   m256_permute4x64_epi64_for_hadamard. This function does the exact same as m256_permute_4x4_epi64
but it uses a pre-determined mask (defined inside the function). This function exists solely to
coerce GCC into producing better object code. \param[in] a: the vector to be permuted. \return `a`
shuffled according to some pre-determined mask.
**/
inline VecType m256_permute4x64_epi64_for_hadamard(const VecType a);

/**
   m256_sign_epi16. This function accepts two vectors `a`, `mask` and emulates the _mm256_sign_epi16
function. Let `c` denote the output of this function. Then, for i = 0,...,15 we have: c[i] = -a[i]
if mask[i] < 0 c[i] = 0 if mask[i] == 0 c[i] = a[i] if mask[i] > 1

   In other words, this function negates a[i] when b[i] < 0 and stores a[i] when b[i] > 0.

   \param[in] a: the vector to be negated.
   \param[in] mask: the mask applied for the negation.
   \return a vector as described above.
**/
inline VecType m256_sign_epi16(const VecType a, const VecType mask);

/**
   m256_sign_epi16_ternary. This function is semantically the exact same as m256_sign_epi16.
   The difference is that if you know `mask` is a ternary vector then the sign_epi16 function
   can be made far cheaper than the generic version by a simple multiplication. Thus, if you
   know that `mask` is a ternary vector, then calling this function instead is a good idea.
   Note that the behaviour of this function is undefined* if mask is not ternary.
   \param[in] a: the vector to be negated.
   \param[in] mask: the mask applied for the negation.
   \return a vector negated as above.

   * undefined in the sense that this does not do what you want it to do: it's just a simple
   multiplication.
**/
inline VecType m256_sign_epi16_ternary(const VecType a, const VecType mask);

/**
 m256_broadcast_si128_si256. This function double packs a 128-bit vector into a 256-bit vector.
 In other words, this function produces `out` where out[0:7] == in and out[8:15] == in;
 This function mimics the _mm256_broadcastsi128_si256 function.
 \param[in] in: the vector to broadcast.
 \return a double-packed version of `in`.
**/
inline VecType m256_broadcastsi128_si256(const SmallVecType in);

/**
 * m256_shuffle_epi8.
 *
 * This function accepts two vectors, `in` and `mask`.
 * This function shuffles the bytes in
 * `in` according to the `mask` and returns the result, denoted as c.
 *
 * This corresponds exactly to the behaviour of _mm256_shuffle_epi8.
 *
 * The best way to understand
 * what this function does is to view it as applying _mm_shuffle_epi8 twice:
 * once to the lower lane of `in`, and once to the upper lane of `in`. The
 * description of this function can be found at:
 * https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=shuffle_epi8&expand=5153.
 *
 * This function will only work if `in` and `mask` are not equal.
 * \param[in] in: the vector to be manipulated.
 * \param[in] mask: the mask to use.
 * \return `in` shuffled according to `mask`.
 */
inline VecType m256_shuffle_epi8(const VecType in, const VecType mask);

/**
 * m128_shuffle_epi.
 * This function accepts two vectors, `in` and `mask`.
 * This function shuffles the bytes in
 * `in` according to the `mask` and returns the result, denoted as c.
 *
 * This corresponds exactly to the behaviour of _mm128_shuffle_epi8.
 * \param[in] in: the vector to be manipulated.
 * \param[in] mask: the mask to use.
 * \return `in` shuffled according to `mask`.
 */
inline SmallVecType m128_shuffle_epi8(const SmallVecType in, const SmallVecType mask);

/**
   m256_hadamard16_epi16. This function computes the hadamard transform of `x1` and stores the
result in `r1`. This function does not directly mimick an existing Intel Intrinsic: instead, it
simply moves the functionality from FastHadamardLSH to this namespace. \param[in] x1: the vector to
be transformed. \param[out] r1: the location of the result.
**/
inline void m256_hadamard16_epi16(const VecType x1, VecType &r1);

/**
   m256_mix. This function swaps v0[i] and v1[i] iff mask[i] for 0 <= i < 255.
   \param[in] v0: the first vector.
   \param[in] v1: the second vector.
   \param[in] mask: the control mask.
**/
inline void m256_mix(VecType &v0, VecType &v1, const VecType &mask);

/**
 * get_randomness. This function is not SIMD in nature: instead, given two sources of randomness,
 * prg_state and key, it produces a 128-bit random value.
 *
 * In it's naive form, this is based on xorshift, which is a simple (fast) random number generator.
 *
 * However, in some situations it can be a little bit slow compared to the available alternatives.
 * As a result -- and where applicable -- we delegate to the aes_enc function, which is really
 * fast, to produce randomness. (This trick was originally in
 * https://github.com/lducas/AVX2-BDGL-bucketer.)
 *
 * WARNING WARNING WARNING: this should _not_ be used for any situation
 * where you need true randomness. It may be fast -- it may even appear reasonable --
 * but please, don't use this for anything that needs anything sensible.
 * Here we *just* use it for speed: it's fast and small, but beyond that there's no reason
 * to use it. You have been warned.
 * \param[in] prg_state: the state of the random number generator.
 * \param[in] key: extra part of state for the rng.
 * \param[in] extra_state: extra mutable state for the rng. Only used in GCC mode.
 * \return a new random piece of data.
 */
inline SmallVecType m128_random_state(SmallVecType prg_state, SmallVecType key,
                                      SmallVecType *extra_state);

/**
 * m256_permute_epi16. The goal of this function is to permute the input vector v,
 * according to the randomness from prg_state & the tailmask.  Note that this function is a
 * specialisation of the broader m256_permute_epi16. \tparam[in] regs: the number of registers to
 * use. \param[in] v: a pointer to a sequence of VecTypes. \param[in] prgstate: this is the state of
 * the prg for the current hashing round \param[in] tailmask: a mask for handling mixing when the
 * length of v is not a multiple of 16. \param[in] key: the rest of the state of the random number
 * generator.
 */
template <int regs_>
inline void m256_permute_epi16(VecType *const v, SmallVecType &prg_state, const VecType tailmask,
                               const SmallVecType &key, SmallVecType *extra_state);

};  // namespace SimdGeneric

// Inline definitions are in this file.
#include "simd_generic.inl"
#endif