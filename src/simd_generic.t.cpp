// This file contains tests for the functions given in simd.hpp against generic
// versions of the intel functions. These are written by hand and meant to be used
// to test the bucketer on arbitrary future platforms, but without dependence on particular
// ISAs.

#include "simd_generic.hpp"
#include "gtest/gtest.h"

#include "fht_lsh_old.h"  // Included to allow access to AVX2 stuff directly.
#include <immintrin.h>
#include <random>

using VecType = SimdGeneric::VecType;

TEST(SimdGeneric, testStoreEuSi256)
{
  int16_t in[16];
  for (unsigned i = 0; i < 16; i++)
  {
    in[i] = rand();
  }

  __m256i m_a;
  VecType a;

  // Now build vectors. This loads in reverse, so they're
  // the same as what you'd expect from the m256i elements
  memcpy(&a, in, sizeof(VecType));
  m_a = _mm256_loadu_si256(reinterpret_cast<__m256i *>(in));

  // Assert that the copy worked.
  ASSERT_EQ(memcmp(&a, &m_a, sizeof(m_a)), 0);

  // Now we'll create two new temps to copy into & check that it worked
  int16_t m_out[16], out[16];

  SimdGeneric::m256_storeu_si256(out, a);
  _mm256_storeu_si256(reinterpret_cast<__m256i *>(m_out), m_a);
  EXPECT_EQ(memcmp(&out, &m_out, sizeof(__m256i)), 0);
}

TEST(SimdGeneric, testLoadSi256)
{
  int16_t in[16];
  for (unsigned i = 0; i < 16; i++)
  {
    in[i] = rand();
  }

  // Now check that stores work
  auto c1 = SimdGeneric::m256_loadu_si256(in);
  auto c2 = _mm256_loadu_si256(reinterpret_cast<__m256i *>(in));
  EXPECT_EQ(memcmp(&c1, &c2, sizeof(c1)), 0);
}

TEST(SimdGeneric, testGetEpi64)
{
  // This function just checks that the swapping works during set.
  const int64_t first  = rand();
  const int64_t second = rand();
  auto c1              = SimdGeneric::m128_set_epi64x(second, first);
  auto c2              = _mm_set_epi64x(second, first);

  ASSERT_EQ(memcmp(&c1, &c2, sizeof(c1)), 0);
  EXPECT_EQ(SimdGeneric::m128_extract_epi64<0>(c1), _mm_extract_epi64(c2, 0));
  EXPECT_EQ(SimdGeneric::m128_extract_epi64<1>(c1), _mm_extract_epi64(c2, 1));
}

TEST(SimdGeneric, testSetSi256)
{
  // This function just checks that the swapping works during set.
  const int64_t first  = rand();
  const int64_t second = rand();
  auto c1              = SimdGeneric::m128_set_epi64x(second, first);
  auto c2              = _mm_set_epi64x(second, first);

  ASSERT_EQ(memcmp(&c1, &c2, sizeof(c1)), 0);
  EXPECT_EQ(SimdGeneric::m128_extract_epi64<0>(c1), _mm_extract_epi64(c2, 0));
  EXPECT_EQ(SimdGeneric::m128_extract_epi64<1>(c1), _mm_extract_epi64(c2, 1));
}

// Now to make life easier we'll set-up a routine that does everything for us.
class SimdFixture : public ::testing::Test
{
protected:
  void SetUp() override
  {

    for (unsigned i = 0; i < 16; i++)
    {
      a_arr[i] = rand() % 100;
      b_arr[i] = rand() % 100;
    }

    // Now build vectors. This loads in reverse, so they're
    // the same as what you'd expect from the m256i elements
    memcpy(&a, a_arr, sizeof(VecType));
    memcpy(&b, b_arr, sizeof(VecType));

    // Convert to m256i
    am = _mm256_loadu_si256(reinterpret_cast<__m256i *>(a_arr));
    bm = _mm256_loadu_si256(reinterpret_cast<__m256i *>(b_arr));
    assert(memcmp(&am, &a, sizeof(VecType)) == 0);
    assert(memcmp(&bm, &b, sizeof(VecType)) == 0);
  }

  template <typename intrin_type, typename VectorType>
  void random_vec(intrin_type &m_vec, VectorType &vec, int lower, int upper)
  {
    // Calculate the number of elements to generate
    constexpr unsigned nr_elems = sizeof(VectorType) / sizeof(int16_t);

    // For obvious reasons I hope
    static_assert(sizeof(intrin_type) == sizeof(VectorType), "Error: mismatched vector types");
    static_assert(std::is_same<intrin_type, __m256i>::value ||
                      std::is_same<intrin_type, __m128i>::value,
                  "Error: unsupported vector type.");

    int16_t rand_vec[nr_elems];

    std::uniform_int_distribution<> dist(lower, upper);
    for (unsigned i = 0; i < nr_elems; i++)
    {
      rand_vec[i] = dist(gen);
    }

    // Copy over the results
    memcpy(&vec, rand_vec, sizeof(VectorType));

    // Need to do this delegation sadly. This means we have to use constexpr if (or at least we
    // could SFINAE it but this is test code! Not worth the hassle)
    if constexpr (nr_elems == 16)
    {
      m_vec = _mm256_loadu_si256(reinterpret_cast<__m256i *>(rand_vec));
    }
    else
    {
      m_vec = _mm_load_si128(reinterpret_cast<__m128i *>(rand_vec));
    }
    // enforce that they're the same
    assert(memcmp(&m_vec, &vec, sizeof(VectorType)) == 0);
  }

  std::random_device rd;
  std::mt19937 gen{rd()};

  VecType a, b;
  int16_t a_arr[16], b_arr[16];
  __m256i am, bm;
};

TEST_F(SimdFixture, testM256TestzSi256)
{
  auto c1 = SimdGeneric::m256_testz_si256(a, b);
  auto c2 = _mm256_testz_si256(am, bm);
  EXPECT_EQ(c1, c2);
}

TEST_F(SimdFixture, testAbsEpi16)
{
  auto c1 = SimdGeneric::m256_abs_epi16(a);
  auto c2 = _mm256_abs_epi16(am);
  EXPECT_EQ(memcmp(&c1, &c2, sizeof(c1)), 0);
}

TEST_F(SimdFixture, testCmpgtEpi16)
{
  auto c1 = SimdGeneric::m256_cmpgt_epi16(a, b);
  auto c2 = _mm256_cmpgt_epi16(am, bm);
  EXPECT_EQ(memcmp(&c1, &c2, sizeof(VecType)), 0);
}

TEST_F(SimdFixture, testSlliEpi16)
{
  const int amount = rand() % 16;
  auto c1          = SimdGeneric::m256_slli_epi16(a, amount);
  auto c2          = _mm256_slli_epi16(am, amount);
  EXPECT_EQ(memcmp(&c1, &c2, sizeof(VecType)), 0);
}

TEST_F(SimdFixture, testHaddEpi16)
{
  auto c1 = SimdGeneric::m256_hadd_epi16(a, b);
  auto c2 = _mm256_hadd_epi16(am, bm);
  EXPECT_EQ(memcmp(&c1, &c2, sizeof(VecType)), 0);
}

TEST_F(SimdFixture, testAddEpi16)
{
  auto c1 = SimdGeneric::m256_add_epi16(a, b);
  auto c2 = _mm256_add_epi16(am, bm);
  EXPECT_EQ(memcmp(&c1, &c2, sizeof(VecType)), 0);
}

TEST_F(SimdFixture, testAddEpi64)
{
  // Note: since this is 128 bits, we'll just make new arrays
  int64_t arr1[2]{rand(), rand()};
  int64_t arr2[2]{rand(), rand()};
  const auto am = _mm_load_si128(reinterpret_cast<__m128i *>(&arr1));
  const auto bm = _mm_load_si128(reinterpret_cast<__m128i *>(&arr2));

  const auto a = SimdGeneric::m128_set_epi64x(arr1[1], arr1[0]);
  const auto b = SimdGeneric::m128_set_epi64x(arr2[1], arr2[0]);

  const auto c1 = SimdGeneric::m128_add_epi64(a, b);
  const auto c2 = _mm_add_epi64(am, bm);
  EXPECT_EQ(memcmp(&c1, &c2, sizeof(c1)), 0);
}

TEST_F(SimdFixture, testSubEpi16)
{
  auto c1 = SimdGeneric::m256_sub_epi16(a, b);
  auto c2 = _mm256_sub_epi16(am, bm);
  EXPECT_EQ(memcmp(&c1, &c2, sizeof(VecType)), 0);
}

TEST_F(SimdFixture, testAndSi256)
{
  auto c1 = SimdGeneric::m256_and_si256(a, b);
  auto c2 = _mm256_and_si256(am, bm);
  EXPECT_EQ(memcmp(&c1, &c2, sizeof(VecType)), 0);
}

TEST_F(SimdFixture, testXorSi256)
{
  auto c1 = SimdGeneric::m256_xor_si256(a, b);
  auto c2 = _mm256_xor_si256(am, bm);
  EXPECT_EQ(memcmp(&c1, &c2, sizeof(VecType)), 0);
}

TEST_F(SimdFixture, testXorSi128)
{
  // Note: since this is 128 bits, we'll just make new arrays
  int64_t arr1[2]{rand(), rand()};
  int64_t arr2[2]{rand(), rand()};
  const auto am = _mm_load_si128(reinterpret_cast<__m128i *>(&arr1));
  const auto bm = _mm_load_si128(reinterpret_cast<__m128i *>(&arr2));

  const auto a = SimdGeneric::m128_set_epi64x(arr1[1], arr1[0]);
  const auto b = SimdGeneric::m128_set_epi64x(arr2[1], arr2[0]);

  const auto c1 = SimdGeneric::m128_xor_si128(a, b);
  const auto c2 = _mm_xor_si128(am, bm);
  EXPECT_EQ(memcmp(&c1, &c2, sizeof(c1)), 0);
}

TEST_F(SimdFixture, testPermute4x64Epi64)
{
  // We need to use a fixed mask for Intel to be happy, so...
  constexpr int mask{0b01001110};
  auto c1 = SimdGeneric::m256_permute4x64_epi64<mask>(a);
  auto c2 = _mm256_permute4x64_epi64(am, mask);
  EXPECT_EQ(memcmp(&c1, &c2, sizeof(VecType)), 0);
}

TEST_F(SimdFixture, testPermute4x64Epi64ForHadamard)
{
  // We don't need a mask here, since it's hardcoded
  auto c1 = SimdGeneric::m256_permute4x64_epi64_for_hadamard(a);
  auto c2 = _mm256_permute4x64_epi64(am, 0b01001110);
  EXPECT_EQ(memcmp(&c1, &c2, sizeof(VecType)), 0);
}

TEST_F(SimdFixture, testSignEpi16)
{
  // Generate a random mask of elements.
  // These should ideally by both positive and negative, so we'll generate in a range of [-5, 5]
  __m256i m_mask;
  VecType mask;
  random_vec(m_mask, mask, -5, 5);
  // Now we can do the *actual* work
  auto c1 = SimdGeneric::m256_sign_epi16(a, mask);
  auto c2 = _mm256_sign_epi16(am, m_mask);
  EXPECT_EQ(memcmp(&c1, &c2, sizeof(VecType)), 0);
}

TEST_F(SimdFixture, testSignEpi16Ternary)
{
  // This is a special variant of the above function.
  // Essentially, to make things faster we just do a multiplication rather than all of the
  // shuffling. This mandates using some slightly different code, but the logic is the same.
  __m256i m_mask;
  VecType mask;
  random_vec(m_mask, mask, -1, 1);
  // Now we can do the *actual* work
  auto c1 = SimdGeneric::m256_sign_epi16_ternary(a, mask);
  auto c2 = _mm256_sign_epi16(am, m_mask);
  EXPECT_EQ(memcmp(&c1, &c2, sizeof(VecType)), 0);
}

TEST_F(SimdFixture, testBroadcastSi128Si256)
{
  __m128i m_broadcast;
  SimdGeneric::Vec8s broadcast;

  // No importance on the bounds
  random_vec(m_broadcast, broadcast, 0, 100);

  auto c1 =
      SimdGeneric::m256_broadcastsi128_si256(static_cast<SimdGeneric::SmallVecType>(broadcast));
  auto c2 = _mm256_broadcastsi128_si256(m_broadcast);
  EXPECT_EQ(memcmp(&c1, &c2, sizeof(VecType)), 0);
}

TEST_F(SimdFixture, testM128iShuffleEpi8)
{
  // Generate a random 16-element shuffle mask.
  // The entries don't really matter.
  __m128i m_mask;
  SimdGeneric::SmallVecType mask;
  static_assert(sizeof(mask) == sizeof(m_mask), "Error: Vec8s is too big!");

  random_vec(m_mask, mask, -100, 100);

  // Need to generate random non-mask vectors too
  __m128i m_actual;
  SimdGeneric::SmallVecType actual;
  random_vec(m_actual, actual, -100, 100);

  // Now check the shuffling
  auto c1 = SimdGeneric::m128_shuffle_epi8(actual, mask);
  auto c2 = _mm_shuffle_epi8(m_actual, m_mask);

  EXPECT_EQ(memcmp(&c1, &c2, sizeof(c1)), 0);
}

TEST_F(SimdFixture, testM256iShuffleEpi8)
{
  // Generate a random 32-element shuffle mask.
  // The entries don't really matter.
  __m256i m_mask;
  VecType mask;
  random_vec(m_mask, mask, -100, 100);

  // Now check the shuffling
  auto c1 = SimdGeneric::m256_shuffle_epi8(a, static_cast<SimdGeneric::VecType>(mask));
  auto c2 = _mm256_shuffle_epi8(am, m_mask);
  EXPECT_EQ(memcmp(&c1, &c2, sizeof(VecType)), 0);
}

TEST_F(SimdFixture, testM256Permute2Regs)
{
  // This tests that the permutations of some vectors does the same thing with both
  // our version and the FastHadamardLSH version. Note that this will only hold exactly for one
  // iteration: the randomness will be different, since we can't use AES to randomise on all
  // machines.

  // Generate a random key to use
  __m128i m_key, m_prg_state;
  SimdGeneric::SmallVecType key, prg_state;

  auto extra_state   = key;
  auto m_extra_state = m_key;

  // Again, keys don't really matter here.
  random_vec(m_key, key, -100, 100);
  random_vec(m_prg_state, prg_state, -100, 100);

  // Now we'll generate a series of random vectors to permute.
  std::array<VecType, 100> vecs;
  std::array<__m256i, 100> m_vecs;

  for (unsigned i = 0; i < 100; i++)
  {
    random_vec(m_vecs[i], vecs[i], -100, 100);
  }

  // This is enforced by rand_vec, but to be sure
  ASSERT_EQ(memcmp(&vecs[0], &m_vecs[0], sizeof(vecs)), 0);

  // Now we'll call the hashing routines
  // The exact tail mask doesn't matter
  FastHadamardLSH::m256_permute_epi16<2>(&m_vecs[0], m_prg_state, tailmasks[0], m_key,
                                         &m_extra_state);
  SimdGeneric::m256_permute_epi16<2>(&vecs[0], prg_state, SimdGeneric::tailmasks[0], key,
                                     &extra_state);

  // And finally, compare that the vectors are the same.
  EXPECT_EQ(memcmp(&vecs[0], &m_vecs[0], sizeof(vecs)), 0);
}

TEST_F(SimdFixture, testM256PermuteNRegs)
{
  // Same as above but for many registers.
  // Thankfully FHT only requires 2 =< n < 8
  // And we can write the rest by hand
#define testM256PermuteRegs(n)                                                                     \
  {                                                                                                \
    __m128i m_key, m_prg_state;                                                                    \
    SimdGeneric::SmallVecType key, prg_state;                                                      \
    random_vec(m_key, key, -100, 100);                                                             \
    random_vec(m_prg_state, prg_state, -100, 100);                                                 \
    auto extra_state   = key;                                                                      \
    auto m_extra_state = m_key;                                                                    \
    std::array<VecType, 100> vecs;                                                                 \
    std::array<__m256i, 100> m_vecs;                                                               \
    for (unsigned i = 0; i < 100; i++)                                                             \
    {                                                                                              \
      random_vec(m_vecs[i], vecs[i], -100, 100);                                                   \
    }                                                                                              \
    ASSERT_EQ(memcmp(&vecs[0], &m_vecs[0], sizeof(vecs)), 0);                                      \
    FastHadamardLSH::m256_permute_epi16<n>(&m_vecs[0], m_prg_state, tailmasks[n], m_key,           \
                                           &m_extra_state);                                        \
    SimdGeneric::m256_permute_epi16<n>(&vecs[0], prg_state, SimdGeneric::tailmasks[n], key,        \
                                       &extra_state);                                              \
    EXPECT_EQ(memcmp(&vecs[0], &m_vecs[0], sizeof(vecs)), 0);                                      \
  }

  testM256PermuteRegs(3) testM256PermuteRegs(4) testM256PermuteRegs(5) testM256PermuteRegs(6)
      testM256PermuteRegs(7)
}
