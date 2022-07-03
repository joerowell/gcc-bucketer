#include "simd.hpp"
#include "simd_generic.hpp"

#ifdef HAVE_AVX2
#include "fht_lsh_old.h"   // Included to allow access to AVX2 stuff directly.
#include "simd_intel.hpp"  // Included to allow access directly to the intrinsics wrapper.
#endif

#include "gtest/gtest.h"
#include <random>
#include <type_traits>

template <typename...> struct TypeList
{
};

template <typename A> struct SimdFixture;

// Now to make life easier we'll set-up a routine that does everything for us.
template <typename First, typename Second>
struct SimdFixture<TypeList<First, Second>> : public ::testing::Test
{
public:
  using FirstNamespace  = First;
  using SecondNamespace = Second;

  using FirstVecType  = typename First::VecType;
  using SecondVecType = typename Second::VecType;
  static_assert(sizeof(FirstVecType) == sizeof(SecondVecType),
                "Error: mismatched sizes on type arguments");

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
    af = First::build_vec_type(a_arr);
    bf = First::build_vec_type(b_arr);
    as = Second::build_vec_type(a_arr);
    bs = Second::build_vec_type(b_arr);

    assert(memcmp(&af, &as, sizeof(FirstVecType)) == 0);
    assert(memcmp(&bf, &bs, sizeof(FirstVecType)) == 0);
  }

  static constexpr auto vec_size = sizeof(FirstVecType);

  std::random_device rd;
  std::mt19937 gen{rd()};

  FirstVecType af, bf;
  SecondVecType as, bs;
  int16_t a_arr[16], b_arr[16];
};

#ifdef HAVE_AVX2
using SimdTypes = testing::Types<TypeList<SimdIntel, Simd>, TypeList<Simd, SimdGeneric>>;
#else
using SimdTypes = testing::Types<TypeList<SimdGeneric, Simd>>;
#endif

TYPED_TEST_SUITE(SimdFixture, SimdTypes);

TYPED_TEST(SimdFixture, testBuildVecType)
{
  using First  = typename SimdFixture<gtest_TypeParam_>::FirstNamespace;
  using Second = typename SimdFixture<gtest_TypeParam_>::SecondNamespace;

  int16_t a_arr[16];
  auto a = First::build_vec_type(a_arr);
  auto b = Second::build_vec_type(a_arr);
  EXPECT_EQ(memcmp(a_arr, &a, sizeof(a)), 0);
  EXPECT_EQ(memcmp(a_arr, &b, sizeof(b)), 0);
}

TYPED_TEST(SimdFixture, testLoadSi256)
{
  using First  = typename SimdFixture<gtest_TypeParam_>::FirstNamespace;
  using Second = typename SimdFixture<gtest_TypeParam_>::SecondNamespace;

  int16_t in[16];
  for (unsigned i = 0; i < 16; i++)
  {
    in[i] = rand();
  }

  // Now check that stores work

  auto c1 = First::m256_loadu_si256(in);
  auto c2 = Second::m256_loadu_si256(in);
  EXPECT_EQ(memcmp(&c1, &c2, sizeof(c1)), 0);
}

TYPED_TEST(SimdFixture, testBuildVecTypeSingular)
{
  using First  = typename SimdFixture<gtest_TypeParam_>::FirstNamespace;
  using Second = typename SimdFixture<gtest_TypeParam_>::SecondNamespace;

  const int16_t T = rand();
  auto a          = First::build_vec_type(T);
  auto threshold  = Second::build_vec_type(T);
  EXPECT_EQ(memcmp(&threshold, &a, sizeof(a)), 0);
}

TYPED_TEST(SimdFixture, testSetEpi64)
{
  using First  = typename SimdFixture<gtest_TypeParam_>::FirstNamespace;
  using Second = typename SimdFixture<gtest_TypeParam_>::SecondNamespace;

  const int64_t first  = rand();
  const int64_t second = rand();
  auto c1              = First::m128_set_epi64x(second, first);
  auto c2              = Second::m128_set_epi64x(second, first);
  EXPECT_EQ(memcmp(&c1, &c2, sizeof(c1)), 0);
}

TYPED_TEST(SimdFixture, testSetEpix64)
{
  using First  = typename SimdFixture<gtest_TypeParam_>::FirstNamespace;
  using Second = typename SimdFixture<gtest_TypeParam_>::SecondNamespace;

  const uint64_t first  = rand();
  const uint64_t second = rand();
  auto c1               = First::m128_set_epi64x(second, first);
  auto c2               = Second::m128_set_epi64x(second, first);
  EXPECT_EQ(memcmp(&c1, &c2, sizeof(c1)), 0);
}

TYPED_TEST(SimdFixture, testGetEpi64)
{
  using First  = typename SimdFixture<gtest_TypeParam_>::FirstNamespace;
  using Second = typename SimdFixture<gtest_TypeParam_>::SecondNamespace;

  // This function just checks that the swapping works during set.
  const int64_t first  = rand();
  const int64_t second = rand();
  auto c1              = First::m128_set_epi64x(second, first);
  auto c2              = Second::m128_set_epi64x(second, first);

  ASSERT_EQ(memcmp(&c1, &c2, sizeof(c1)), 0);
  EXPECT_EQ(First::template m128_extract_epi64<0>(c1), Second::template m128_extract_epi64<0>(c2));
  EXPECT_EQ(First::template m128_extract_epi64<1>(c1), Second::template m128_extract_epi64<1>(c2));
}

TYPED_TEST(SimdFixture, testExtractEpi64)
{

  using First  = typename SimdFixture<gtest_TypeParam_>::FirstNamespace;
  using Second = typename SimdFixture<gtest_TypeParam_>::SecondNamespace;

  const auto pos = 3;  // This has to be an immediate!
  auto c1        = First::template m256_extract_epi64<pos>(this->af);
  auto c2        = Second::template m256_extract_epi64<pos>(this->as);
  EXPECT_EQ(memcmp(&c1, &c2, sizeof(c1)), 0);
}

TYPED_TEST(SimdFixture, testAbsEpi16)
{
  using First  = typename SimdFixture<gtest_TypeParam_>::FirstNamespace;
  using Second = typename SimdFixture<gtest_TypeParam_>::SecondNamespace;

  auto c1 = First::m256_abs_epi16(this->af);
  auto c2 = Second::m256_abs_epi16(this->as);
  EXPECT_EQ(memcmp(&c1, &c2, sizeof(c1)), 0);
}

TYPED_TEST(SimdFixture, testExtractEpi64Small)
{
  using First   = typename SimdFixture<gtest_TypeParam_>::FirstNamespace;
  using Second  = typename SimdFixture<gtest_TypeParam_>::SecondNamespace;
  using First8s = typename First::Vec8s;
  using Sec8s   = typename Second::Vec8s;

  static_assert(sizeof(First8s) == sizeof(Sec8s), "Error: size mismatch");
  constexpr auto pos = 1;  // This has to be an immediate!
  // Turn each into a smaller vector
  First8s cp;
  Sec8s v_cp;

  memcpy(&cp, &this->as, sizeof(cp));
  memcpy(&v_cp, &this->af, sizeof(v_cp));

  auto c1 = First::template m128_extract_epi64<pos>(cp);
  auto c2 = Second::template m128_extract_epi64<pos>(v_cp);
  EXPECT_EQ(memcmp(&c1, &c2, sizeof(c1)), 0);
}

TYPED_TEST(SimdFixture, testCmpgtEpi16)
{
  using First  = typename SimdFixture<gtest_TypeParam_>::FirstNamespace;
  using Second = typename SimdFixture<gtest_TypeParam_>::SecondNamespace;

  auto c1 = First::m256_cmpgt_epi16(this->af, this->bf);
  auto c2 = Second::m256_cmpgt_epi16(this->as, this->bs);
  EXPECT_EQ(memcmp(&c1, &c2, sizeof(c1)), 0);
}

TYPED_TEST(SimdFixture, testSlliEpi16)
{
  using First  = typename SimdFixture<gtest_TypeParam_>::FirstNamespace;
  using Second = typename SimdFixture<gtest_TypeParam_>::SecondNamespace;

  const int amount = rand() % 16;
  auto c1          = First::m256_slli_epi16(this->af, amount);
  auto c2          = Second::m256_slli_epi16(this->as, amount);
  EXPECT_EQ(memcmp(&c1, &c2, sizeof(c1)), 0);
}

TYPED_TEST(SimdFixture, testHaddEpi16)
{
  using First  = typename SimdFixture<gtest_TypeParam_>::FirstNamespace;
  using Second = typename SimdFixture<gtest_TypeParam_>::SecondNamespace;

  auto c1 = First::m256_hadd_epi16(this->af, this->bf);
  auto c2 = Second::m256_hadd_epi16(this->as, this->bs);
  EXPECT_EQ(memcmp(&c1, &c2, sizeof(c1)), 0);
}

TYPED_TEST(SimdFixture, testAddEpi16)
{
  using First  = typename SimdFixture<gtest_TypeParam_>::FirstNamespace;
  using Second = typename SimdFixture<gtest_TypeParam_>::SecondNamespace;

  auto c1 = First::m256_add_epi16(this->af, this->bf);
  auto c2 = Second::m256_add_epi16(this->as, this->bs);
  EXPECT_EQ(memcmp(&c1, &c2, sizeof(c1)), 0);
}

TYPED_TEST(SimdFixture, testSubEpi16)
{
  using First  = typename SimdFixture<gtest_TypeParam_>::FirstNamespace;
  using Second = typename SimdFixture<gtest_TypeParam_>::SecondNamespace;

  auto c1 = First::m256_sub_epi16(this->af, this->bf);
  auto c2 = Second::m256_sub_epi16(this->as, this->bs);
  EXPECT_EQ(memcmp(&c1, &c2, sizeof(c1)), 0);
}

TYPED_TEST(SimdFixture, testAndSi256)
{
  using First  = typename SimdFixture<gtest_TypeParam_>::FirstNamespace;
  using Second = typename SimdFixture<gtest_TypeParam_>::SecondNamespace;

  auto c1 = First::m256_and_si256(this->af, this->bf);
  auto c2 = Second::m256_and_si256(this->as, this->bs);
  EXPECT_EQ(memcmp(&c1, &c2, sizeof(c1)), 0);
}

TYPED_TEST(SimdFixture, testXorSi256)
{
  using First  = typename SimdFixture<gtest_TypeParam_>::FirstNamespace;
  using Second = typename SimdFixture<gtest_TypeParam_>::SecondNamespace;

  auto c1 = First::m256_xor_si256(this->af, this->bf);
  auto c2 = Second::m256_xor_si256(this->as, this->bs);
  EXPECT_EQ(memcmp(&c1, &c2, sizeof(c1)), 0);
}

/*
TYPED_TEST(SimdFixture, testXorSi128)
{

}

TEST_F(SimdFixture, testAddEpi64)
{

}
*/

TYPED_TEST(SimdFixture, testPermute4x64Epi64)
{
  using First  = typename SimdFixture<gtest_TypeParam_>::FirstNamespace;
  using Second = typename SimdFixture<gtest_TypeParam_>::SecondNamespace;

  // We need to use a fixed mask for Intel to be happy, so...
  constexpr int mask{0b01001110};
  auto c1 = First::template m256_permute4x64_epi64<mask>(this->af);
  auto c2 = Second::template m256_permute4x64_epi64<mask>(this->as);
  EXPECT_EQ(memcmp(&c1, &c2, sizeof(c1)), 0);
}

TYPED_TEST(SimdFixture, testPermute4x64Epi64ForHadamard)
{
  using First  = typename SimdFixture<gtest_TypeParam_>::FirstNamespace;
  using Second = typename SimdFixture<gtest_TypeParam_>::SecondNamespace;

  // We don't need a mask here, since it's hardcoded
  auto c1 = First::m256_permute4x64_epi64_for_hadamard(this->af);
  auto c2 = Second::m256_permute4x64_epi64_for_hadamard(this->as);
  EXPECT_EQ(memcmp(&c1, &c2, sizeof(c1)), 0);
}

TYPED_TEST(SimdFixture, testSignEpi16)
{
  // Generate a random mask of elements.
  // These should ideally by both positive and negative, so we'll generate in a range of [-5, 5]
  using First        = typename SimdFixture<gtest_TypeParam_>::FirstNamespace;
  using Second       = typename SimdFixture<gtest_TypeParam_>::SecondNamespace;
  using FirstVecType = typename First::VecType;
  using SecVecType   = typename Second::VecType;

  int16_t arr[16];
  for (unsigned i = 0; i < 16; i++)
  {
    arr[i] = rand() % (11) - 5;
  }

  auto mask   = First::build_vec_type(arr);
  auto m_mask = Second::build_vec_type(arr);

  // Now we can do the *actual* work
  auto c1 = First::m256_sign_epi16(this->af, mask);
  auto c2 = Second::m256_sign_epi16(this->as, m_mask);
  EXPECT_EQ(memcmp(&c1, &c2, sizeof(c1)), 0);
}

TYPED_TEST(SimdFixture, testSignEpi16Ternary)
{
  // This is a special variant of the above function.
  // Essentially, to make things faster we just do a multiplication rather than all of the
  // shuffling. This mandates using some slightly different code, but the logic is the same.
  using First        = typename SimdFixture<gtest_TypeParam_>::FirstNamespace;
  using Second       = typename SimdFixture<gtest_TypeParam_>::SecondNamespace;
  using FirstVecType = typename First::VecType;
  using SecVecType   = typename Second::VecType;

  int16_t arr[16];
  for (unsigned i = 0; i < 16; i++)
  {
    arr[i] = rand() % (1) - 1;
  }

  auto mask   = First::build_vec_type(arr);
  auto m_mask = Second::build_vec_type(arr);

  // Now we can do the *actual* work
  auto c1 = First::m256_sign_epi16_ternary(this->af, mask);
  auto c2 = Second::m256_sign_epi16(this->as, m_mask);
  EXPECT_EQ(memcmp(&c1, &c2, sizeof(c1)), 0);
}

TYPED_TEST(SimdFixture, testBroadcastSi128Si256)
{

  using First   = typename SimdFixture<gtest_TypeParam_>::FirstNamespace;
  using Second  = typename SimdFixture<gtest_TypeParam_>::SecondNamespace;
  using First8s = typename First::SmallVecType;
  using Sec8s   = typename Second::SmallVecType;

  // No importance on the bounds
  int16_t arr[8];
  for (unsigned i = 0; i < 8; i++)
  {
    arr[i] = rand();
  }

  auto mask   = First::build_small_vec_type(arr);
  auto m_mask = Second::build_small_vec_type(arr);

  auto c1 = First::m256_broadcastsi128_si256(mask);
  auto c2 = Second::m256_broadcastsi128_si256(m_mask);
  EXPECT_EQ(memcmp(&c1, &c2, sizeof(c1)), 0);
}

TYPED_TEST(SimdFixture, testHadamard16Epi16)
{
  using First         = typename SimdFixture<gtest_TypeParam_>::FirstNamespace;
  using Second        = typename SimdFixture<gtest_TypeParam_>::SecondNamespace;
  using FirstVecType  = typename First::VecType;
  using SecondVecType = typename Second::VecType;

  FirstVecType c1;
  SecondVecType c2;

  First::m256_hadamard16_epi16(this->af, c1);
  Second::m256_hadamard16_epi16(this->as, c2);
  EXPECT_EQ(memcmp(&c1, &c2, sizeof(c1)), 0);
}

TYPED_TEST(SimdFixture, testHadamard32Epi16)
{
  using First         = typename SimdFixture<gtest_TypeParam_>::FirstNamespace;
  using Second        = typename SimdFixture<gtest_TypeParam_>::SecondNamespace;
  using FirstVecType  = typename First::VecType;
  using SecondVecType = typename Second::VecType;

  FirstVecType res1, res2;
  SecondVecType res1_v, res2_v;

  First::m256_hadamard32_epi16(this->af, this->bf, res1, res2);
  Second::m256_hadamard32_epi16(this->as, this->bs, res1_v, res2_v);
  EXPECT_EQ(memcmp(&res1, &res1_v, sizeof(res1)), 0);
  EXPECT_EQ(memcmp(&res2, &res2_v, sizeof(res2)), 0);
}

TYPED_TEST(SimdFixture, testM256iMix)
{
  using First  = typename SimdFixture<gtest_TypeParam_>::FirstNamespace;
  using Second = typename SimdFixture<gtest_TypeParam_>::SecondNamespace;

  int16_t arr[16];
  for (unsigned i = 0; i < 16; i++)
  {
    arr[i] = rand() % (201) - 100;
  }

  auto m_mask = First::build_vec_type(arr);
  auto mask   = Second::build_vec_type(arr);

  First::m256_mix(this->af, this->bf, m_mask);
  Second::m256_mix(this->as, this->bs, mask);
  EXPECT_EQ(memcmp(&this->bf, &this->bs, sizeof(this->bs)), 0);
  EXPECT_EQ(memcmp(&this->af, &this->as, sizeof(this->as)), 0);
}

TYPED_TEST(SimdFixture, testM128iShuffleEpi8)
{
  using First  = typename SimdFixture<gtest_TypeParam_>::FirstNamespace;
  using Second = typename SimdFixture<gtest_TypeParam_>::SecondNamespace;

  int16_t arr[8];
  for (unsigned i = 0; i < 8; i++)
  {
    arr[i] = rand() % (201) - 100;
  }

  auto m_mask = First::build_small_vec_type(arr);
  auto mask   = Second::build_small_vec_type(arr);

  for (unsigned i = 0; i < 8; i++)
  {
    arr[i] = rand() % (201) - 100;
  }

  auto m_actual = First::build_small_vec_type(arr);
  auto actual   = Second::build_small_vec_type(arr);
  auto c1       = First::m128_shuffle_epi8(m_actual, m_mask);
  auto c2       = Second::m128_shuffle_epi8(actual, mask);
  EXPECT_EQ(memcmp(&c1, &c2, sizeof(c1)), 0);
}

TYPED_TEST(SimdFixture, testM256iShuffleEpi8)
{
  using First  = typename SimdFixture<gtest_TypeParam_>::FirstNamespace;
  using Second = typename SimdFixture<gtest_TypeParam_>::SecondNamespace;

  int16_t arr[16];
  for (unsigned i = 0; i < 16; i++)
  {
    arr[i] = rand() % (201) - 100;
  }

  auto m_mask = First::build_vec_type(arr);
  auto mask   = Second::build_vec_type(arr);

  for (unsigned i = 0; i < 16; i++)
  {
    arr[i] = rand() % (201) - 100;
  }

  auto m_actual = First::build_vec_type(arr);
  auto actual   = Second::build_vec_type(arr);
  auto c1       = First::m256_shuffle_epi8(m_actual, m_mask);
  auto c2       = Second::m256_shuffle_epi8(actual, mask);
  EXPECT_EQ(memcmp(&c1, &c2, sizeof(c1)), 0);
}

TYPED_TEST(SimdFixture, testM256TestzSi256)
{
  using First  = typename SimdFixture<gtest_TypeParam_>::FirstNamespace;
  using Second = typename SimdFixture<gtest_TypeParam_>::SecondNamespace;

  auto c1 = First::m256_testz_si256(this->af, this->bf);
  auto c2 = Second::m256_testz_si256(this->as, this->bs);
  EXPECT_EQ(c1, c2);
}

TYPED_TEST(SimdFixture, testM256Permute2Regs)
{
  using First         = typename SimdFixture<gtest_TypeParam_>::FirstNamespace;
  using Second        = typename SimdFixture<gtest_TypeParam_>::SecondNamespace;
  using FirstVecType  = typename First::VecType;
  using SecondVecType = typename Second::VecType;

  // This tests that the permutations of some vectors does the same thing with both
  // our version and the FastHadamardLSH version. Note that this will only hold exactly for one
  // iteration: the randomness will be different, since we can't use AES to randomise on all
  // machines.

  int16_t arr[16];
  for (unsigned i = 0; i < 16; i++)
  {
    arr[i] = rand() % 201 - 100;
  }

  // Setup a random key
  auto m_key = First::build_small_vec_type(arr);
  auto key   = Second::build_small_vec_type(arr);
  for (unsigned i = 0; i < 16; i++)
  {
    arr[i] = rand() % 201 - 100;
  }

  // And a random PRG state
  auto m_prg_state = First::build_small_vec_type(arr);
  auto prg_state   = Second::build_small_vec_type(arr);

  auto m_extra_state = m_prg_state;
  auto extra_state   = prg_state;

  // Now we'll create lots of vectors that we're going shuffle
  std::array<FirstVecType, 100> m_vecs;
  std::array<SecondVecType, 100> vecs;

  for (unsigned i = 0; i < 100; i++)
  {
    for (unsigned j = 0; j < 16; j++)
    {
      arr[j] = rand() % 201 - 100;
    }
    m_vecs[i] = First::build_vec_type(arr);
    vecs[i]   = Second::build_vec_type(arr);
  }

  // Make sure our randomness didn't mess anything up
  ASSERT_EQ(memcmp(&vecs[0], &m_vecs[0], sizeof(vecs)), 0);

  // And now we'll hash them. The exact tail mask doesn't matter.
  // Now we'll call the hashing routines
  // The exact tail mask doesn't matter
  First::template m256_permute_epi16<2>(&m_vecs[0], m_prg_state, First::tailmasks[0], m_key,
                                        &m_extra_state);
  Second::template m256_permute_epi16<2>(&vecs[0], prg_state, Second::tailmasks[0], key,
                                         &extra_state);

  // And finally, compare that the vectors are the same.
  EXPECT_EQ(memcmp(&vecs[0], &m_vecs[0], sizeof(vecs)), 0);
}

TYPED_TEST(SimdFixture, testM256PermuteNRegs)
{
  // Same as above, but for many registers.
  // Thankfully FHT only requires 2 =< n < 8
  // And we can write the rest by hand
  using First         = typename SimdFixture<gtest_TypeParam_>::FirstNamespace;
  using Second        = typename SimdFixture<gtest_TypeParam_>::SecondNamespace;
  using FirstVecType  = typename First::VecType;
  using SecondVecType = typename Second::VecType;

#define testM256PermuteRegs(n)                                                                     \
  {                                                                                                \
    int16_t arr[16];                                                                               \
    for (unsigned i = 0; i < 16; i++)                                                              \
    {                                                                                              \
      arr[i] = rand() % 201 - 100;                                                                 \
    }                                                                                              \
    auto m_key = First::build_small_vec_type(arr);                                                 \
    auto key   = Second::build_small_vec_type(arr);                                                \
    for (unsigned i = 0; i < 16; i++)                                                              \
    {                                                                                              \
      arr[i] = rand() % 201 - 100;                                                                 \
    }                                                                                              \
    auto m_prg_state   = First::build_small_vec_type(arr);                                         \
    auto prg_state     = Second::build_small_vec_type(arr);                                        \
    auto m_extra_state = m_prg_state;                                                              \
    auto extra_state   = prg_state;                                                                \
    std::array<FirstVecType, 100> m_vecs;                                                          \
    std::array<SecondVecType, 100> vecs;                                                           \
    for (unsigned i = 0; i < 100; i++)                                                             \
    {                                                                                              \
      for (unsigned j = 0; j < 16; j++)                                                            \
      {                                                                                            \
        arr[j] = rand() % 201 - 100;                                                               \
      }                                                                                            \
      m_vecs[i] = First::build_vec_type(arr);                                                      \
      vecs[i]   = Second::build_vec_type(arr);                                                     \
    }                                                                                              \
    ASSERT_EQ(memcmp(&vecs[0], &m_vecs[0], sizeof(vecs)), 0);                                      \
    First::template m256_permute_epi16<2>(&m_vecs[0], m_prg_state, First::tailmasks[0], m_key,     \
                                          &m_extra_state);                                         \
    Second::template m256_permute_epi16<2>(&vecs[0], prg_state, Second::tailmasks[0], key,         \
                                           &extra_state);                                          \
    EXPECT_EQ(memcmp(&vecs[0], &m_vecs[0], sizeof(vecs)), 0);                                      \
  }

  testM256PermuteRegs(3) testM256PermuteRegs(4) testM256PermuteRegs(5) testM256PermuteRegs(6)
      testM256PermuteRegs(7)
}
