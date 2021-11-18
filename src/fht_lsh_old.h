#ifndef G6K_FHTLSH_H
#define G6K_FHTLSH_H

#include <immintrin.h>
#include <stdio.h>
#include <assert.h>
#include <cstring>
#include <algorithm>
#include <array>
#include <vector>
#include <math.h>
#include <iostream>

#include "simd.hpp"

const __m256i mixmask_threshold = _mm256_set_epi16(
    0x5555, 0x5555, 0x5555, 0x5555, 0x5555, 0x5555, 0x5555, 0x5555, 
    0xAAAA, 0xAAAA, 0xAAAA, 0xAAAA, 0xAAAA, 0xAAAA, 0xAAAA, 0xAAAA);

const __m256i _7FFF_epi16 = _mm256_set_epi16(
    0x7FFF, 0x7FFF, 0x7FFF, 0x7FFF, 0x7FFF, 0x7FFF, 0x7FFF, 0x7FFF, 
    0x7FFF, 0x7FFF, 0x7FFF, 0x7FFF, 0x7FFF, 0x7FFF, 0x7FFF, 0x7FFF);

const __m256i sign_mask_2 = _mm256_set_epi16(
    0xFFFF, 0x0001, 0xFFFF, 0x0001, 0xFFFF, 0x0001, 0xFFFF, 0x0001, 
    0xFFFF, 0x0001, 0xFFFF, 0x0001, 0xFFFF, 0x0001, 0xFFFF, 0x0001);

const __m256i mask_even_epi16 = _mm256_set_epi16(
    0xFFFF, 0x0000, 0xFFFF, 0x0000, 0xFFFF, 0x0000, 0xFFFF, 0x0000, 
    0xFFFF, 0x0000, 0xFFFF, 0x0000, 0xFFFF, 0x0000, 0xFFFF, 0x0000);

const __m256i mask_odd_epi16 = _mm256_set_epi16(
    0x0000, 0xFFFF, 0x0000, 0xFFFF, 0x0000, 0xFFFF, 0x0000, 0xFFFF, 
    0x0000, 0xFFFF, 0x0000, 0xFFFF, 0x0000, 0xFFFF, 0x0000, 0xFFFF);

const __m256i regroup_for_max = _mm256_set_epi8(
    0x0F, 0x0E, 0x07, 0x06, 0x0D, 0x0C, 0x05, 0x04, 
    0x0B, 0x0A, 0x03, 0x02, 0x09, 0x08, 0x01, 0x00,
    0x1F, 0x1E, 0x17, 0x16, 0x1D, 0x1C, 0x15, 0x14, 
    0x1B, 0x1A, 0x13, 0x12, 0x19, 0x18, 0x11, 0x10);

const __m256i sign_mask_8 = _mm256_set_epi16(
    0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF,
    0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001);

const __m256i sign_shuffle = _mm256_set_epi16(
    0xFFFF, 0xFFFF, 0xFFFF, 0x0001, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF,
    0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0xFFFF, 0xFFFF);

const __m256i indices_epi8 = _mm256_set_epi8(
    0x0F, 0x0E, 0x0D, 0x0C, 0x0B, 0x0A, 0x09, 0x08,
    0x07, 0x06, 0x05, 0x04, 0x03, 0x02, 0x01, 0x00,
    0x1F, 0x1E, 0x1D, 0x1C, 0x1B, 0x1A, 0x19, 0x18,
    0x17, 0x16, 0x15, 0x14, 0x13, 0x12, 0x11, 0x10);

const __m256i indices_epi16 = _mm256_set_epi16(
    0x000F, 0x000E, 0x000D, 0x000C, 0x000B, 0x000A, 0x0009, 0x0008,
    0x0007, 0x0006, 0x0005, 0x0004, 0x0003, 0x0002, 0x0001, 0x0000);

const __m256i indices_sa1_epi16 = _mm256_set_epi16(
    0x0010, 0x000F, 0x000E, 0x000D, 0x000C, 0x000B, 0x000A, 0x0009,
    0x0008, 0x0007, 0x0006, 0x0005, 0x0004, 0x0003, 0x0002, 0x0001);

const __m256i _0010_epi16 = _mm256_set_epi16(
    0x0010, 0x0010, 0x0010, 0x0010, 0x0010, 0x0010, 0x0010, 0x0010,
    0x0010, 0x0010, 0x0010, 0x0010, 0x0010, 0x0010, 0x0010, 0x0010);


const __m256i rnd_mult_epi32 = _mm256_set_epi32(
    0xF010A011, 0x70160011, 0x70162011, 0x00160411,
    0x0410F011, 0x02100011, 0xF0160011, 0x00107010);

// 0xFFFF = -1, 0x0001 = 1
const __m256i negation_masks_epi16[2] = {
    _mm256_set_epi16(
    0xFFFF, 0x0001, 0xFFFF, 0xFFFF, 0xFFFF, 0x0001, 0x0001, 0xFFFF,
    0xFFFF, 0x0001, 0xFFFF, 0x0001, 0xFFFF, 0xFFFF, 0x0001, 0xFFFF),
    _mm256_set_epi16(
    0xFFFF, 0x0001, 0x0001, 0xFFFF, 0xFFFF, 0x0001, 0x0001, 0xFFFF,
    0xFFFF, 0x0001, 0xFFFF, 0x0001, 0xFFFF, 0xFFFF, 0x0001, 0xFFFF)
    };

const __m256i permutations_epi16[4] = {
    _mm256_set_epi16(
    0x0F0E, 0x0706, 0x0100, 0x0908, 0x0B0A, 0x0D0C, 0x0504, 0x0302,
    0x0706, 0x0F0E, 0x0504, 0x0302, 0x0B0A, 0x0908, 0x0D0C, 0x0100),
    _mm256_set_epi16(
    0x0D0C, 0x0504, 0x0302, 0x0B0A, 0x0F0E, 0x0908, 0x0706, 0x0100, 
    0x0B0A, 0x0908, 0x0706, 0x0F0E, 0x0302, 0x0100, 0x0504, 0x0D0C),
    _mm256_set_epi16(
    0x0D0C, 0x0B0A, 0x0706, 0x0100, 0x0F0E, 0x0908, 0x0504, 0x0302, 
    0x0B0A, 0x0908, 0x0302, 0x0100, 0x0504, 0x0D0C, 0x0706, 0x0F0E),
    _mm256_set_epi16(
    0x0D0C, 0x0F0E, 0x0908, 0x0706, 0x0100, 0x0504, 0x0302, 0x0B0A,
    0x0302, 0x0100, 0x0504, 0x0B0A, 0x0908, 0x0706, 0x0F0E, 0x0D0C)
    };


const __m256i tailmasks[16] = {
    _mm256_set_epi16(0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF,
                     0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF),
    _mm256_set_epi16(0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
                     0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0xFFFF),
    _mm256_set_epi16(0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
                     0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0xFFFF, 0xFFFF),
    _mm256_set_epi16(0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
                     0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0xFFFF, 0xFFFF, 0xFFFF),
    _mm256_set_epi16(0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
                     0x0000, 0x0000, 0x0000, 0x0000, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF),
    _mm256_set_epi16(0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
                     0x0000, 0x0000, 0x0000, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF),
    _mm256_set_epi16(0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
                     0x0000, 0x0000, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF),
    _mm256_set_epi16(0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
                     0x0000, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF),
    _mm256_set_epi16(0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
                     0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF),
    _mm256_set_epi16(0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0xFFFF,
                     0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF),
    _mm256_set_epi16(0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0xFFFF, 0xFFFF,
                     0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF),
    _mm256_set_epi16(0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0xFFFF, 0xFFFF, 0xFFFF,
                     0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF),
    _mm256_set_epi16(0x0000, 0x0000, 0x0000, 0x0000, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF,
                     0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF),
    _mm256_set_epi16(0x0000, 0x0000, 0x0000, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF,
                     0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF),
    _mm256_set_epi16(0x0000, 0x0000, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF,
                     0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF),
    _mm256_set_epi16(0x0000, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF,
                     0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF)};


class FastHadamardLSH
    {
      // The full version of the bucketer has these as private: these are *just* public here for better
      // testing.
    public:
    // full_seed contains the source of randomness. We extract from this in our permutations and update
    // it to keep it fresh.
    Simd::SmallVecType full_seed;
    // aes_key contains the key we use for using the singular AES tour for updating our randomness.
    Simd::SmallVecType aes_key;


    /*
     * see fht_lsh.cpp for commments on these functions.
     */
    // NOTE: these are in this file for testing purposes.
        static inline void m256_hadamard16_epi16(const __m256i &x1,__m256i &r1);
        static inline void m256_hadamard32_epi16(const __m256i &x1,const __m256i &x2, __m256i &r1, __m256i &r2);

        template<int regs_> 
        static inline void m256_permute_epi16(__m256i * const v, __m128i &randomness,const __m256i &tailmask, const __m128i key, __m128i* extra_state = nullptr);
        static inline void m256_mix(__m256i &v0, __m256i &v1, const __m256i &mask);

        inline void insert_in_maxs(int32_t * const maxs, const int16_t val, const int32_t index);
      inline void insert_in_maxs_epi16(int32_t * const maxs, const int i_high, const Simd::VecType vals);

        template<int regs_> 
        void hash_templated(const int16_t * const vv, int32_t * const res);

    public: 

    // n is the adjusted hashing dimension that we use in this subcode.
        size_t n;

    // codesize is the size of this subcode
        size_t codesize;
    // multi_hash tells us how many subcode blocks we are hashing against.
        unsigned multi_hash;

        // regs is the number of registers we have available to use for hashing
    unsigned regs;
        int64_t seed;

    // Prints out an m256i vector as 16bit chunks
        void pprint(const __m256i& x)
        {
            int16_t* f = (int16_t*) &x;
            printf("%4i %4i %4i %4i %4i %4i %4i %4i %4i %4i %4i %4i %4i %4i %4i %4i\n",
                    f[  0], f[  1], f[  2], f[  3], f[  4], f[  5], f[  6], f[  7],
                    f[8+0], f[8+1], f[8+2], f[8+3], f[8+4], f[8+5], f[8+6], f[8+7]);
        }


        explicit FastHadamardLSH(const size_t _n,const  size_t _codesize, 
                                 const unsigned _multi_hash, const  int64_t _seed) : 
        n(_n), codesize(_codesize), multi_hash(_multi_hash), seed(_seed)
        { 
            // Generate some initial randomness - this is used for building the permutations later 
            aes_key   = Simd::m128_set_epi64x(0xDEADBEAF * _seed + 0xC00010FF, 0x00BAB10C * _seed + 0xBAADA555);
            full_seed = Simd::m128_set_epi64x(0xC0DED00D * _seed + 0xBAAAAAAD, 0x000FF1CE * _seed + 0xCAFED00D);

        // It only makes sense to hash to one or more buckets, so we do this check here
            if (multi_hash == 0) throw std::invalid_argument( "multi_hash should be >= 1" );

            if (n <= 16 || n > 128) throw std::invalid_argument( "lsh dimension invalid (must be 16 < n <= 128)");
            regs = (n+15)/16;
            if (regs < 2 || regs > 8) throw std::invalid_argument( "lsh dimension invalid (must be 16 < n <= 128)");
        };

    // This hashes the vector v against this subcode, producing an array of hash values (stored in hashes)
        void hash(const float * v, float * coeff, int32_t * hashes);

    };

class ProductLSH
    {

    public:
        // Permutation & sign are used to permute the input vector
            std::vector<size_t> permutation;
            std::vector<int> sign;

        // codesizes denotes how long each subcode is
            std::vector<size_t> codesizes;
            std::vector<int> ns, is;

            // This holds all of the different subcodes
        std::vector<FastHadamardLSH> lshs;
            
        // This function inserts the score into the correct position of the scores array
        // as well as the index into the right position of the indices array.
            inline bool insert_in_maxs(float * const scores, int32_t * const indices, const float score, const int32_t index);

        // These are used for the prg_state when building the permutation
        Simd::SmallVecType full_seed;
        Simd::SmallVecType aes_key;

        // This function is a specialisation of the hash function - the only difference here is that we handle differently
        // depending on the number of blocks we're hashing against 
            template<int blocks_> 
        void hash_templated(const float * vv, int32_t * res);

    public: 
        
        // n denotes the dimension of the lattice, blocks denotes the width of each hash block
        size_t n, blocks;
        // codesize denotes how long this code is 
        int64_t codesize;

        //multi_hash is how many buckets we're targeting
        //multi_hash_block is the size of each hash_block
        unsigned multi_hash;
        unsigned multi_hash_block;
        explicit ProductLSH(const size_t _n,const size_t _blocks, const  int64_t _codesize, 
                            const unsigned _multi_hash, const int64_t _seed) : 
        permutation(_n), sign(_n),
        codesizes(_blocks), ns(_blocks), is(_blocks), 
        n(_n), blocks(_blocks), codesize(_codesize), multi_hash(_multi_hash)
        {
        // Set up our permutation randomness    
            aes_key =  Simd::m128_set_epi64x(0xFACEFEED * _seed + 0xDEAD10CC, 0xFFBADD11 * _seed + 0xDEADBEEF);
            full_seed = Simd::m128_set_epi64x(0xD0D0FA11 * _seed + 0xD15EA5E5, 0xFEE1DEAD * _seed + 0xB105F00D);
            auto prg_state = full_seed;

	    // Taken is a vector denoting if we've used this position in our permutation before
	    std::vector<bool> taken(n,false);
        
	    // Build the permutation that's applied to each vector upon hashing
            for (size_t i = 0; i < n;)
            {
	      // produce a new prng state & take the first 64-bits as output
	      // We then use this to correspond to an array position - repeating if we fail
	      // to find an unused element
              prg_state  = Simd::m128_random_state(prg_state, aes_key, nullptr);
              size_t pos = Simd::m128_extract_epi64<0>(prg_state) & n;
              if (taken[pos]) continue;
	      // Note that we've used this element, and put it in the permutation array
              taken[pos] = true;
              permutation[i] = pos;
	      // We also take this chance to permute the signs too - if the second 64-bit number is odd then 
	      // we will negate in future. Then, just continue producing the permutation
              sign[pos] = (Simd::m128_extract_epi64<1>(prg_state) % 2) ? 1 : -1;
              ++i;
            }


        // rn is the number of remaining dimensions we have to divide up
        // Similarly with rblocks and rcodesize
            int rn = n;
            int rblocks = blocks;
            double rcodesize = codesize / (1 << (blocks - 1));
            multi_hash_block = multi_hash;
    
        // We take this opportunity to reserve some memory, so that we don't have to keep 
        // allocating more as we push back    
            lshs.reserve(blocks);

            for (size_t i = 0; i < blocks; ++i)
            {
        // Divide up the number of dimensions 
                ns[i] = rn / rblocks;
                is[i] = n - rn;
                codesizes[i] = int64_t(pow(rcodesize, 1./rblocks)+0.0001);
        // Check that we haven't given more work than necessary to a single sub-code
                assert(multi_hash <= codesizes[i]);
        // Put the new subcode into the lshs array at position i.
                lshs.emplace_back(FastHadamardLSH(ns[i], codesizes[i], multi_hash, _seed + uint64_t(i) * 1641633149));
        // Subtract the number of dimensions we've allocated to lsh[i] & the code size we've dealt with 
                rn -= ns[i];
                rcodesize /= codesizes[i];
                rblocks --;
            }
        codesize = 1;
        for (size_t i = 0; i < blocks; ++i)
        {
            codesize *= codesizes[i];
            if (i>0) codesize *= 2;
        }
        assert(codesize <= _codesize);
        };

    // Hash. Given a vector v as input, hash it against the subcodes and produce the results, stored in res. 
        void hash(const float* v, int32_t * res);
    };

// INLINE FUNCTION DEFS
/*
 * m256_hadamard16_epi16. This function applies the Hadamard transformation
 * over 16 entries of 16-bit integers stored in a single __m256i vector x1, 
 * storing result in r1.
 */

inline void FastHadamardLSH::m256_hadamard16_epi16(const __m256i &x1, __m256i &r1)
{
    /* Apply a permutation 0123 -> 1032 to 64 bit words. */ 
    __m256i a1 = _mm256_permute4x64_epi64(x1, 0b01001110);
        
    // From here we treat the input vector x1 as 16, 16-bit integers - which is what the entries are
    // Now negate the first 8 16-bit integers of x1
    __m256i t1 = _mm256_sign_epi16(x1, sign_mask_8);
    
    
    // Add the permutation to the recently negated portion & apply the second sign mask

    a1 = _mm256_add_epi16(a1, t1);

    __m256i b1 = _mm256_sign_epi16(a1, sign_mask_2);

    // With this, we can now build what we want by repeatedly applying the sign mask and adding.
    a1 = _mm256_hadd_epi16(a1, b1);
    b1 = _mm256_sign_epi16(a1, sign_mask_2);

    a1 = _mm256_hadd_epi16(a1, b1);
    b1 = _mm256_sign_epi16(a1, sign_mask_2);

    r1 = _mm256_hadd_epi16(a1, b1);
}


inline void FastHadamardLSH::m256_hadamard32_epi16(const __m256i &x1,const __m256i &x2, __m256i &r1, __m256i &r2)
{
    // Permute 64-bit chunks of a1 and a2 and then negate the first 8 16-bit integers of each 
    // x1 and x2.  
    __m256i a1 = _mm256_permute4x64_epi64(x1, 0b01001110);
    __m256i a2 = _mm256_permute4x64_epi64(x2, 0b01001110);


    __m256i t1 = _mm256_sign_epi16(x1, sign_mask_8);
    __m256i t2 = _mm256_sign_epi16(x2, sign_mask_8);
    
    // Add the results and negate the second 8 16-bit integers
    a1 = _mm256_add_epi16(a1, t1);
    a2 = _mm256_add_epi16(a2, t2);


    __m256i b1 = _mm256_sign_epi16(a1, sign_mask_2);
    __m256i b2 = _mm256_sign_epi16(a2, sign_mask_2);

    // Now apply the 16-bit Hadamard transforms and repeat the process
    a1 = _mm256_hadd_epi16(a1, b1);
    a2 = _mm256_hadd_epi16(a2, b2);
    b1 = _mm256_sign_epi16(a1, sign_mask_2);
    b2 = _mm256_sign_epi16(a2, sign_mask_2);
    a1 = _mm256_hadd_epi16(a1, b1);
    a2 = _mm256_hadd_epi16(a2, b2);
    b1 = _mm256_sign_epi16(a1, sign_mask_2);
    b2 = _mm256_sign_epi16(a2, sign_mask_2);
    a1 = _mm256_hadd_epi16(a1, b1);
    a2 = _mm256_hadd_epi16(a2, b2);

    r1 = _mm256_add_epi16(a1, a2);
    r2 = _mm256_sub_epi16(a1, a2);
}

/*
 * m256_mix. Swaps V0[i] and V1[i] iff mask[i] = 1 for 0 <= i < 255.
 */
inline void FastHadamardLSH::m256_mix(__m256i &v0, __m256i &v1, const __m256i &mask)
{
    __m256i diff;
    diff = _mm256_xor_si256(v0, v1);
    diff = _mm256_and_si256(diff, mask);
    v0 = _mm256_xor_si256(v0, diff);
    v1 = _mm256_xor_si256(v1, diff);
}

template<>
inline void FastHadamardLSH::m256_permute_epi16<2>(__m256i * const v, __m128i &prg_state, const __m256i &tailmask, __m128i key, __m128i* extra_state)
{
    // double pack the prg state in rnd (has impact of doubly repeating the prg state in rnd)
    // Though we will use different threshold on each part decorrelating the permutation
    // on each halves 
    
    __m256i rnd = _mm256_broadcastsi128_si256(prg_state);
    __m256i mask;

    // With only 2 registers, we may not have enough room to randomize via m256_mix, 
    // so we also choose at random among a few precomputed permutation to apply on
    // the first register

    uint32_t x = _mm_extract_epi64(prg_state, 0);
    uint32_t x1 = (x  >> 16) & 0x03;
    uint32_t x2 = x & 0x03;

    // Apply the precomputed permutations to the input vector
    v[0] = _mm256_shuffle_epi8(v[0], permutations_epi16[x1]);
    m256_mix(v[0], v[1], tailmask);
    v[0] = _mm256_permute4x64_epi64(v[0], 0b10010011);
    v[0] = _mm256_shuffle_epi8(v[0], permutations_epi16[x2]);

    mask = _mm256_cmpgt_epi16(rnd, mixmask_threshold);
    mask = _mm256_and_si256(mask, tailmask);
    m256_mix(v[0], v[1], mask);

    // update the very fast but non-cryptographic PRG (one tour of AES)
    prg_state = _mm_aesenc_si128(prg_state, key);    
}



/* Same as above, but for inputs of arbitrary lengths.  */

template<int regs_> 
inline void FastHadamardLSH::m256_permute_epi16(__m256i * const v, __m128i &prg_state, const __m256i &tailmask, __m128i key, __m128i* extra_state)
{
    // double pack the prg state in rnd (has impact of doubly repeating the prg state in rnd)
    // Though we will use different threshold on each part decorrelating the permutation
    // on each halves 

    __m256i rnd = _mm256_broadcastsi128_si256(prg_state);
    __m256i tmp;

    // We treat the even and the odd positions differently
    // This is for the goal of decorrelating the permutation on the 
    // double packed prng state.
    for (int i = 0; i < (regs_-1)/2; ++i)
    {
        // shuffle 8 bit parts in each 128 bit lane
	// Note - the exact semantics of what this function does are a bit confusing.
	// See the Intel intrinsics guide if you're curious
        v[2*i  ] = _mm256_shuffle_epi8(v[2*i], permutations_epi16[i % 3]);
	// For the odd positions we permute each 64-bit chunk according to the mask.
        v[2*i+1] = _mm256_permute4x64_epi64(v[2*i+1], 0b10010011);
    }

    // Now we negate the first two vectors according to the negation masks
    v[0] = _mm256_sign_epi16(v[0], negation_masks_epi16[0]);
    v[1] = _mm256_sign_epi16(v[1], negation_masks_epi16[1]);


    // swap int16 entries of v[0] and v[1] where rnd > threshold
    tmp = _mm256_cmpgt_epi16(rnd, mixmask_threshold);
    m256_mix(v[0], v[1], tmp);
    // Shift the randomness around before extracting more (sonmewhat independent) mixing bits
    rnd = _mm256_slli_epi16(rnd, 1);

    // Now do random swaps between v[0] and v[last-1]
    m256_mix(v[0], v[regs_- 2], tmp);
    rnd = _mm256_slli_epi16(rnd, 1);

    // Now do swaps between v[1] and v[last], avoiding padding data
    m256_mix(v[1], v[regs_ - 1], tailmask);


    // More permuting
    for (int i = 2; i + 2 < regs_; i+=2)
    {
        rnd = _mm256_slli_epi16(rnd, 1);
        tmp = _mm256_cmpgt_epi16(rnd, mixmask_threshold);
        m256_mix(v[0], v[i], tmp);
        rnd = _mm256_slli_epi16(rnd, 1);
        tmp = _mm256_cmpgt_epi16(rnd, mixmask_threshold);
        m256_mix(v[1], v[i+1], tmp);
    }

    // update the very fast but non-cryptographic PRG (one tour of AES)
    prg_state = _mm_aesenc_si128(prg_state, key);    
}



#endif
