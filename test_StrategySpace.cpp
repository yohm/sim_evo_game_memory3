#include <iostream>
#include <cassert>
#include "StrategySpace.hpp"
#include "StrategyM3.hpp"

#define myassert(x) do {                              \
if (!(x)) {                                           \
  printf("Assertion failed: %s, file %s, line %d\n"   \
         , #x, __FILE__, __LINE__);                   \
  exit(1);                                            \
  }                                                   \
} while (0)


int main(int argc, char* argv[]) {
  using mem_t = StrategySpace::mem_t;
  const uint64_t ALLC = 0ul, ALLD = 0xFFFF'FFFF'FFFF'FFFF;
  const uint64_t TFT = 0b1010101010101010101010101010101010101010101010101010101010101010ull;
  const uint64_t ATFT = ~TFT;
  const uint64_t WSLS = 0b0101010110101010010101011010101001010101101010100101010110101010ull;
  const uint64_t TFT_ATFT = 0b1001100100100010100110011010101010011001001000101001100110101010ull;

  {
    StrategySpace m0(0,0);
    myassert(m0.Size() == 2ul);

    myassert(m0.ToLocalID(ALLC) == 0ul);
    myassert(m0.ToLocalID(ALLD) == 1ul);

    myassert(m0.ToGlobalID(0) == ALLC);
    myassert(m0.ToGlobalID(1) == ALLD);

    // ALLC, AllD are memory 0
    myassert(StrategySpace::MemLengths(ALLC) == mem_t({0ul, 0ul}));
    myassert(StrategySpace::MemLengths(ALLD) == mem_t({0ul, 0ul}));
  }

  {
    StrategySpace reactive(0, 1);

    myassert(reactive.Size() == 4ul);

    myassert(reactive.ToLocalID(ALLC) == 0ul);
    myassert(reactive.ToLocalID(ATFT) == 1ul);
    myassert(reactive.ToLocalID(TFT)  == 2ul);
    myassert(reactive.ToLocalID(ALLD) == 3ul);

    myassert(reactive.ToGlobalID(0) == ALLC);
    myassert(reactive.ToGlobalID(1) == ATFT);
    myassert(reactive.ToGlobalID(2) == TFT);
    myassert(reactive.ToGlobalID(3) == ALLD);

    myassert(StrategySpace::MemLengths(TFT) == mem_t({0ul, 1ul}));
    myassert(StrategySpace::MemLengths(ATFT) == mem_t({0ul, 1ul}));
  }

  {
    for (int m0 = 0; m0 <= 3; m0++) {
      for (int m1 = 0; m1 <= 3; m1++) {
        if (m0 + m1 >= 5) continue;
        StrategySpace m(m0, m1);
        myassert(m.Size() == (1ul << (1ul << (m0+m1))) );
        myassert(m.Max() == m.Size() - 1);
        for (uint64_t i = 0; i < m.Size(); i++) {
          uint64_t gid = m.ToGlobalID(i);
          auto ml = StrategySpace::MemLengths(gid);
          myassert(ml[0] <= m0 && ml[1] <= m1);
        }
      }
    }
  }

  {
    StrategySpace mem1(1, 1);
    myassert(mem1.Size() == 16ul);
    myassert(mem1.Max() == 15ul);
    myassert(StrategySpace::MemLengths(WSLS) == mem_t({1ul, 1ul}));
  }

  {
    StrategySpace m22(2, 2);

    myassert(m22.Size() == 65536ul);
    myassert(m22.Max() == 65535ul);
    myassert(StrategySpace::MemLengths(TFT_ATFT) == mem_t({2ul, 2ul}));
  }

  {
    StrategySpace m33(3, 3);
    myassert(m33.Max() == 18446744073709551615ull);
  }

  return 0;
}