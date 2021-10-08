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

  {
    StrategySpace m0(0,0);
    myassert(m0.IDMax() == 1ul);

    myassert(m0.ToLocalID(0) == 0ul);
    // myassert(m0.ToLocalID(0xFFFF'FFFF'FFFF'FFFF) == 1ul);

    myassert(m0.ToMem3ID(0) == 0ul);
    // myassert(m0.ToMem3ID(1) == 0xFFFF'FFFF'FFFF'FFFFul);
  }

  {
    StrategySpace reactive(0, 1);
    myassert(reactive.IDMax() == 3ul);
  }

  {
    StrategySpace mem1(1, 1);
    myassert(mem1.IDMax() == 15ul);
  }

  {
    StrategySpace mem1(2, 2);
    myassert(mem1.IDMax() == 65535ul);
  }

  {
    StrategySpace mem1(1, 3);
    myassert(mem1.IDMax() == 65535ul);
  }

  {
    StrategySpace mem3(3, 3);
    myassert(mem3.IDMax() == 0xFFFF'FFFF'FFFF'FFFF);
  }
  return 0;
}