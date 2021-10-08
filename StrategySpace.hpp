#include <iostream>
#include <array>
#include <bitset>
#include <cstdint>

#ifndef STRATEGY_SPACE_HPP
#define STRATEGY_SPACE_HPP

class StrategySpace {
public:
  StrategySpace(size_t mem_self, size_t mem_cop) : mem({mem_self, mem_cop}) {
    if (mem[0] > 3 || mem[1] > 3) { throw std::runtime_error("unsupported memory length"); }
  };
  const std::array<size_t,2> mem;
  uint64_t IDMax() const {
    uint64_t ent = 1ull << (mem[0] + mem[1]);
    std::bitset<64> b = 0ull;
    for (int i = 0; i < ent; i++) { b.set(i, true); }
    return b.to_ullong();
  }
  uint64_t ToMem3ID(uint64_t local_id) const {
    if (local_id > IDMax()) {
      throw std::runtime_error("invalid ID");
    }
    return 0ul;
  }
  uint64_t ToLocalID(uint64_t mem3_id) const {
    return 0ul;
  }

  static std::pair<size_t,size_t> MemLength(uint64_t mem3_id) {
    // [IMPLEMENT ME]
    return {0, 0};
  }
};

#endif