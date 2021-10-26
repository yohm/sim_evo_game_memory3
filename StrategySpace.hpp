#include <iostream>
#include <array>
#include <bitset>
#include <cstdint>

#ifndef STRATEGY_SPACE_HPP
#define STRATEGY_SPACE_HPP

class StrategySpace {
public:
  using mem_t = std::array<size_t,2>;
  StrategySpace(size_t mem_self, size_t mem_cop) : mem({mem_self, mem_cop}) {
    if (mem[0] > 3 || mem[1] > 3) { throw std::runtime_error("unsupported memory length"); }
  };
  const mem_t mem;
  uint64_t Size() const {
    if (mem[0] == 3 && mem[1] == 3) { throw std::runtime_error("unsupported memory length"); }
    uint64_t ent = 1ull << (mem[0] + mem[1]);
    return 1ull << ent;
  }
  uint64_t Max() const {
    uint64_t ent = 1ull << (mem[0] + mem[1]);
    uint64_t ans = 0ul;
    for (size_t i = 0; i < ent; i++) {
      ans <<= 1ul;
      ans += 1ul;
    }
    return ans;
  }
  uint64_t ToGlobalID(uint64_t local_id) const {
    if (local_id > Max()) { throw std::runtime_error("invalid ID"); }

    std::bitset<64> gid(0ull), lid(local_id);
    uint64_t upper_mask = ((1ul<<mem[0])-1) << 3ul, lower_mask = (1ul<<mem[1])-1;
    for (u_int64_t i = 0; i < 64; i++) {
      int j = ((i&upper_mask) >> (3ul-mem[1])) | (i&lower_mask);  // index at local space
      if (lid[j]) gid.set(i, true);
    }
    return gid.to_ullong();
  }
  uint64_t ToLocalID(uint64_t mem3_id) const {
    auto _m = StrategySpace::MemLengths(mem3_id);
    if (_m[0] > mem[0] || _m[1] > mem[1]) {
      std::cerr << "[Error] Strategy " << mem3_id << "does not exist in the strategy space" << std::endl;
      throw std::runtime_error("invalid strategy ID");
    }

    std::bitset<64> gid(mem3_id), lid(0ull);
    uint64_t upper_mask = ((1ul<<mem[0])-1) << 3ul, lower_mask = (1ul<<mem[1])-1;
    // upper_mask: 0b111000 if m[0]==3, 0b011000 if m[0]==2, 0b001000 if m[0]==1, 0b000000 if m[0] == 0
    // lower_mask: 0b000111 if m[1]==3, 0b000011 if m[1]==2, 0b000001 if m[1]==1, 0b000000 if m[0] == 0
    for (u_int64_t i = 0; i < 64; i++) {
      int j = ((i&upper_mask) >> (3ul-mem[1])) | (i&lower_mask);  // index at local space
      if (gid[i]) lid.set(j, true);
    }
    return lid.to_ullong();
  }

  static mem_t MemLengths(uint64_t mem3_id) {
    mem_t mem = {3, 3};
    if (IsM2Self(mem3_id)) {
      if (IsM1Self(mem3_id)) {
        if (IsM0Self(mem3_id)) {
          mem[0] = 0;
        }
        else {
          mem[0] = 1;
        }
      }
      else {
        mem[0] = 2;
      }
    }
    if (IsM2Coplayer(mem3_id)) {
      if (IsM1Coplayer(mem3_id)) {
        if (IsM0Coplayer(mem3_id)) {
          mem[1] = 0;
        }
        else {
          mem[1] = 1;
        }
      }
      else {
        mem[1] = 2;
      }
    }
    return mem;
  }
private:
  static bool IsM2Self(uint64_t mem3_id) {
    std::bitset<64> b(mem3_id);
    for (size_t i = 0; i < 32; i++) {
      if (b[i] != b[i+32]) return false;
    }
    return true;
  }
  static bool IsM1Self(uint64_t mem3_id) {
    std::bitset<64> b(mem3_id);
    for (size_t i = 0; i < 16; i++) {
      if (b[i] != b[i+16] || b[i] != b[i+32] || b[i] != b[i+48]) return false;
    }
    return true;
  }
  static bool IsM0Self(uint64_t mem3_id) {
    std::bitset<64> b(mem3_id);
    for (size_t i = 0; i < 8; i++) {
      if (b[i] != b[i+8] || b[i] != b[i+16] || b[i] != b[i+24] ||
          b[i] != b[i+32] || b[i] != b[i+40] || b[i] != b[i+48] ) return false;
    }
    return true;
  }
  static bool IsM2Coplayer(uint64_t mem3_id) {
    std::bitset<64> b(mem3_id);
    for (size_t i = 0; i < 64; i++) {
      if ((i & 4ul) == 0ul) {
        if (b[i] != b[i+4]) return false;
      }
    }
    return true;
  }
  static bool IsM1Coplayer(uint64_t mem3_id) {
    std::bitset<64> b(mem3_id);
    for (size_t i = 0; i < 64; i++) {
      if ((i & 6ul) == 0) {
        if (b[i] != b[i+2] || b[i] != b[i+4] || b[i] != b[i+6]) return false;
      }
    }
    return true;
  }
  static bool IsM0Coplayer(uint64_t mem3_id) {
    std::bitset<64> b(mem3_id);
    for (size_t i = 0; i < 64; i++) {
      if ((i & 7ul) == 0) {
        if (b[i] != b[i+1] || b[i] != b[i+2] || b[i] != b[i+3] ||
            b[i] != b[i+4] || b[i] != b[i+5] || b[i] != b[i+6]) return false;
      }
    }
    return true;
  }
};

#endif