#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <utility>
#include "StrategySpace.hpp"


uint64_t NumStrategies(const std::string& line) {
  if (line.size() != 64) {
    throw std::runtime_error("invalid format");
  }
  uint64_t ans = 1ull;
  for (int i = 0; i < 64; i++) {
    if (line[i] == '*') ans = ans << 1ull;
  }
  return ans;
}

bool ContainsShorterStrategies(const std::string& line) {
  if (line.size() != 64) {
    throw std::runtime_error("invalid format");
  }

  int m2 = 2;
  for (int i = 0; i < 64; i++) {
    char c1 = line[i], c2 = line[i^0b000100];
    if ((c1 == 'c' && c2 == 'd') || (c1 == 'd' && c2 == 'c')) m2 = 3;
  }

  int m1 = 2;
  for (int i = 0; i < 64; i++) {
    char c1 = line[i], c2 = line[i^0b100000];
    if ((c1 == 'c' && c2 == 'd') || (c1 == 'd' && c2 == 'c')) m1 = 3;
  }

  return (m1 == 2 || m2 == 2);
}

std::vector<std::string> ExpandStrategies(const std::string& line) {
  if (line.size() != 64) {
    throw std::runtime_error("invalid format");
  }

  for (int i = 0; i < 64; i++) {
    if (line[i] == '*') {
      std::vector<std::string> ans;
      std::string l = line;
      l[i] = 'c';
      if (ContainsShorterStrategies(l)) {
        auto a = ExpandStrategies(l);
        ans.insert(ans.end(), a.begin(), a.end());
      }
      l[i] = 'd';
      if (ContainsShorterStrategies(l)) {
        auto a = ExpandStrategies(l);
        ans.insert(ans.end(), a.begin(), a.end());
      }
      return ans;
    }
  }

  if (ContainsShorterStrategies(line)) { return {line}; }
  else return {};
}

uint64_t ToUint64(const std::string& s) {
  uint64_t ans = 0ull;
  for (size_t i = 0ul; i < 64ul; i++) {
    if (s[i] == 'd') {
      ans += (1ull << i);
    }
  }
  return ans;
}

int main(int argc, char* argv[]) {

  if (argc < 2) {
    std::cerr << "[Error] invalid number of arguments" << std::endl;
    std::cerr << "usage : " << argv[0] << " <input files> ..." << std::endl;
    throw std::runtime_error("invalid number of arguments");
  }

  std::vector< std::vector<size_t> > histo(4, std::vector<size_t>(4, 0));
  uint64_t total_num_strategies = 0ull;

  for (int i = 1; i < argc; i++) {
    std::ifstream fin(argv[i]);
    if (!fin) {
      std::cerr << "failed to open " << argv[i] << std::endl;
    }
    else {
      std::cerr << "reading: " << argv[i] << std::endl;
    }

    std::string line;
    long count = 0;
    while (fin >> line) {
      total_num_strategies += NumStrategies(line);
      if (count % 10'000'000 == 0) { std::cerr << "line :" << count << std::endl; }
      if (ContainsShorterStrategies(line) ) {
        const auto expanded = ExpandStrategies(line);
        for (const std::string& s: expanded) {
          uint64_t sid = ToUint64(s);
          const auto mem = StrategySpace::MemLengths(sid);
          if (mem[0] < 3 || mem[1] < 3) {
            std::cout << mem[0] << mem[1] << ' ' << s << std::endl;
            histo[mem[0]][mem[1]]++;
          }
        }
      }
      count++;
    }
  }

  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      if (i == 3 && j == 3) { continue; }
      std::cerr << i << ' ' << j << ' ' << histo[i][j] << std::endl;
      total_num_strategies -= histo[i][j];
    }
  }
  std::cerr << 3 << ' ' << 3 << ' ' << total_num_strategies << std::endl;

  return 0;

}