#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <utility>


bool JudgeM32(const std::string& line) {
  if (line.size() != 64) {
    throw std::runtime_error("invalid format");
  }

  for (int i = 0; i < 64; i++) {
    if ((i & 4) == 0) {
      if (line[i] == '*' || line[i+4] == '*') continue;
      if (line[i] != line[i+4]) return false;
    }
  }
  return true;
}

bool JudgeM31(const std::string& line) {
  if (line.size() != 64) {
    throw std::runtime_error("invalid format");
  }

  for (int i = 0; i < 64; i++) {
    if ((i & 4) == 0 && (i & 2) == 0) {
      char c = '*';
      c = (line[i] == '*') ? c : line[i];
      c = (line[i+2] == '*') ? c : line[i];
      c = (line[i+4] == '*') ? c : line[i];
      c = (line[i+6] == '*') ? c : line[i];
      if (c == '*') continue;
      else {
        char nc = (c == 'c') ? 'd' : 'c';
        if (nc == line[i] || nc == line[i+2] || nc == line[i+4] || nc == line[i+6]) return false;
      }
    }
  }
  return true;
}

bool JudgeM23(const std::string& line) {
  if (line.size() != 64) {
    throw std::runtime_error("invalid format");
  }

  for (int i = 0; i < 32; i++) {
    if (line[i] == '*' || line[i+32] == '*') continue;
    if (line[i] != line[i+32]) return false;
  }
  return true;
}

bool JudgeM13(const std::string& line) {
  if (line.size() != 64) {
    throw std::runtime_error("invalid format");
  }

  for (int i = 0; i < 16; i++) {
    char c = '*';
    c = (line[i] == '*') ? c : line[i];
    c = (line[i+16] == '*') ? c : line[i];
    c = (line[i+32] == '*') ? c : line[i];
    c = (line[i+48] == '*') ? c : line[i];
    if (c == '*') continue;
    else {
      char nc = (c == 'c') ? 'd' : 'c';
      if (nc == line[i] || nc == line[i+16] || nc == line[i+32] || nc == line[i+48]) return false;
    }
  }
  return true;
}

std::pair<int,int> MemoryLengths(const std::string& line) {
  bool m23 = JudgeM23(line);
  bool m32 = JudgeM32(line);
  bool m13 = m23 && JudgeM13(line);
  bool m31 = m32 && JudgeM31(line);
  int m1 = 3;
  if (m13) { m1 = 1; }
  else if (m23) { m1 = 2; }
  int m2 = 3;
  if (m31) { m2 = 1; }
  else if (m32) { m2 = 2; }
  return std::make_pair(m1, m2);
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
      auto m = MemoryLengths(l);
      if (m.first < 3 || m.second < 3) {
        auto a = ExpandStrategies(l);
        ans.insert(ans.end(), a.begin(), a.end());
      }
      l[i] = 'd';
      m = MemoryLengths(l);
      if (m.first < 3 || m.second < 3) {
        auto a = ExpandStrategies(l);
        ans.insert(ans.end(), a.begin(), a.end());
      }
      return ans;
    }
  }
  return {line};
}

int main(int argc, char* argv[]) {

  if (argc < 2) {
    std::cerr << "[Error] invalid number of arguments" << std::endl;
    std::cerr << "usage : " << argv[0] << " <input files> ..." << std::endl;
    throw std::runtime_error("invalid number of arguments");
  }

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
      if (count % 10'000'000 == 0) { std::cerr << "line :" << count << std::endl; }
      auto m = MemoryLengths(line);
      if (m.first < 3 || m.second < 3) {
        const auto expanded = ExpandStrategies(line);
        for (const std::string& s: expanded) {
          const auto mem = MemoryLengths(s);
          if (mem.first < 3 || mem.second < 3) {
            std::cout << mem.first << mem.second << ' ' << s << std::endl;
          }
        }
      }
      count++;
    }
  }

  return 0;

}