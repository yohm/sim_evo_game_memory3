//
// Created by Yohsuke Murase on 2021/10/25.
//

#include <iostream>
#include <chrono>
#include <random>
#include "omp.h"
#include "StrategySpace.hpp"
#include "StrategyM3.hpp"
#include "icecream-cpp/icecream.hpp"


std::string prev_key;
std::chrono::system_clock::time_point start;
void MeasureElapsed(const std::string& key) {
  std::chrono::system_clock::time_point end = std::chrono::system_clock::now();
  if (!prev_key.empty()) {
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();
    std::cerr << "T " << prev_key << " finished in " << elapsed << " ms" << std::endl;
  }
  start = end;
  prev_key = key;
}

std::array<uint64_t,4> MCSample(int m1, int m2, uint64_t num_samples = 100'000, uint64_t seed = 12345689ull) {
  // Monte Carlo sampling
  uint64_t num_all = 0, num_eff = 0, num_rival = 0, num_fr = 0;
  std::mt19937_64 rnd(seed);
  StrategySpace space(m1, m2);
  std::uniform_int_distribution<uint64_t> uni(0, space.Max());

  while (num_all < num_samples) {
    uint64_t lid = uni(rnd);
    uint64_t gid = space.ToGlobalID(lid);
    auto m = StrategySpace::MemLengths(gid);
    if (m[0] != m1 || m[1] != m2) continue;
    StrategyM3 s(gid);
    bool is_def = s.IsDefensible();
    bool is_eff = s.IsEfficientTopo();
    bool is_fr = is_def && is_eff;
    if (is_def) num_rival++;
    if (is_eff) num_eff++;
    if (is_fr) num_fr++;
    num_all++;
  }

  return {num_all, num_eff, num_rival, num_fr};
}


std::pair<double,double> AvgErr(const std::vector<double>& v) {
  double sum = 0.0;
  for (auto x : v) sum += x;
  double avg = sum / v.size();
  double err = 0.0;
  for (auto x : v) err += (x - avg) * (x - avg);
  err = std::sqrt(err / (v.size() * (v.size() - 1)));
  return {avg, err};
}


int main(int argc, char *argv[]) {

  Eigen::initParallel();
  if( argc != 3 ) {
    std::cerr << "Error : invalid argument" << std::endl;
    std::cerr << "  Usage : " << argv[0] << " <m1> <m2>" << std::endl;
    return 1;
  }

  const int m1 = std::stoi(argv[1]);
  const int m2 = std::stoi(argv[2]);

  if (m1 > 3 || m2 > 3) {
    std::cerr << "Error : m1 and m2 must be less than 4" << std::endl;
    return 1;
  }

  MeasureElapsed("initialize");

  if (m1 + m2 > 3) {
    const int n = 10, num_samples = 100'000;
    std::vector<double> v_num_eff, v_num_rival, v_num_fr;
    #pragma omp parallel for shared(v_num_eff, v_num_rival, v_num_fr)
    for (int i = 0; i < n; i++) {
      auto ans = MCSample(m1, m2, num_samples, 123456789ull + i);
      IC(ans);
      assert(ans[0] == num_samples);
      #pragma omp critical
      {
        v_num_eff.push_back((double)ans[1]);
        v_num_rival.push_back((double)ans[2]);
        v_num_fr.push_back((double)ans[3]);
      };
    }
    auto eff = AvgErr(v_num_eff);
    auto rival = AvgErr(v_num_rival);
    auto fr = AvgErr(v_num_fr);
    IC(eff, rival, fr);
    double frac_eff = eff.first / num_samples;
    double frac_rival = rival.first / num_samples;
    double frac_fr = fr.first / num_samples;
    IC(frac_eff*100, frac_rival*100, frac_fr*100);
  }
  else {
    // Exhaustive search
    StrategySpace space(m1, m2);
    uint64_t num_all = 0, num_eff = 0, num_rival = 0, num_fr = 0;
    for (uint64_t i = 0; i <= space.Max(); i++) {
      uint64_t gid = space.ToGlobalID(i);
      auto m = StrategySpace::MemLengths(gid);
      if (m[0] != m1 || m[1] != m2) continue;
      StrategyM3 s(gid);
      bool is_def = s.IsDefensible();
      bool is_eff = s.IsEfficientTopo();
      bool is_fr = is_def && is_eff;
      if (is_def) num_rival++;
      if (is_eff) num_eff++;
      if (is_fr) num_fr++;
      num_all++;
    }
    double frac_eff = (double)num_eff / num_all;
    double frac_rival = (double)num_rival / num_all;
    double frac_fr = (double)num_fr / num_all;
    IC(num_all, num_eff, num_rival, num_fr, frac_eff*100.0, frac_rival*100.0, frac_fr*100.0);
  }

  MeasureElapsed("done");
  return 0;
}