//
// Created by Yohsuke Murase on 2021/10/29.
//

#ifndef CPP_MULTILEVELEVOGAME_HPP
#define CPP_MULTILEVELEVOGAME_HPP

#include <iostream>
#include <vector>
#include <array>
#include <map>
#include <random>
#include <cassert>
#include <fstream>
#include <chrono>
#include <regex>
#include <Eigen/Dense>
#include <nlohmann/json.hpp>
#include <omp.h>
#include "icecream-cpp/icecream.hpp"
#include "StrategyM3.hpp"
#include "StrategySpace.hpp"


class MultiLevelEvoGame {
  public:
  class Parameters {
    public:
    Parameters() { };
    size_t T_max;
    size_t T_print;  // output interval
    size_t T_init;   // initial period
    size_t M, N;
    double benefit;
    double error_rate;
    double sigma, sigma_g;
    double p_intra;
    std::array<size_t,2> strategy_space;
    std::string initial_condition; // "random", "TFT", "WSLS", "TFT-ATFT", "CAPRI"
    uint64_t _seed;

    NLOHMANN_DEFINE_TYPE_INTRUSIVE(Parameters, T_max, T_print, T_init,
      M, N, benefit, error_rate, sigma, sigma_g, p_intra, strategy_space,
      initial_condition, _seed);
  };


  class Species {
    public:
    explicit Species(uint64_t _strategy_id, double e) : strategy_id(_strategy_id) {
      StrategyM3 strategy(strategy_id);
      auto p = strategy.StationaryState(e);
      double c = 0.0;
      for (size_t n = 0; n < 64; n++) {
        StateM3 state(n);
        if (state.a_1 == C) { c += p[n]; }
      }
      cooperation_level = c;
      is_efficient = strategy.IsEfficientTopo();
      is_defensible = strategy.IsDefensible();
    };
    uint64_t strategy_id;
    double cooperation_level;
    double is_efficient;
    double is_defensible;

    NLOHMANN_DEFINE_TYPE_INTRUSIVE(Species, strategy_id, cooperation_level, is_efficient, is_defensible);
  };

  MultiLevelEvoGame(const Parameters& _prm) :
    prm(_prm), space(prm.strategy_space[0], prm.strategy_space[1]) {
    for (uint32_t t = 0; t < omp_get_max_threads(); t++) {
      std::seed_seq s = {static_cast<uint32_t>(prm._seed), t};
      a_rnd.emplace_back(s);
    }

    species.reserve(prm.M);
    if (prm.initial_condition == "random") {
      for (size_t i = 0; i < prm.M; i++) {
        uint64_t id = WeightedSampleStrategySpace();
        species.emplace_back(id, prm.error_rate);
      }
    }
    else {
      uint64_t str_id = 0;
      if (prm.initial_condition == "ALLC") str_id = StrategyM3::ALLC().ID();
      else if (prm.initial_condition == "ALLD") str_id = StrategyM3::ALLD().ID();
      else if (prm.initial_condition == "TFT") str_id = StrategyM3::TFT().ID();
      else if (prm.initial_condition == "WSLS") str_id = StrategyM3::WSLS().ID();
      else if (prm.initial_condition == "TFT-ATFT") str_id = StrategyM3::TFT_ATFT().ID();
      else if (prm.initial_condition == "CAPRI") str_id = StrategyM3::CAPRI().ID();
      else if (prm.initial_condition == "AON2") str_id = StrategyM3::AON(2).ID();
      else if (prm.initial_condition == "AON3") str_id = StrategyM3::AON(3).ID();
      else if (std::regex_match(prm.initial_condition, std::regex(R"(\d+)")) ){
        str_id = std::stoull(prm.initial_condition);
      }
      else { throw std::runtime_error("unknown initial condition"); }
      for (size_t i = 0; i < prm.M; i++) {
        species.emplace_back(str_id, prm.error_rate);
      }
    }
  };
  Parameters prm;
  StrategySpace space;
  std::vector<Species> species;
  std::vector<std::mt19937_64> a_rnd;
  std::uniform_real_distribution<double> uni;

  // payoff of species i and j when the game is played by (i,j)
  std::array<double,2> Payoffs(uint64_t strategy_i, uint64_t strategy_j) const {
    StrategyM3 si(strategy_i);
    StrategyM3 sj(strategy_j);
    auto p = si.StationaryState(prm.error_rate, &sj);
    double c_ij = 0.0, c_ji = 0.0;  // cooperation level from i to j and vice versa
    for (size_t n = 0; n < 64; n++) {
      StateM3 s(n);
      if (s.a_1 == C) {
        c_ij += p[n];
      }
      if (s.b_1 == C) {
        c_ji += p[n];
      }
    }
    double benefit = prm.benefit, cost = 1.0;
    return { c_ji * benefit - c_ij * cost, c_ij * benefit - c_ji * cost };
  }

  // double FixationProb(uint64_t mutant_id, uint64_t resident_id, double mutant_coop_level, double resident_coop_level) const {
  double FixationProb(const Species& mutant, const Species& resident) const {
    // \frac{1}{\rho} = \sum_{i=0}^{N-1} \exp\left( \sigma \sum_{j=1}^{i} \left[(N-j-1)s_{yy} + js_{yx} - (N-j)s_{xy} - (j-1)s_{xx} \right] \right) \\
    //                = \sum_{i=0}^{N-1} \exp\left( \frac{\sigma i}{2} \left[(-i+2N-3)s_{yy} + (i+1)s_{yx} - (-i+2N-1)s_{xy} - (i-1)s_{xx} \right] \right)
    double s_xx = (prm.benefit - 1.0) * mutant.cooperation_level;     // == Payoffs(mutant_id, mutant_id)[0];
    double s_yy = (prm.benefit - 1.0) * resident.cooperation_level;   // == Payoffs(resident_id, resident_id)[0];
    auto xy = Payoffs(mutant.strategy_id, resident.strategy_id);
    double s_xy = xy[0];
    double s_yx = xy[1];
    size_t N = prm.N;
    double sigma = prm.sigma;

    auto num_games = static_cast<double>(N-1);
    s_xx /= num_games;
    s_yy /= num_games;
    s_xy /= num_games;
    s_yx /= num_games;
    double rho_inv = 0.0;
    for (int i=0; i < N; i++) {
      double x = sigma * i * 0.5 * (
        (double)(2*N-3-i) * s_yy
        + (double)(i+1) * s_yx
        - (double)(2*N-1-i) * s_xy
        - (double)(i-1) * s_xx
      );
      rho_inv += std::exp(x);
    }
    return 1.0 / rho_inv;
  }

  double SelectionProb(const Species& s_i, const Species& s_j) const {
    double pi = (prm.benefit - 1.0) * s_i.cooperation_level;
    double pj = (prm.benefit - 1.0) * s_j.cooperation_level;
    // f_{A\to B} = { 1 + \exp[ \sigma_g (s_A - s_B) ] }^{-1}
    return 1.0 / (1.0 + std::exp( prm.sigma_g * (pi - pj) ));
  }

  uint64_t UniformSampleStrategySpace() {
    const int th = omp_get_thread_num();
    std::uniform_int_distribution<uint64_t> sample(0ull, space.Max());
    return space.ToGlobalID( sample(a_rnd[th]));
  }

  uint64_t WeightedSampleStrategySpace() {
    const int th = omp_get_thread_num();
    size_t num_spaces = (space.mem[0]+1) * (space.mem[1]+1);
    size_t mi = static_cast<size_t>( uni(a_rnd[th]) * num_spaces );
    size_t m1 = mi % (space.mem[0]+1), m2 = mi / (space.mem[0]+1);
    StrategySpace ss(m1, m2);
    std::uniform_int_distribution<uint64_t> sample(0ull, ss.Max());
    return ss.ToGlobalID( sample(a_rnd[th]));
  }

  void Update() {
    std::vector<Species> species_new = species;
    #pragma omp parallel for
    for (size_t i = 0; i < prm.M; i++) {
      int th = omp_get_thread_num();
      if (uni(a_rnd[th]) < prm.p_intra) {
        species_new[i] = IntraGroupSelection(i);
      }
      else {
        species_new[i] = InterGroupSelection(i);
      }
    }
    species = species_new;
  }

  Species IntraGroupSelection(size_t g) {
    const int th = omp_get_thread_num();
    uint64_t mut_id = WeightedSampleStrategySpace();
    Species mut(mut_id, prm.error_rate);
    // double mut_coop_level = CooperationLevel(mut_id);
    double f = FixationProb(mut, species[g]);
    IC(g, species[g], mut, f);
    if (uni(a_rnd[th]) < f) {
      return mut;
    }
    return species[g];
  }

  Species InterGroupSelection(size_t g1) {
    const int th = omp_get_thread_num();
    size_t g2 = static_cast<size_t>(g1 + 1 + uni(a_rnd[th]) * (prm.M-1)) % prm.M;
    double p = SelectionProb(species[g1], species[g2]);
    IC(species[g1], species[g2], p);
    if (uni(a_rnd[th]) < p) {
      return species[g2];
    }
    return species[g1];
  }

  double CooperationLevel() const {
    double ans = 0.0;
    for (const Species& s: species) {
      ans += s.cooperation_level;
    }
    return ans / species.size();
  }

  size_t NumEfficient() const {
    size_t count = 0;
    for (const Species& s: species) {
      if (s.is_efficient) count++;
    }
    return count;
  }

  size_t NumDefensible() const {
    size_t count = 0;
    for (const Species& s: species) {
      if (s.is_defensible) count++;
    }
    return count;
  }

  size_t NumFriendlyRival() const {
    size_t count = 0;
    for (const Species& s: species) {
      if (s.is_defensible && s.is_efficient) count++;
    }
    return count;
  }

  bool HasSpecies(uint64_t species_id) const {
    for (const Species& s: species) {
      if (s.strategy_id == species_id) return true;
    }
    return false;
  }

  double Diversity() const {
    // exponential Shannon entropy
    std::map<uint64_t,double> freq;
    for (const Species& s: species) {
      if (freq.find(s.strategy_id) == freq.end()) { freq[s.strategy_id] = 0.0; }
      freq[s.strategy_id] += 1.0;
    }
    double entropy = 0.0;
    for (auto pair: freq) {
      double p = pair.second / prm.M;
      entropy += -p * std::log(p);
    }
    return std::exp(entropy);
  }
};


#endif //CPP_MULTILEVELEVOGAME_HPP
