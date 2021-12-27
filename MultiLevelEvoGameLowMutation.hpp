//
// Created by Yohsuke Murase on 2021/10/29.
//

#ifndef CPP_MULTILEVELEVOGAME_LOW_MUTATION_HPP
#define CPP_MULTILEVELEVOGAME_LOW_MUTATION_HPP

#define EIGEN_DONT_PARALLELIZE

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


class MultiLevelEvoGameLowMutation {
  public:
  class Parameters {
    public:
    Parameters() = default;
    size_t T_max;
    size_t T_print;  // output interval
    size_t T_init;   // initial period
    size_t M, N;
    double benefit;
    double error_rate;
    double sigma, sigma_g;
    std::array<size_t,2> strategy_space;
    std::string initial_condition; // "random", "TFT", "WSLS", "TFT-ATFT", "CAPRI"
    uint64_t _seed;

    NLOHMANN_DEFINE_TYPE_INTRUSIVE(Parameters, T_max, T_print, T_init,
      M, N, benefit, error_rate, sigma, sigma_g,
      strategy_space, initial_condition, _seed);
  };


  class Species {
    public:
    explicit Species(uint64_t _strategy_id, double e) :
    strategy_id(_strategy_id), mem_lengths{0,0}, automaton_sizes{0,0} {
      StrategyM3 strategy(strategy_id);
      cooperation_level = strategy.CooperationLevel(e);
      is_efficient = strategy.IsEfficientTopo();
      is_defensible = strategy.IsDefensible();
      mem_lengths = StrategySpace::MemLengths(_strategy_id);
      automaton_sizes[0] = strategy.MinimizeDFA(false).to_map().size();
      automaton_sizes[1] = strategy.MinimizeDFA(true).to_map().size();
    };
    uint64_t strategy_id;
    double cooperation_level;
    bool is_efficient;
    bool is_defensible;
    StrategySpace::mem_t mem_lengths;
    std::array<size_t,2> automaton_sizes;

    NLOHMANN_DEFINE_TYPE_INTRUSIVE(Species, strategy_id, cooperation_level, is_efficient, is_defensible, mem_lengths, automaton_sizes);
  };

  explicit MultiLevelEvoGameLowMutation(Parameters _prm) :
    prm(std::move(_prm)), space(prm.strategy_space[0], prm.strategy_space[1]),
    current_species(0ull, 1.0e-4), rnd(prm._seed) {

    if (prm.initial_condition == "random") {
      uint64_t id = WeightedSampleStrategySpace();
      current_species = Species(id, prm.error_rate);
    }
    else {
      uint64_t str_id;
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
      current_species = Species(str_id, prm.error_rate);
    }
  };
  Parameters prm;
  StrategySpace space;
  Species current_species;
  std::mt19937_64 rnd;
  std::uniform_real_distribution<double> uni;

  // double FixationProb(uint64_t mutant_id, uint64_t resident_id, double mutant_coop_level, double resident_coop_level) const {
  double FixationProb(const Species& mutant, const Species& resident) const {
    // [TODO] implement me
    return 0.0;
  }

  double MigrationProb(const Species& s_target, const Species& s_focal) const {
    double pi = (prm.benefit - 1.0) * s_focal.cooperation_level;
    double p_target = (prm.benefit - 1.0) * s_target.cooperation_level;
    // f_{A\to B} = { 1 + \exp[ \sigma_g (s_A - s_B) ] }^{-1}
    return 1.0 / (1.0 + std::exp( prm.sigma_g * (pi - p_target) ));
  }

  uint64_t UniformSampleStrategySpace() {
    std::uniform_int_distribution<uint64_t> sample(0ull, space.Max());
    return space.ToGlobalID( sample(rnd) );
  }

  uint64_t WeightedSampleStrategySpace() {
    size_t num_spaces = (space.mem[0]+1) * (space.mem[1]+1);
    auto mi = static_cast<size_t>( uni(rnd) * (double)num_spaces );
    size_t m1 = mi % (space.mem[0]+1), m2 = mi / (space.mem[0]+1);
    StrategySpace ss(m1, m2);
    std::uniform_int_distribution<uint64_t> sample(0ull, ss.Max());
    uint64_t gid = ss.ToGlobalID( sample(rnd));
    const StrategySpace::mem_t target({m1, m2});
    while (StrategySpace::MemLengths(gid) != target) {
      gid = ss.ToGlobalID( sample(rnd));
    }
    return gid;
  }

  void Update() {
    uint64_t mut_id = WeightedSampleStrategySpace();
    Species mut(mut_id, prm.error_rate);
    double prob = FixationProb(mut, current_species);
    if (uni(rnd) < prob) {
      current_species = mut;
    }
  }
};


#endif //CPP_MULTILEVELEVOGAME_LOW_MUTATION_HPP
