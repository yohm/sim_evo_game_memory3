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

  double FixationProb(const Species& mutant, const Species& resident) const {
    //\Psi_{A} = 1/{1 + \exp[\sigma(\pi_{BA}-\pi_{AB})]}
    //         * {1-\exp[  \sigma(\pi_{BA}-\pi_{AB}) + \sigma_g( \pi_{BB}-\pi_{AA} ) ]}
    //         / {1-\exp[ M\sigma(\pi_{BA}-\pi_{AB}) +M\sigma_g( \pi_{BB}-\pi_{AA} ) ]}
    double pi_mut_mut = (prm.benefit - 1.0) * mutant.cooperation_level;
    double pi_res_res = (prm.benefit - 1.0) * resident.cooperation_level;
    auto payoffs = StrategyM3(mutant.strategy_id).Payoffs(StrategyM3(resident.strategy_id), prm.benefit, prm.error_rate);
    double pi_mut_res = payoffs[0], pi_res_mut = payoffs[1];

    double rho_mut = IntraGroupFixationProb(pi_mut_mut, pi_mut_res, pi_res_mut, pi_res_res);
    double rho_res = IntraGroupFixationProb(pi_res_res, pi_res_mut, pi_mut_res, pi_mut_mut);
    // \eta = Q_i^{-}/Q_i^{+} = \rho_B / \rho_A * \exp[ \sigma_g (\pi_B - \pi_A) ]
    double eta = rho_res / rho_mut * std::exp( prm.sigma_g * (pi_res_res - pi_mut_mut) );
    constexpr double tolerance = 1.0e-8;
    if (std::abs(eta - 1.0) < tolerance) {  // eta == 1
      return rho_mut / static_cast<double>(prm.M);
    }
    return rho_mut * (1.0 - eta) / (1.0 - std::pow(eta, prm.M));

    // when N == 2
    // double x1 = 1.0 + std::exp( prm.sigma*(pi_ba - pi_ab) );
    // double x2 = 1.0 - std::exp(         prm.sigma*(pi_ba - pi_ab) +         prm.sigma_g*(pi_bb - pi_aa) );
    // double x3 = 1.0 - std::exp( prm.M * prm.sigma*(pi_ba - pi_ab) + prm.M * prm.sigma_g*(pi_bb - pi_aa) );
    // return 1.0 / x1 * x2 / x3;
  }

  double IntraGroupFixationProb(double pi_mut_mut, double pi_mut_res, double pi_res_mut, double pi_res_res) const {
    // \frac{1}{\rho} = \sum_{i=0}^{N-1} \exp\left( \sigma \sum_{j=1}^{i} \left[(N-j-1)s_{yy} + js_{yx} - (N-j)s_{xy} - (j-1)s_{xx} \right] \right) \\
    //                = \sum_{i=0}^{N-1} \exp\left( \frac{\sigma i}{2} \left[(-i+2N-3)s_{yy} + (i+1)s_{yx} - (-i+2N-1)s_{xy} - (i-1)s_{xx} \right] \right)
    double s_xx = pi_mut_mut;
    double s_xy = pi_mut_res;  // mutant's payoff when played against resident
    double s_yx = pi_res_mut;  // resident's payoff when played against mutant
    double s_yy = pi_res_res;

    size_t N = prm.N;
    double c = prm.sigma * 0.5 / static_cast<double>(N-1);
    double rho_inv = 0.0;
    for (int i=0; i < N; i++) {
      double x = c * i * (
        (double)(2*N-3-i) * s_yy
        + (double)(i+1) * s_yx
        - (double)(2*N-1-i) * s_xy
        - (double)(i-1) * s_xx
      );
      rho_inv += std::exp(x);
    }
    return 1.0 / rho_inv;
  }

  double UnconditionalFixationTime(const Species& mutant, const Species& resident) const {
    double pi_mut_mut = (prm.benefit - 1.0) * mutant.cooperation_level;
    double pi_res_res = (prm.benefit - 1.0) * resident.cooperation_level;
    auto payoffs = StrategyM3(mutant.strategy_id).Payoffs(StrategyM3(resident.strategy_id), prm.benefit, prm.error_rate);
    double pi_mut_res = payoffs[0], pi_res_mut = payoffs[1];

    double rho_mut = IntraGroupFixationProb(pi_mut_mut, pi_mut_res, pi_res_mut, pi_res_res);
    double rho_res = IntraGroupFixationProb(pi_res_res, pi_res_mut, pi_mut_res, pi_mut_mut);
    // \eta = Q_i^{-}/Q_i^{+} = \rho_B / \rho_A * \exp[ \sigma_g (\pi_B - \pi_A) ]
    double eta = rho_res / rho_mut * std::exp( prm.sigma_g * (pi_res_res - pi_mut_mut) );

    // for eta == 1
    // t_1 = (M-1){ 1 + \exp[ \sigma_g(\pi_B - \pi_A)]} / \rho_A
    //       * \sum_{l=1}^{M-1}\frac{1}{l}
    constexpr double tolerance = 1.0e-8;
    if (std::abs(eta - 1.0) < tolerance) {  // eta == 1
      double x = static_cast<double>(prm.M-1) * (1.0 + std::exp( prm.sigma_g * (pi_res_res - pi_mut_mut)) ) / rho_mut;
      double sum = 0.0;
      for (size_t l = 1; l < prm.M; l++) { sum += 1.0 / static_cast<double>(l); }
      return x * sum;
    }

    // for eta != 1
    // t_1 = M(M-1){ 1 + \exp[ \sigma_g(\pi_B - \pi_A)]} / (1 - \eta^M)\rho_A}
    //       * \sum_{l=1}^{M-1} (1-\eta^{l}) / l(M-l)
    double num1 = prm.M * (prm.M-1) * (1.0 + std::exp( prm.sigma_g * (pi_res_res - pi_mut_mut) ));
    double den1 = (1.0 - std::pow(eta, prm.M)) * rho_mut;
    double sum = 0.0;
    for (size_t l = 1; l < prm.M; l++) {
      sum += (1.0 - std::pow(eta, l)) / static_cast<double>(l*(prm.M-l));
    }
    return num1 / den1 * sum;
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
