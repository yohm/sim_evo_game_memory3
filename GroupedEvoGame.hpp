//
// Created by Yohsuke Murase on 2021/10/29.
//

#ifndef CPP_GROUPED_EVO_GAME_HPP
#define CPP_GROUPED_EVO_GAME_HPP

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


class GroupedEvoGame {
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
    double sigma_in, sigma_out;
    double p_nu;  // probability of introducing a mutant
    std::array<size_t,2> strategy_space;
    int weighted_sampling;  // 1: weighted sampling, 0: uniform sampling
    int parallel_update;    // 1: parallel update, 0: serial update
    std::string initial_condition; // "random", "TFT", "WSLS", "TFT-ATFT", "CAPRI"
    uint64_t _seed;

    NLOHMANN_DEFINE_TYPE_INTRUSIVE(Parameters, T_max, T_print, T_init,
                                   M, N, benefit, error_rate, sigma_in, sigma_out, p_nu,
                                   strategy_space, weighted_sampling, parallel_update, initial_condition, _seed);
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

  explicit GroupedEvoGame(Parameters _prm) :
    prm(std::move(_prm)), space(prm.strategy_space[0], prm.strategy_space[1]) {
    for (uint32_t t = 0; t < omp_get_max_threads(); t++) {
      std::seed_seq s = {static_cast<uint32_t>(prm._seed), t};
      a_rnd.emplace_back(s);
    }

    species.reserve(prm.M);
    if (prm.initial_condition == "random") {
      for (size_t i = 0; i < prm.M; i++) {
        uint64_t id = SampleStrategySpace();
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
    if (prm.weighted_sampling < 0 || prm.weighted_sampling > 1) {
      throw std::runtime_error("unknown sampling type: Use 0(uniform) or 1(weighted)");
    }
    if (prm.parallel_update < 0 || prm.parallel_update > 1) {
      throw std::runtime_error("unknown update scheme: Use 0(serial update) or 1(paralell update)");
    }
  };
  Parameters prm;
  StrategySpace space;
  std::vector<Species> species;
  std::vector<std::mt19937_64> a_rnd;
  std::uniform_real_distribution<double> uni;
  std::map<std::pair<uint64_t,uint64_t>, double> prob_cache;
  std::map<uint64_t,Species> species_cache;

  double IntraGroupFixationProb(const Species& mutant, const Species& resident) const {
    // \frac{1}{\rho} = \sum_{i=0}^{N-1} \exp\left( \sigma_in \sum_{j=1}^{i} \left[(N-j-1)s_{yy} + js_{yx} - (N-j)s_{xy} - (j-1)s_{xx} \right] \right) \\
    //                = \sum_{i=0}^{N-1} \exp\left( \frac{\sigma_in i}{2} \left[(-i+2N-3)s_{yy} + (i+1)s_{yx} - (-i+2N-1)s_{xy} - (i-1)s_{xx} \right] \right)
    double s_xx = (prm.benefit - 1.0) * mutant.cooperation_level;     // == Payoffs(mutant_id, mutant_id)[0];
    double s_yy = (prm.benefit - 1.0) * resident.cooperation_level;   // == Payoffs(resident_id, resident_id)[0];
    StrategyM3 mut(mutant.strategy_id);
    StrategyM3 res(resident.strategy_id);
    auto xy = mut.Payoffs(res, prm.benefit, prm.error_rate);
    double s_xy = xy[0];
    double s_yx = xy[1];
    size_t N = prm.N;
    double sigma = prm.sigma_in;

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

  double InterGroupImitationProb(const Species& s_target, const Species& s_focal) const {
    double pi = (prm.benefit - 1.0) * s_focal.cooperation_level;
    double p_target = (prm.benefit - 1.0) * s_target.cooperation_level;
    // f_{A\to B} = { 1 + \exp[ \sigma_out (s_A - s_B) ] }^{-1}
    return 1.0 / (1.0 + std::exp(prm.sigma_out * (pi - p_target) ));
  }

  double FixationProbLowMutation(const Species& mutant, const Species& resident) const {
    //\Psi_{A} = 1/{1 + \exp[\sigma_in(\pi_{BA}-\pi_{AB})]}
    //         * {1-\exp[  \sigma_in(\pi_{BA}-\pi_{AB}) + \sigma_out( \pi_{BB}-\pi_{AA} ) ]}
    //         / {1-\exp[ M\sigma_in(\pi_{BA}-\pi_{AB}) +M\sigma_out( \pi_{BB}-\pi_{AA} ) ]}
    double pi_mut = (prm.benefit - 1.0) * mutant.cooperation_level;
    double pi_res = (prm.benefit - 1.0) * resident.cooperation_level;

    double rho_mut = IntraGroupFixationProb(mutant, resident);
    double rho_res = IntraGroupFixationProb(resident, mutant);
    // \eta = Q_i^{-}/Q_i^{+} = \rho_B / \rho_A * \exp[ \sigma_out (\pi_B - \pi_A) ]
    double eta = rho_res / rho_mut * std::exp(prm.sigma_out * (pi_res - pi_mut) );
    if (rho_mut == 0.0 && rho_res == 0.0) {
      icecream::ic.enable();
      IC(mutant, resident, pi_mut, pi_res, rho_mut, rho_res, eta, std::pow(eta, prm.M));
      throw std::runtime_error("cannot calculate fixation prob");
    }
    if (rho_mut == 0.0) {
      return 0.0;
    }
    // numerically calculate geometric series sum  1 + eta + eta^2 + ... + eta^(M-1)
    double denom = 1.0;
    double prod = 1.0;
    for (int i = 1; i < prm.M; i++) {
      prod *= eta;
      denom += prod;
    }
    return rho_mut / denom;
  }

  double UnconditionalFixationTimeLowMutation(const Species& mutant, const Species& resident) const {
    double pi_mut_mut = (prm.benefit - 1.0) * mutant.cooperation_level;
    double pi_res_res = (prm.benefit - 1.0) * resident.cooperation_level;
    double rho_mut = IntraGroupFixationProb(mutant, resident);
    double rho_res = IntraGroupFixationProb(resident, mutant);
    // \eta = Q_i^{-}/Q_i^{+} = \rho_B / \rho_A * \exp[ \sigma_out (\pi_B - \pi_A) ]
    double eta = rho_res / rho_mut * std::exp(prm.sigma_out * (pi_res_res - pi_mut_mut) );

    if (rho_mut == 0.0) {
      return std::numeric_limits<double>::infinity();
    }

    std::vector<double> eta_pow(prm.M, 1.0);  // eta_pow[i] = eta^i
    for (size_t i = 1; i < prm.M; i++) {
      eta_pow[i] = eta_pow[i - 1] * eta;
    }

    // phi_1 = 1 / 1 + eta + eta^2 + ... + eta^(M-1)
    double phi_1;
    {
      double sum = 0.0;
      for (size_t i = 0; i < prm.M; i++) {
        sum += eta_pow[i];
      }
      phi_1 = 1.0 / sum;
    }

    // t_1 = phi_1 M(M-1){ 1 + exp[sigma_out(pi_B - pi_A)]} / rho_A \sum_{k=1}^{M-1}\sum_{l=1}^{k} eta^(k-l)/l(M-l)
    double t = phi_1 * prm.M * (prm.M - 1) * (1.0 + std::exp(prm.sigma_out * (pi_res_res - pi_mut_mut))) / rho_mut;
    double sum = 0.0;
    for (int k = 1; k < prm.M; k++) {
      for (int l = 1; l <= k; l++) {
        sum += eta_pow[k - l] / static_cast<double>(l * (prm.M - l));
      }
    }
    return t * sum;
  }

  double ConditionalFixationTimeLowMutation(const Species& mutant, const Species& resident) const {
    double pi_mut_mut = (prm.benefit - 1.0) * mutant.cooperation_level;
    double pi_res_res = (prm.benefit - 1.0) * resident.cooperation_level;
    double rho_mut = IntraGroupFixationProb(mutant, resident);
    double rho_res = IntraGroupFixationProb(resident, mutant);
    // \eta = Q_i^{-}/Q_i^{+} = \rho_B / \rho_A * \exp[ \sigma_out (\pi_B - \pi_A) ]
    double eta = rho_res / rho_mut * std::exp(prm.sigma_out * (pi_res_res - pi_mut_mut) );

    if (rho_mut == 0.0) {
      return std::numeric_limits<double>::infinity();
    }

    std::vector<double> eta_pow(prm.M, 1.0);  // eta_pow[i] = eta^i
    for (size_t i = 1; i < prm.M; i++) {
      eta_pow[i] = eta_pow[i - 1] * eta;
    }

    // phi[l] = {1 + eta + ... + eta^(l-1)} / {1 + eta + eta^2 + ... + eta^(M-1)}
    std::vector<double> phi(prm.M, 0.0);
    {
      for (int l = 1; l < prm.M; l++) {
        phi[l] = phi[l-1] + eta_pow[l-1];
      }
      double denom = phi[prm.M-1] + eta_pow[prm.M-1];
      for (int l = 1; l < prm.M; l++) {
        phi[l] /= denom;
      }
    }

    // t_1^A = M(M-1){ 1 + exp[sigma_out(pi_B - pi_A)]} / rho_A \sum_{k=1}^{M-1}\sum_{l=1}^{k} eta^(k-l)phi_l/l(M-l)
    double t = prm.M * (prm.M-1) * (1.0 + std::exp(prm.sigma_out * (pi_res_res - pi_mut_mut))) / rho_mut;
    double sum = 0.0;
    for (int k = 1; k < prm.M; k++) {
      for (int l = 1; l <= k; l++) {
        sum += eta_pow[k-l] * phi[l] / static_cast<double>(l*(prm.M-l));
      }
    }
    return t * sum;
  }

  uint64_t UniformSampleStrategySpace() {
    const int th = omp_get_thread_num();
    std::uniform_int_distribution<uint64_t> sample(0ull, space.Max());
    return space.ToGlobalID( sample(a_rnd[th]));
  }

  uint64_t WeightedSampleStrategySpace() {
    const int th = omp_get_thread_num();
    size_t num_spaces = (space.mem[0]+1) * (space.mem[1]+1);
    auto mi = static_cast<size_t>( uni(a_rnd[th]) * (double)num_spaces );
    size_t m1 = mi % (space.mem[0]+1), m2 = mi / (space.mem[0]+1);
    StrategySpace ss(m1, m2);
    std::uniform_int_distribution<uint64_t> sample(0ull, ss.Max());
    uint64_t gid = ss.ToGlobalID( sample(a_rnd[th]));
    const StrategySpace::mem_t target({m1, m2});
    while (StrategySpace::MemLengths(gid) != target) {
      gid = ss.ToGlobalID( sample(a_rnd[th]));
    }
    return gid;
  }

  uint64_t SampleStrategySpace() {
    return (prm.weighted_sampling==1) ? WeightedSampleStrategySpace() : UniformSampleStrategySpace();
  }

  void Update() {
    if (prm.parallel_update == 1) {
      UpdateParallel();
    }
    else {
      UpdateSerial();
    }
  }

  void UpdateSerial() {
    for (int t = 0; t < prm.M; t++) {
      std::uniform_int_distribution<size_t> d0(0, prm.M-1);
      size_t res_index = d0(a_rnd[0]);
      Species resident = species[res_index];
      if (uni(a_rnd[0]) < prm.p_nu) {  // mutation
        uint64_t mut_id = SampleStrategySpace();
        auto it = species_cache.find(mut_id);
        Species mutant = (it == species_cache.end()) ? Species(mut_id, prm.error_rate) : it->second;
        if (it == species_cache.end() && mutant.mem_lengths[0]+mutant.mem_lengths[1] <= 4) {
          species_cache.insert( std::make_pair(mut_id, mutant));
        }
        double f = IntraGroupFixationProb(mutant, resident);
        if (uni(a_rnd[0]) < f) species[res_index] = mutant;
      }
      else {  // migration
        std::uniform_int_distribution<size_t> d1(1, prm.M-1);
        size_t mig_index = static_cast<size_t>(res_index + d1(a_rnd[0])) % prm.M;
        Species immigrant = species[mig_index];
        auto it = prob_cache.find({immigrant.strategy_id, resident.strategy_id});
        double prob;
        if (it == prob_cache.end()) {
          double p = InterGroupImitationProb(immigrant, resident);
          double f = IntraGroupFixationProb(immigrant, resident);
          prob = p*f;
          prob_cache.insert({ {immigrant.strategy_id, resident.strategy_id}, prob});
        }
        else {
          prob = it->second;
        }
        if (uni(a_rnd[0]) < prob) {
          species[res_index] = immigrant;
        }
      }
    }
    constexpr size_t CACHE_SIZE_MAX = 1000;
    if (prob_cache.size() > CACHE_SIZE_MAX) {
      std::cerr << "deleting cache" << std::endl;
      ClearCache();
      std::cerr << "  cache size: " << prob_cache.size() << std::endl;
    }
  }

  void UpdateParallel() {
    std::vector<Species> candidates = species;
    #pragma omp parallel for
    for (size_t i = 0; i < prm.M; i++) {
      int th = omp_get_thread_num();
      if (uni(a_rnd[th]) < prm.p_nu) {
        uint64_t mut_id = SampleStrategySpace();
        candidates[i] = Species(mut_id, prm.error_rate);
      }
      else {
        std::uniform_int_distribution<size_t> dist(1, prm.M-1);
        size_t target = static_cast<size_t>(i + dist(a_rnd[th])) % prm.M;
        double p = InterGroupImitationProb(species[target], species[i]);
        if (uni(a_rnd[th]) < p) {
          candidates[i] = species[target];
        }
      }
    }

    #pragma omp parallel for
    for (size_t i = 0; i < prm.M; i++) {
      if (candidates[i].strategy_id == species[i].strategy_id) continue;
      double f = IntraGroupFixationProb(candidates[i], species[i]);
      int th = omp_get_thread_num();
      if (uni(a_rnd[th]) < f) {
        species[i] = candidates[i];
      }
    }
  }

  void ClearCache() {
    std::set<uint64_t> existing;
    for (const auto& s: species) {
      existing.insert(s.strategy_id);
    }

    for (auto it = prob_cache.begin(); it != prob_cache.end(); ) {
      uint64_t sid1 = it->first.first, sid2 = it->first.second;
      if (existing.find(sid1) == existing.end() || existing.find(sid2) == existing.end()) {
        // if strategy does not exist in the current species
        it = prob_cache.erase(it);
      }
      else {
        it++;
      }
    }
  }

  double CooperationLevel() const {
    double ans = 0.0;
    for (const Species& s: species) {
      ans += s.cooperation_level;
    }
    return ans / (double)species.size();
  }

  // number of efficient but not FR species
  size_t NumEfficient() const {
    size_t count = 0;
    for (const Species& s: species) {
      if (s.is_efficient && !s.is_defensible) count++;
    }
    return count;
  }

  // number of defensible but not FR species
  size_t NumDefensible() const {
    size_t count = 0;
    for (const Species& s: species) {
      if (s.is_defensible && !s.is_efficient) count++;
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
    return std::any_of(species.begin(), species.end(), [species_id](const Species& s) {
      return s.strategy_id == species_id;
    });
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
      double p = pair.second / (double)prm.M;
      entropy += -p * std::log(p);
    }
    return std::exp(entropy);
  }

  std::array<double,2> AverageMemLengths() const {
    std::array<double, 2> ans = {0.0, 0.0};
    for (const Species& s: species) {
      ans[0] += static_cast<double>(s.mem_lengths[0]);
      ans[1] += static_cast<double>(s.mem_lengths[1]);
    }
    ans[0] /= (double)species.size();
    ans[1] /= (double)species.size();
    return ans;
  }

  std::array<double,2> AverageAutomatonSizes() const { // average number of states of minimized automaton
    std::array<double,2> ans = {0.0, 0.0};
    for (const Species& s: species) {
      ans[0] += (double)s.automaton_sizes[0];
      ans[1] += (double)s.automaton_sizes[1];
    }
    ans[0] /= (double)species.size();
    ans[1] /= (double)species.size();
    return ans;
  }
};


#endif //CPP_GROUPED_EVO_GAME_HPP
