//
// Created by Yohsuke Murase on 2021/10/25.
//

#include <iostream>
#include <vector>
#include <array>
#include <map>
#include <random>
#include <cassert>
#include <fstream>
#include <chrono>
#include <Eigen/Dense>
#include <nlohmann/json.hpp>
#include <omp.h>
#include "icecream-cpp/icecream.hpp"
#include "StrategyM3.hpp"
#include "StrategySpace.hpp"


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


class MultilevelParallelEvoGame {
 public:
  MultilevelParallelEvoGame(const Parameters& _prm) :
  prm(_prm), space(prm.strategy_space[0], prm.strategy_space[1]),
  sample_space(0ull, space.Max()) {
    for (uint32_t t = 0; t < omp_get_max_threads(); t++) {
      std::seed_seq s = {static_cast<uint32_t>(prm._seed), t};
      a_rnd.emplace_back(s);
    }

    species.reserve(prm.M);
    if (prm.initial_condition == "random") {
      for (size_t i = 0; i < prm.M; i++) {
        uint64_t id = space.ToGlobalID( sample_space(a_rnd[0]));
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
      else { throw std::runtime_error("unknown initial condition"); }
      for (size_t i = 0; i < prm.M; i++) {
        species.emplace_back(str_id, prm.error_rate);
      }
    }
    IC(species);
  };
  Parameters prm;
  StrategySpace space;
  std::vector<Species> species;
  std::vector<std::mt19937_64> a_rnd;
  std::uniform_real_distribution<double> uni;
  std::uniform_int_distribution<uint64_t> sample_space;

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
    uint64_t mut_id = space.ToGlobalID( sample_space(a_rnd[th]) );
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


int main(int argc, char *argv[]) {
  #if defined(NDEBUG)
  icecream::ic.disable();
  #endif
  icecream::ic.prefix("[", omp_get_thread_num, "/" , omp_get_max_threads, "]: ");

  Eigen::initParallel();
  if( argc != 2 ) {
    std::cerr << "Error : invalid argument" << std::endl;
    std::cerr << "  Usage : " << argv[0] << " <parameter_json_file>" << std::endl;
    return 1;
  }

  Parameters prm;
  {
    std::ifstream fin(argv[1]);
    nlohmann::json input;
    fin >> input;
    prm = input.get<Parameters>();
  }

  MeasureElapsed("initialize");

  MultilevelParallelEvoGame eco(prm);

  uint64_t initial_species_id;
  bool is_measuring_lifetime = false;
  int64_t lifetime = -1;
  if (prm.initial_condition != "random") {
    initial_species_id = eco.species.at(0).strategy_id;
    is_measuring_lifetime = true;
  }

  MeasureElapsed("simulation");

  double c_level_avg = 0.0, fr_fraction = 0.0, efficient_fraction = 0.0, defensible_fraction = 0.0;
  double avg_diversity = 0.0;
  size_t count = 0ul;

  std::ofstream tout("timeseries.dat");

  for (size_t t = 0; t < prm.T_max; t++) {
    eco.Update();
    if (is_measuring_lifetime) {
      if (!eco.HasSpecies(initial_species_id)) {
        lifetime = t;
        is_measuring_lifetime = false;
      }
      else {
        lifetime = t+1;
      }
    }
    if (t > prm.T_init) {
      c_level_avg += eco.CooperationLevel();
      fr_fraction += (double)eco.NumFriendlyRival() / prm.M;
      efficient_fraction += (double)eco.NumEfficient() / prm.M;
      defensible_fraction += (double)eco.NumDefensible() / prm.M;
      avg_diversity += eco.Diversity() / prm.M;
      count++;
    }
    if (t % prm.T_print == prm.T_print - 1) {
      double m_inv = 1.0 / prm.M;
      tout << t + 1 << ' ' << eco.CooperationLevel() << ' ' << eco.NumFriendlyRival() * m_inv
                    << ' ' << eco.NumEfficient() * m_inv << ' ' << eco.NumDefensible() * m_inv
                    << ' ' << eco.Diversity() * m_inv << std::endl;
      // IC(t, eco.species);
    }
  }
  tout.close();

  {
    nlohmann::json output;
    output["cooperation_level"] = c_level_avg / count;
    output["friendly_rival_fraction"] = fr_fraction / count;
    output["efficient_fraction"] = efficient_fraction / count;
    output["defensible_fraction"] = defensible_fraction / count;
    output["diversity"] = avg_diversity / count;
    output["lifetime_init_species"] = lifetime;
    std::ofstream fout("_output.json");
    fout << output.dump(2);
    fout.close();
  }

  {
    nlohmann::json j;
    for (const Species& s: eco.species) {
      j.emplace_back(s);
    }
    std::ofstream fout("final_state.json");
    fout << j.dump(2);
    fout.close();
  }

  MeasureElapsed("done");
  return 0;
}