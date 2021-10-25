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
  size_t M, N;
  double benefit;
  double error_rate;
  double sigma, sigma_g;
  double T_g;
  std::array<size_t,2> strategy_space;
  uint64_t _seed;

  NLOHMANN_DEFINE_TYPE_INTRUSIVE(Parameters, T_max, M, N, benefit, error_rate, sigma, sigma_g, T_g, strategy_space, _seed);
};


class MultilevelEvoGame {
 public:
  MultilevelEvoGame(const Parameters& _prm) : prm(_prm), space(prm.strategy_space[0], prm.strategy_space[1]), rnd(prm._seed) {
    species.resize(prm.M);
    for (size_t i = 0; i < species.size(); i++) {
      species[i] = uni(rnd) * space.Size();
    }
  };
  Parameters prm;
  StrategySpace space;
  std::vector<uint64_t> species;
  std::mt19937_64 rnd;
  std::uniform_real_distribution<double> uni;

  // payoff of species i and j when the game is played by (i,j)
  std::array<double,2> Payoffs(size_t i, size_t j) const {
    StrategyM3 si(space.ToGlobalID(i) );
    StrategyM3 sj(space.ToGlobalID(j) );
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

  double FixationProb(size_t mutant_idx, size_t resident_idx) const {
    // \frac{1}{\rho} = \sum_{i=0}^{N-1} \exp\left( \sigma \sum_{j=1}^{i} \left[(N-j-1)s_{yy} + js_{yx} - (N-j)s_{xy} - (j-1)s_{xx} \right] \right) \\
    //                = \sum_{i=0}^{N-1} \exp\left( \frac{\sigma i}{2} \left[(-i+2N-3)s_{yy} + (i+1)s_{yx} - (-i+2N-1)s_{xy} - (i-1)s_{xx} \right] \right)
    double s_xx = Payoffs(mutant_idx, mutant_idx)[0];
    double s_yy = Payoffs(resident_idx, resident_idx)[0];
    auto xy = Payoffs(mutant_idx, resident_idx);
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

  double SelectionProb(uint64_t species_a, uint64_t species_b) const {
    double s_a = Payoffs(species_a, species_a)[0];
    double s_b = Payoffs(species_b, species_b)[0];
    // f_{A\to B} = { 1 + \exp[ \sigma_g (s_A - s_B) ] }^{-1}
    return 1.0 / (1.0 + std::exp( prm.sigma_g * (s_a - s_b) ));
  }

  void Update() {
    double intra = prm.T_g / (1.0 + prm.T_g);  // probability of intra-group selection
    if (uni(rnd) < intra) {
      IntraGroupSelection();
    }
    else {
      InterGroupSelection();
    }
  }

  void IntraGroupSelection() {
    size_t g = uni(rnd) * prm.M;
    uint64_t mut_id = uni(rnd) * space.Size();
    uint64_t res_id = species[g];
    double f = FixationProb(mut_id, res_id);
    if (uni(rnd) < f) {
      species[g] = mut_id;
    }
  }

  void InterGroupSelection() {
    size_t g1 = uni(rnd) * prm.M;
    size_t g2 = static_cast<size_t>(g1 + 1 + uni(rnd) * (prm.M-1)) % prm.M;
    double p = SelectionProb(species[g1], species[g2]);
    if (uni(rnd) < p) {
      species[g1] = species[g2];
    }
  }

  double CooperationLevel() const {
    // [TODO] IMPLEMENT ME
    return 0.0;
  }
};


int main(int argc, char *argv[]) {
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

  MultilevelEvoGame eco(prm);

  MeasureElapsed("simulation");

  for (size_t t = 0; t < prm.T_max; t++) {
    eco.Update();
    IC(t, eco.species);
    eco.CooperationLevel();
  }

  MeasureElapsed("done");
  return 0;
}