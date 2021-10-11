//
// Created by Yohsuke Murase on 2021/10/10.
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
#include "StrategyM3.hpp"
#include "StrategySpace.hpp"


std::string prev_key;
std::chrono::system_clock::time_point start;
void MeasureElapsed(const std::string& key) {
  std::chrono::system_clock::time_point end = std::chrono::system_clock::now();
  if (!prev_key.empty()) {
    double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();
    std::cerr << "T " << prev_key << " finished in " << elapsed << " ms" << std::endl;
  }
  start = end;
  prev_key = key;
}


class EvolutionaryGame {
 public:
  EvolutionaryGame(const StrategySpace& _space, double error) : space(_space), N_SPECIES(_space.Size()), e(error) {
    CalculateSSCache();
  };
  const StrategySpace space;
  const size_t N_SPECIES;
  const double e;
  using ss_cache_t = std::array<double,2>;  // probability of getting benefit & paying cost
  std::vector<std::vector<ss_cache_t> > ss_cache;
  // ss_cache[i][j] stores the stationary state when PG game is played by (i,j)

  void CalculateSSCache() {
    ss_cache.resize(N_SPECIES);
    for (size_t i = 0; i < N_SPECIES; i++) {
      ss_cache[i].resize(N_SPECIES, {0.0, 0.0});
    }

    #pragma omp parallel for schedule(dynamic,1)
    for (uint64_t I=0; I < N_SPECIES * N_SPECIES; I++) {
      uint64_t i = I / N_SPECIES;
      uint64_t j = I % N_SPECIES;
      if (i < j) continue;
      StrategyM3 si(space.ToGlobalID(i) );
      StrategyM3 sj(space.ToGlobalID(j) );
      auto p = si.StationaryState(e, &sj);
      for (size_t n = 0; n < 64; n++) {
        StateM3 s(n);
        if (i > j) {
          if (s.a_1 == C) {
            ss_cache[i][j][1] += p[n];
            ss_cache[j][i][0] += p[n];
          }
          if (s.b_1 == C) {
            ss_cache[i][j][0] += p[n];
            ss_cache[j][i][1] += p[n];
          }
        }
        else if (i == j) {
          if (s.a_1 == C) {
            ss_cache[i][j][1] += p[n];
          }
          if (s.b_1 == C) {
            ss_cache[i][j][0] += p[n];
          }
        }
      }
    }
  }

  // payoff of species i and j when the game is played by (i,j)
  std::array<double,2> PayoffVersus(size_t i, size_t j, double benefit, double cost) const {
    return {
      ss_cache[i][j][0] * benefit - ss_cache[i][j][1] * cost,
      ss_cache[j][i][0] * benefit - ss_cache[j][i][1] * cost
    };
  }

  // calculate the equilibrium distribution exactly by linear algebra
  std::vector<double> CalculateEquilibrium(double benefit, double cost, uint64_t N, double sigma) const {
    Eigen::MatrixXd A(N_SPECIES, N_SPECIES);
    #pragma omp parallel for
    for (size_t ii = 0; ii < N_SPECIES * N_SPECIES; ii++) {
      size_t i = ii / N_SPECIES;
      size_t j = ii % N_SPECIES;
      if (i == j) { A(i, j) = 0.0; continue; }
      double p = FixationProb(benefit, cost, N, sigma, i, j);
      // std::cerr << "Fixation prob of mutant (mutant,resident): " << p << " (" << pool[i].ToString() << ", " << pool[j].ToString() << ")" << std::endl;
      A(i, j) = p * (1.0 / N_SPECIES);
    }

    for (size_t j = 0; j < N_SPECIES; j++) {
      double p_sum = 0.0;
      for (size_t i = 0; i < N_SPECIES; i++) {
        p_sum += A(i, j);
      }
      assert(p_sum <= 1.0);
      A(j, j) = 1.0 - p_sum; // probability that the state doesn't change
    }

    // subtract Ax = x => (A-I)x = 0
    for (size_t i = 0; i < A.rows(); i++) {
      A(i, i) -= 1.0;
    }
    // normalization condition
    for (size_t i = 0; i < A.rows(); i++) {
      A(A.rows()-1, i) += 1.0;
    }

    Eigen::VectorXd b(A.rows());
    for(int i=0; i<A.rows()-1; i++) { b(i) = 0.0;}
    b(A.rows()-1) = 1.0;
    Eigen::VectorXd x = A.householderQr().solve(b);
    std::vector<double> ans(A.rows());
    double prob_total = 0.0;
    for(int i=0; i<ans.size(); i++) {
      ans[i] = x(i);
      prob_total += x(i);
      assert(x(i) > -0.000001);
    }
    assert(std::abs(prob_total - 1.0) < 0.00001);
    return ans;
  }

  double FixationProb(double benefit, double cost, uint64_t N, double sigma, size_t mutant_idx, size_t resident_idx) const {
    // \frac{1}{\rho} = \sum_{i=0}^{N-1} \exp\left( \sigma \sum_{j=1}^{i} \left[(N-j-1)s_{yy} + js_{yx} - (N-j)s_{xy} - (j-1)s_{xx} \right] \right) \\
    //                = \sum_{i=0}^{N-1} \exp\left( \frac{\sigma i}{2} \left[(-i+2N-3)s_{yy} + (i+1)s_{yx} - (-i+2N-1)s_{xy} - (i-1)s_{xx} \right] \right)

    double s_xx = PayoffVersus(mutant_idx, mutant_idx, benefit, cost)[0];
    double s_yy = PayoffVersus(resident_idx, resident_idx, benefit, cost)[0];
    auto xy = PayoffVersus(mutant_idx, resident_idx, benefit, cost);
    double s_xy = xy[0];
    double s_yx = xy[1];

    auto num_games = static_cast<double>(N-1);
    s_xx /= num_games;
    s_yy /= num_games;
    s_xy /= num_games;
    s_yx /= num_games;
    double rho_inv = 0.0;
    for (int i=0; i < N; i++) {
      double x = sigma * i * 0.5 * (
          (2*N-3-i) * s_yy
          + (i+1) * s_yx
          - (2*N-1-i) * s_xy
          - (i-1) * s_xx
      );
      rho_inv += std::exp(x);
    }
    return 1.0 / rho_inv;
  }

  double CooperationLevelSpecies(size_t i) const {
    return ss_cache[i][i][0];
  }
  double CooperationLevel(const std::vector<double> &eq_rate) const {
    assert(eq_rate.size() == N_SPECIES);
    double ans = 0.0;
    for (size_t i = 0; i < N_SPECIES; i++) {
      double c_lev = CooperationLevelSpecies(i);
      ans += eq_rate[i] * c_lev;
    }
    return ans;
  }
};


int main(int argc, char *argv[]) {
  Eigen::initParallel();
  if( argc != 2 ) {
    std::cerr << "Error : invalid argument" << std::endl;
    std::cerr << "  Usage : " << argv[0] << " <parameter_json_file>" << std::endl;
    return 1;
  }

  double cost = 1.0;
  nlohmann::json input;
  {
    std::ifstream fin(argv[1]);
    fin >> input;
  }

  const uint64_t Nmax = input.at("Nmax").get<uint64_t>();
  const double sigma = input.at("sigma").get<double>();
  const double e = input.at("error_rate").get<double>();
  int m0 = input.at("strategy_space").at(0).get<int>();
  int m1 = input.at("strategy_space").at(1).get<int>();
  StrategySpace space(m0, m1);

  /*
  for (const nlohmann::json& _s: input.at("additional")) {
    const std::string s = _s.get<std::string>();
    if (s == "CAPRI2") {
      pool.emplace_back(N_M1+0, discrete_level);
    }
    else if (s == "CAPRI") {
      pool.emplace_back(N_M1+1, discrete_level);
    }
    else if (s == "TFT-ATFT") {
      pool.emplace_back(N_M1 + 2, discrete_level);
    }
    else if (s == "AON2") {
      pool.emplace_back(N_M1 + 3, discrete_level);
    }
    else if (s == "AON3") {
      pool.emplace_back(N_M1 + 4, discrete_level);
    }
    else {
      std::cerr << "[Error] unknown strategy " << s << " is given." << std::endl;
      throw std::runtime_error("unknown species");
    }
  }
   */

  MeasureElapsed("initialize");

  EvolutionaryGame eco(space, e);

  MeasureElapsed("sweep over beta_N");

  auto SweepOverBeta = [&eco,cost,sigma](size_t N)->std::vector<std::pair<double,double>> {
    char fname1[100];
    sprintf(fname1, "abundance_%zu.dat", N);
    std::ofstream eqout(fname1);
    std::vector<std::pair<double,double>> c_levels;
    for (int i = 5; i <= 300; i+=5) {
      double benefit = 1.0 + i / 100.0;
      auto eq = eco.CalculateEquilibrium(benefit, cost, N, sigma);
      eqout << benefit << ' ';
      for (double x: eq) { eqout << x << ' '; }
      eqout << std::endl;
      double c_lev = eco.CooperationLevel(eq);
      c_levels.emplace_back(benefit, c_lev);
    }
    return c_levels;
  };

  std::vector<std::vector<std::pair<double,double>>> ans;
  for (int N = 2; N <= Nmax; N++) {
    auto a = SweepOverBeta(N);
    ans.push_back(a);
  }

  MeasureElapsed("calculate cooperation level");

  std::ofstream fout("cooperation_level.dat");
  for (size_t i = 0; i < ans[0].size(); i++) {
    fout << ans[0][i].first;
    for (size_t j = 0; j < ans.size(); j++) {
      fout << ' ' << ans[j][i].second;
    }
    fout << "\n";
  }
  fout.close();

  MeasureElapsed("done");
  return 0;
}