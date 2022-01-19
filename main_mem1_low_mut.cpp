//
// Created by Yohsuke Murase on 2021/10/25.
//

#include <iostream>
#include <vector>
#include <array>
#include <chrono>
#include "MultiLevelEvoGameLowMutation.hpp"
#include "icecream-cpp/icecream.hpp"


// calculate the equilibrium distribution by linear algebra
// fixation_probs[i][j]: fixation probability of i into j
std::vector<double> CalculateEquilibrium(const std::vector<std::vector<double>>& fixation_probs) {
  const size_t N_SPECIES = fixation_probs.size();
  Eigen::MatrixXd A(N_SPECIES, N_SPECIES);
  for (size_t i = 0; i < N_SPECIES; i++) {
    for (size_t j = 0; j < N_SPECIES; j++) {
      if (i == j) { A(i, j) = 0.0; continue; }
      A(i, j) = fixation_probs[i][j] * (1.0 / N_SPECIES);
    }
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


int main(int argc, char *argv[]) {
  #if defined(NDEBUG)
  icecream::ic.disable();
  #endif

  Eigen::initParallel();

  MultiLevelEvoGameLowMutation::Parameters prm;
  prm.benefit = 2.0;
  prm.error_rate = 1.0e-3;
  prm.N = 2;
  prm.M = 100;
  prm.sigma = 10.0;
  prm.sigma_g = 10.0;
  prm.strategy_space = {1, 1};
  prm.initial_condition = "random";

  MultiLevelEvoGameLowMutation eco(prm);

  // prepare memory-1 species
  StrategySpace ss(1, 1);
  constexpr size_t N_SPECIES = 16;
  std::vector<MultiLevelEvoGameLowMutation::Species> v_species;
  for (uint64_t i = 0; i < N_SPECIES; i++) {
    uint64_t gid = ss.ToGlobalID(i);
    v_species.emplace_back(gid, prm.error_rate);
  }
  // IC(v_species);

  // calculate payoff matrix
  std::vector<std::vector<double>> pi(N_SPECIES, std::vector<double>(N_SPECIES, 0.0));
  // pi[i][j] => payoff of i when pitted against j
  for (size_t i = 0; i < N_SPECIES; i++) {
    for (size_t j = 0; j < N_SPECIES; j++) {
      StrategyM3 str_i(v_species[i].strategy_id);
      StrategyM3 str_j(v_species[j].strategy_id);
      auto payoffs = StrategyM3(str_i).Payoffs(str_j, prm.benefit, prm.error_rate);
      pi[i][j] = payoffs[0];
    }
  }
  // IC(pi);

  // calculate fixation prob matrix
  std::vector<std::vector<double>> psi(N_SPECIES, std::vector<double>(N_SPECIES, 0.0));
  // psi[i][j] => fixation probability of mutant i into resident j community
  for (size_t i = 0; i < N_SPECIES; i++) {
    for (size_t j = 0; j < N_SPECIES; j++) {
      psi[i][j] = eco.FixationProb(v_species[i], v_species[j]);
    }
  }
  IC(psi);

  // calculate equilibrium fraction
  auto px = CalculateEquilibrium(psi);
  IC(px);

  return 0;
}