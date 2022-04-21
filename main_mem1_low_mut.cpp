//
// Created by Yohsuke Murase on 2021/10/25.
//

#include <iostream>
#include <vector>
#include <array>
#include <chrono>
#include <Eigen/Dense>
#include "GroupedEvoGame.hpp"
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

  if (argc < 7) {
    std::cerr << "[Error] invalid arguments" << std::endl;
    std::cerr << "  Usage: " << argv[0] << " <benefit> <error_rate> <N> <M> <sigma_in> <sigma_out>" << std::endl;
  }

  GroupedEvoGame::Parameters prm;
  prm.benefit = std::stod(argv[1]);
  prm.error_rate = std::stod(argv[2]);
  prm.N = std::stoi(argv[3]);
  prm.M = std::stoi(argv[4]);
  prm.sigma_in = std::stod(argv[5]);
  prm.sigma_out = std::stod(argv[6]);
  prm.strategy_space = {1, 1};
  prm.initial_condition = "random";
  prm.weighted_sampling = 0;
  prm.parallel_update = 0;

  std::cerr
    << "benefit: " << prm.benefit << std::endl
    << "error_rate: " << prm.error_rate << std::endl
    << "N: " << prm.N << std::endl
    << "M: " << prm.M << std::endl
    << "sigma_in: " << prm.sigma_in << std::endl
    << "sigma_out: " << prm.sigma_out << std::endl;

  GroupedEvoGame eco(prm);

  // prepare memory-1 species
  StrategySpace ss(1, 1);
  constexpr size_t N_SPECIES = 16;
  std::vector<GroupedEvoGame::Species> v_species;
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
      psi[i][j] = eco.FixationProbLowMutation(v_species[i], v_species[j]);
    }
  }
  {
    std::ofstream psi_out("fixation_probs.dat");
    for (size_t i = 0; i < N_SPECIES; i++) {
      for (size_t j = 0; j < N_SPECIES; j++) {
        psi_out << psi[i][j] << ' ';
      }
      psi_out << "\n";
    }
    psi_out.close();
  }

  // calculate equilibrium fraction
  auto px = CalculateEquilibrium(psi);
  {
    std::ofstream fout("abundance.dat");
    for (size_t i = 0; i < N_SPECIES; i++) {
      fout << px[i] << "\n";
    }
    fout.close();

    std::ofstream jout("_output.json");
    double c = 0.0;
    for (size_t i = 0; i < N_SPECIES; i++) {
      c += v_species[i].cooperation_level * px[i];
    }
    jout << "{\"cooperation_level\": " << c << " }" << std::endl;
  }

  // calculate unconditional fixation time matrix
  // t_1 = \frac{M(M-1) { 1 + \exp[ \sigma_out(\pi_B - \pi_A)]} }{(1 - \eta^M)\rho_A} \sum_{l=1}^{M-1} \frac{1-\eta^{l}}{l(M-l)}
  //     = M(M-1){ 1 + \exp[ \sigma_out(\pi_B - \pi_A)]} / (1 - \eta^M)\rho_A}
  //       * \sum_{l=1}^{M-1} (1-\eta^{l}) / l(M-l)
  std::vector<std::vector<double>> t_1(N_SPECIES, std::vector<double>(N_SPECIES, 0.0));
  for (size_t i = 0; i < N_SPECIES; i++) {
    for (size_t j = 0; j < N_SPECIES; j++) {
      t_1[i][j] = eco.UnconditionalFixationTimeLowMutation(v_species[i], v_species[j]);
    }
  }
  {
    std::ofstream fout("fixation_times.dat");
    for (size_t i = 0; i < N_SPECIES; i++) {
      for (size_t j = 0; j < N_SPECIES; j++) {
        fout << t_1[i][j] << ' ';
      }
      fout << "\n";
    }
    fout.close();
  }

  return 0;
}