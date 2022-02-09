//
// Created by Yohsuke Murase on 2021/10/25.
//

#include <iostream>
#include <vector>
#include <array>
#include <chrono>
#include <Eigen/Dense>
#include "MultiLevelEvoGame.hpp"
#include "icecream-cpp/icecream.hpp"


int main(int argc, char *argv[]) {
  #if defined(NDEBUG)
  icecream::ic.disable();
  #endif
  Eigen::initParallel();

  if (argc < 10) {
    std::cerr << "[Error] invalid arguments" << std::endl;
    std::cerr << "  Usage: " << argv[0] << " <benefit> <error_rate> <N> <M> <sigma> <sigma_g> <T_max> <T_init> <_seed> [SPECIES_LIST]" << std::endl;
  }

  MultiLevelEvoGame::Parameters prm;
  prm.benefit = std::stod(argv[1]);
  prm.error_rate = std::stod(argv[2]);
  prm.N = std::stoi(argv[3]);
  prm.M = std::stoi(argv[4]);
  prm.sigma = std::stod(argv[5]);
  prm.sigma_g = std::stod(argv[6]);
  prm.T_max = std::stoul(argv[7]);
  prm.T_init = std::stoul(argv[8]);
  prm._seed = std::stoull(argv[9]);
  prm.strategy_space = {1, 1};
  prm.initial_condition = "random";
  prm.weighted_sampling = 0;
  prm.parallel_update = 0;

  std::cerr
    << "benefit: " << prm.benefit << std::endl
    << "error_rate: " << prm.error_rate << std::endl
    << "N: " << prm.N << std::endl
    << "M: " << prm.M << std::endl
    << "sigma: " << prm.sigma << std::endl
    << "sigma_g: " << prm.sigma_g << std::endl
    << "T_max: " << prm.T_max << std::endl
    << "T_init: " << prm.T_init << std::endl;

  MultiLevelEvoGame eco(prm);

  // prepare memory-1 species
  StrategySpace ss(1, 1);
  std::vector<MultiLevelEvoGame::Species> v_species;
  size_t N_SPECIES = 16;
  if (argc == 10) {
    for (uint64_t i = 0; i < N_SPECIES; i++) {
      uint64_t gid = ss.ToGlobalID(i);
      v_species.emplace_back(gid, prm.error_rate);
    }
  }
  else { // argc > 10
    N_SPECIES = 0;
    for (int i = 10; i < argc; i++) {
      uint64_t gid = ss.ToGlobalID( std::stoull(argv[i]));
      v_species.emplace_back(gid, prm.error_rate);
      N_SPECIES++;
    }
  }
  std::cerr << "v_species:\n";
  for (const auto& s: v_species) {
    std::cerr << "  " << ss.ToLocalID(s.strategy_id) << "\n";
  }
  IC(v_species);

  // calculate fixation prob matrix
  std::vector<std::vector<double>> psi(16, std::vector<double>(16, 0.0));
  // psi[i][j] => fixation probability of mutant i into resident j community
  for (size_t i = 0; i < N_SPECIES; i++) {
    for (size_t j = 0; j < N_SPECIES; j++) {
      // psi[i][j] = eco.FixationProbLowMutation(v_species[i], v_species[j]);
      double p = eco.FixationProbLowMutation(v_species[i], v_species[j]);
      psi[ss.ToLocalID(v_species[i].strategy_id)][ss.ToLocalID(v_species[j].strategy_id)] = p;
    }
  }

  {
    std::ofstream psi_out("fixation_probs.dat");
    for (size_t i = 0; i < psi.size(); i++) {
      for (size_t j = 0; j < psi[i].size(); j++) {
        psi_out << psi[i][j] << ' ';
      }
      psi_out << "\n";
    }
    psi_out.close();
  }

  // calculate equilibrium fraction
  {
    std::vector<long> freq(N_SPECIES, 0);
    long total_count = 0;
    size_t current = 0ul;
    std::uniform_real_distribution<> uni(0.0, 1.0);
    std::uniform_int_distribution<size_t> sample_species(0, N_SPECIES - 1);
    for (size_t t = 0; t < prm.T_max; t++) {
      size_t mut_idx = sample_species(eco.a_rnd[0]);
      double p = eco.FixationProbLowMutation(v_species[mut_idx], v_species[current]);
      double r = uni(eco.a_rnd[0]);
      if (r < p) {
        current = mut_idx;
      }
      if (t >= prm.T_init) {
        freq[current]++;
        total_count++;
      }
    }
    std::vector<double> abundance(16, 0.0);
    for (size_t i = 0; i < N_SPECIES; i++) {
      abundance[ss.ToLocalID(v_species[i].strategy_id)] = (double)freq[i] / total_count;
    }
    std::ofstream fout("abundance.dat");
    for (double x: abundance) { fout << x << "\n"; }
    fout.close();

    std::ofstream jout("_output.json");
    double c = 0.0;
    for (size_t i = 0; i < N_SPECIES; i++) {
      c += v_species[i].cooperation_level * (double)freq[i] / (double)total_count;
    }
    jout << "{\"cooperation_level\": " << c << " }" << std::endl;
  }

  return 0;
}