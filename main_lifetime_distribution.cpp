//
// Created by Yohsuke Murase on 2021/10/25.
//

#include <iostream>
#include <vector>
#include <array>
#include <chrono>
#include <regex>
#include "omp.h"
#include "icecream-cpp/icecream.hpp"
#include "GroupedEvoGame.hpp"


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

size_t MeasureLifetime(const GroupedEvoGame::Parameters & prm) {
  GroupedEvoGame eco(prm);

  uint64_t initial_species_id = eco.species.at(0).strategy_id;

  for (size_t t = 0; t < prm.T_max; t++) {
    eco.Update();
    if (!eco.HasSpecies(initial_species_id)) {
      return t;
    }
  }
  return prm.T_max;
}


int main(int argc, char *argv[]) {
  #if not defined(NDEBUG)
  icecream::ic.prefix("[", omp_get_thread_num, "/" , omp_get_max_threads, "]: ");
  #endif

  Eigen::initParallel();
  if( argc != 3 ) {
    std::cerr << "Error : invalid argument" << std::endl;
    std::cerr << "  Usage : " << argv[0] << " <parameter_json_file> <num_samples>" << std::endl;
    return 1;
  }


  GroupedEvoGame::Parameters org_prm;
  {
    std::ifstream fin(argv[1]);
    nlohmann::json input;
    fin >> input;
    org_prm = input.get<GroupedEvoGame::Parameters>();
    org_prm.alld_mutant_prob = input.value("alld_mutant_prob", 0.0);
  }

  size_t num_samples = std::stoul(argv[2]);

  std::map<size_t, size_t> lifetime_distribution;
  double sum = 0.0;
  #pragma omp parallel for schedule(dynamic, 1)
  for (size_t i = 0; i < num_samples; ++i) {
    GroupedEvoGame::Parameters prm = org_prm;
    prm._seed = org_prm._seed + i;
    size_t l = MeasureLifetime(prm);
    std::cerr << "(i,l,seed):" << i << ", " << l << ", " << prm._seed << std::endl;
    #pragma omp critical
    {
      if (lifetime_distribution.find(l) == lifetime_distribution.end()) {
        lifetime_distribution[l] = 0;
      }
      lifetime_distribution[l] += 1;
      sum += l;
    }
  }

  std::cout << "mean: " << sum / num_samples << std::endl;
  std::cout << "num_mut: " << sum / num_samples * org_prm.p_nu << std::endl;
  icecream::ic.enable();
  IC(lifetime_distribution);

  return 0;
}
