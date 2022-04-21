//
// Created by Yohsuke Murase on 2021/10/25.
//

#include <iostream>
#include <vector>
#include <array>
#include <chrono>
#include <regex>
#include "GroupedEvoGame.hpp"
#include "icecream-cpp/icecream.hpp"


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

  GroupedEvoGame::Parameters prm;
  {
    std::ifstream fin(argv[1]);
    nlohmann::json input;
    fin >> input;
    prm = input.get<GroupedEvoGame::Parameters>();

    if (prm.strategy_space != std::array<size_t,2>{1,1}) {
      throw std::runtime_error("strategy space must be {1,1}");
    }
  }

  MeasureElapsed("initialize");

  GroupedEvoGame eco(prm);

  uint64_t initial_species_id;
  bool is_measuring_lifetime = false;
  int64_t lifetime = -1;
  if (prm.initial_condition != "random") {
    initial_species_id = eco.species.at(0).strategy_id;
    is_measuring_lifetime = true;
  }

  MeasureElapsed("simulation");

  double c_level_avg = 0.0, fr_fraction = 0.0, efficient_fraction = 0.0, defensible_fraction = 0.0;
  double avg_diversity = 0.0, avg_mem_0 = 0.0, avg_mem_1 = 0.0;
  std::array<double,2> avg_automaton_sizes = {0.0, 0.0};
  std::vector<long> frequency(16, 0l);
  StrategySpace mem1space(1, 1);
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
      auto mem = eco.AverageMemLengths();
      avg_mem_0 += mem[0];
      avg_mem_1 += mem[1];
      auto a_sizes = eco.AverageAutomatonSizes();
      avg_automaton_sizes[0] += a_sizes[0];
      avg_automaton_sizes[1] += a_sizes[1];

      for (const auto& s: eco.species) {
        uint64_t lid = mem1space.ToLocalID(s.strategy_id);
        frequency[lid]++;
      }
      count++;
    }
    if (t % prm.T_print == prm.T_print - 1) {
      double m_inv = 1.0 / prm.M;
      auto mem = eco.AverageMemLengths();
      auto a_sizes = eco.AverageAutomatonSizes();
      tout << t + 1 << ' ' << eco.CooperationLevel() << ' ' << eco.NumFriendlyRival() * m_inv
                    << ' ' << eco.NumEfficient() * m_inv << ' ' << eco.NumDefensible() * m_inv
                    << ' ' << eco.Diversity() * m_inv
                    << ' ' << mem[0] << ' ' << mem[1]
                    << ' ' << a_sizes[0] << ' ' << a_sizes[1] << ' ';
      std::vector<long> v(16, 0);
      for (const auto& s: eco.species) {
        uint64_t lid = mem1space.ToLocalID(s.strategy_id);
        v[lid]++;
      }
      for (long x: v) { tout << (double)x/prm.M << ' '; }
      tout << std::endl;
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
    output["mem_length_self"] = avg_mem_0 / count;
    output["mem_length_opponent"] = avg_mem_1 / count;
    output["automaton_size_simple"] = avg_automaton_sizes[0] / count;
    output["automaton_size_full"] = avg_automaton_sizes[1] / count;
    output["lifetime_init_species"] = lifetime;
    std::ofstream fout("_output.json");
    fout << output.dump(2);
    fout.close();
  }

  {
    std::ofstream fout("frequency.dat");
    for (long x: frequency) {
      fout << (double)x / prm.M / count << ' ';
    }
    fout << std::endl;
  }

  {
    nlohmann::json j;
    for (const auto& s: eco.species) {
      j.emplace_back(s);
    }
    std::ofstream fout("final_state.json");
    fout << j.dump(2);
    fout.close();
  }

  MeasureElapsed("done");
  return 0;
}