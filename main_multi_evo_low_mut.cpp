//
// Created by Yohsuke Murase on 2021/10/25.
//

#include <iostream>
#include <vector>
#include <array>
#include <chrono>
#include <regex>
#include "MultiLevelEvoGameLowMutation.hpp"
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

  Eigen::initParallel();
  if( argc != 2 ) {
    std::cerr << "Error : invalid argument" << std::endl;
    std::cerr << "  Usage : " << argv[0] << " <parameter_json_file>" << std::endl;
    return 1;
  }

  MultiLevelEvoGameLowMutation::Parameters prm;
  {
    std::ifstream fin(argv[1]);
    nlohmann::json input;
    fin >> input;
    prm = input.get<MultiLevelEvoGameLowMutation::Parameters>();
  }

  MeasureElapsed("initialize");

  MultiLevelEvoGameLowMutation eco(prm);

  uint64_t initial_species_id;
  bool is_measuring_lifetime = false;
  int64_t lifetime = -1;
  if (prm.initial_condition != "random") {
    initial_species_id = eco.current_species.strategy_id;
    is_measuring_lifetime = true;
  }

  MeasureElapsed("simulation");

  double c_level_avg = 0.0;
  size_t fr_count = 0, efficient_count = 0, defensible_count = 0;
  double avg_mem_0 = 0.0, avg_mem_1 = 0.0, avg_mem_diff = 0.0;
  std::array<double,2> avg_automaton_sizes = {0.0, 0.0};
  size_t count = 0ul;

  std::ofstream tout("timeseries.dat");

  for (size_t t = 0; t < prm.T_max; t++) {
    eco.Update();
    if (is_measuring_lifetime) {
      if (eco.current_species.strategy_id != initial_species_id) {
        lifetime = t;
        is_measuring_lifetime = false;
      }
      else {
        lifetime = t+1;
      }
    }
    if (t > prm.T_init) {
      c_level_avg += eco.current_species.cooperation_level;
      if (eco.current_species.is_defensible && eco.current_species.is_efficient) {
        fr_count++;
      }
      else if (eco.current_species.is_defensible) {
        defensible_count++;
      }
      else if (eco.current_species.is_efficient) {
        efficient_count++;
      }
      const auto& mem = eco.current_species.mem_lengths;
      avg_mem_0 += mem[0];
      avg_mem_1 += mem[1];
      avg_mem_diff += (mem[0] - mem[1]);
      avg_automaton_sizes[0] += eco.current_species.automaton_sizes[0];
      avg_automaton_sizes[1] += eco.current_species.automaton_sizes[1];
      count++;
    }
    if (t % prm.T_print == prm.T_print - 1) {
      double m_inv = 1.0 / prm.M;
      const auto& mem = eco.current_species.mem_lengths;
      const auto& a_sizes = eco.current_species.automaton_sizes;
      std::string label = "O";
      if (eco.current_species.is_defensible && eco.current_species.is_efficient) { label = "FR"; }
      else if (eco.current_species.is_defensible) { label = "D"; }
      else if (eco.current_species.is_efficient) { label = "E"; }
      tout << t + 1 << ' ' << eco.current_species.cooperation_level
                    << ' ' << label
                    << ' ' << eco.current_species.mem_lengths[0] << ' ' << eco.current_species.mem_lengths[1]
                    << ' ' << eco.current_species.automaton_sizes[0] << ' ' << eco.current_species.automaton_sizes[1]
                    << std::endl;
      // IC(t, eco.species);
    }
  }
  tout.close();

  {
    nlohmann::json output;
    double count_inv = 1.0 / (double)count;
    output["cooperation_level"] = c_level_avg * count_inv;
    output["friendly_rival_fraction"] = fr_count * count_inv;
    output["efficient_fraction"] = efficient_count * count_inv;
    output["defensible_fraction"] = defensible_count * count_inv;
    output["mem_length_self"] = avg_mem_0 * count_inv;
    output["mem_length_opponent"] = avg_mem_1 * count_inv;
    output["mem_length_diff"] = avg_mem_diff * count_inv;
    output["automaton_size_simple"] = avg_automaton_sizes[0] * count_inv;
    output["automaton_size_full"] = avg_automaton_sizes[1] * count_inv;
    output["lifetime_init_species"] = lifetime;
    std::ofstream fout("_output.json");
    fout << output.dump(2);
    fout.close();
  }

  {
    nlohmann::json j = eco.current_species;
    std::ofstream fout("final_state.json");
    fout << j.dump(2);
    fout.close();
  }

  MeasureElapsed("done");
  return 0;
}