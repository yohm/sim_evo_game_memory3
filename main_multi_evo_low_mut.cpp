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

class Counter {
  public:
  Counter() { Reset();}
  double cooperation_level;
  long defensible;
  long efficient;
  long friendly_rival;
  std::array<long,2> mem_lengths;
  std::array<long,2> automaton_sizes;
  long count;
  void Reset() {
    cooperation_level = 0.0;
    defensible = 0;
    efficient = 0;
    friendly_rival = 0;
    mem_lengths[0] = 0;
    mem_lengths[1] = 0;
    automaton_sizes[0] = 0;
    automaton_sizes[1] = 0;
    count = 0;
  }
};

void Measure(const MultiLevelEvoGameLowMutation& eco, Counter& counter) {
  counter.cooperation_level += eco.current_species.cooperation_level;
  if (eco.current_species.is_defensible && eco.current_species.is_efficient) {
    counter.friendly_rival++;
  }
  else if (eco.current_species.is_defensible) {
    counter.defensible++;
  }
  else if (eco.current_species.is_efficient) {
    counter.efficient++;
  }
  const auto& mem = eco.current_species.mem_lengths;
  counter.mem_lengths[0] += mem[0];
  counter.mem_lengths[1] += mem[1];
  counter.automaton_sizes[0] += eco.current_species.automaton_sizes[0];
  counter.automaton_sizes[1] += eco.current_species.automaton_sizes[1];
  counter.count++;
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

  Counter counter_all, counter_interval;

  std::ofstream tout("timeseries.dat");

  for (size_t t = 0; t < prm.T_max; t++) {
    eco.Update();
    Measure(eco, counter_interval);
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
      Measure(eco, counter_all);
    }
    if (t % prm.T_print == prm.T_print - 1) {
      double c_inv = 1.0 / static_cast<double>(counter_interval.count);
      tout << t + 1
        << ' ' << counter_interval.cooperation_level * c_inv
        << ' ' << counter_interval.efficient * c_inv
        << ' ' << counter_interval.defensible * c_inv
        << ' ' << counter_interval.friendly_rival * c_inv
        << ' ' << counter_interval.mem_lengths[0] * c_inv << ' ' << counter_interval.mem_lengths[1] * c_inv
        << ' ' << counter_interval.automaton_sizes[0] * c_inv << ' ' << counter_interval.automaton_sizes[1] * c_inv
        << std::endl;
      counter_interval.Reset();
    }
  }
  tout.close();

  {
    nlohmann::json output;
    double count_inv = 1.0 / (double)counter_all.count;
    output["cooperation_level"] = counter_all.cooperation_level * count_inv;
    output["friendly_rival_fraction"] = counter_all.friendly_rival * count_inv;
    output["efficient_fraction"] = counter_all.efficient * count_inv;
    output["defensible_fraction"] = counter_all.defensible * count_inv;
    output["mem_length_self"] = counter_all.mem_lengths[0] * count_inv;
    output["mem_length_opponent"] = counter_all.mem_lengths[1] * count_inv;
    output["automaton_size_simple"] = counter_all.automaton_sizes[0] * count_inv;
    output["automaton_size_full"] = counter_all.automaton_sizes[1] * count_inv;
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