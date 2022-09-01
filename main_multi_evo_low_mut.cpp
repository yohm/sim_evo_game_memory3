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

void Measure(const GroupedEvoGame::Species& current_species, Counter& counter) {
  counter.cooperation_level += current_species.cooperation_level;
  if (current_species.is_defensible && current_species.is_efficient) {
    counter.friendly_rival++;
  }
  else if (current_species.is_defensible) {
    counter.defensible++;
  }
  else if (current_species.is_efficient) {
    counter.efficient++;
  }
  const auto& mem = current_species.mem_lengths;
  counter.mem_lengths[0] += mem[0];
  counter.mem_lengths[1] += mem[1];
  counter.automaton_sizes[0] += current_species.automaton_sizes[0];
  counter.automaton_sizes[1] += current_species.automaton_sizes[1];
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

  GroupedEvoGame::Parameters prm;
  {
    std::ifstream fin(argv[1]);
    nlohmann::json input;
    fin >> input;
    // populate dummy data for unused parameters
    input["p_nu"] = 0.0;
    input["parallel_update"] = 0;
    prm = input.get<GroupedEvoGame::Parameters>();
  }

  MeasureElapsed("initialize");

  GroupedEvoGame eco(prm);
  GroupedEvoGame::Species current_species = eco.species[0];

  uint64_t initial_species_id;
  bool is_measuring_lifetime = false;
  int64_t lifetime = -1;
  if (prm.initial_condition != "random") {
    initial_species_id = current_species.strategy_id;
    is_measuring_lifetime = true;
  }

  MeasureElapsed("simulation");

  Counter counter_all, counter_interval;

  std::ofstream tout("timeseries.dat");
  std::uniform_real_distribution<double> uni;

  for (size_t t = 0; t < prm.T_max; t++) {
    uint64_t mut_id = eco.SampleStrategySpaceWithExclusion();
    GroupedEvoGame::Species mut(mut_id, prm.error_rate);
    double prob = eco.FixationProbLowMutation(mut, current_species);
    if (uni(eco.a_rnd[0]) < prob) {
      current_species = mut;
    }

    Measure(current_species, counter_interval);
    if (t > prm.T_init) {
      Measure(current_species, counter_all);
    }
    if (is_measuring_lifetime) {
      if (current_species.strategy_id != initial_species_id) {
        lifetime = t;
        is_measuring_lifetime = false;
      }
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
    nlohmann::json j = current_species;
    std::ofstream fout("final_state.json");
    fout << j.dump(2);
    fout.close();
  }

  MeasureElapsed("done");
  return 0;
}