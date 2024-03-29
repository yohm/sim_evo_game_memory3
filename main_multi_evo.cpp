//
// Created by Yohsuke Murase on 2021/10/25.
//

#include <iostream>
#include <vector>
#include <array>
#include <chrono>
#include <regex>
#include "icecream.hpp"
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


int main(int argc, char *argv[]) {

  Eigen::initParallel();
  if( argc != 2 && argc != 3 ) {
    std::cerr << "Error : invalid argument" << std::endl;
    std::cerr << "  Usage : " << argv[0] << " <parameter_json_file>" << std::endl;
    std::cerr << "  Usage : " << argv[0] << " <parameter_json_file> <mutant_list>" << std::endl;
    return 1;
  }


  GroupedEvoGame::Parameters prm;
  {
    std::ifstream fin(argv[1]);
    nlohmann::json input;
    fin >> input;
    prm = input.get<GroupedEvoGame::Parameters>();
    prm.alld_mutant_prob = input.value("alld_mutant_prob", 0.0);
  }

  GroupedEvoGame::MutantList mutant_list;
  if (argc == 3) {
    mutant_list.LoadFromFile(argv[2]);
  }

  MeasureElapsed("initialize");
  GroupedEvoGame eco(prm, mutant_list);

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
  size_t count = 0ul;

  std::ofstream tout("timeseries.dat");
  std::ofstream tout_2("most_freq_species.dat");
  GroupedEvoGame::Species prev_most_frequent = eco.species[0];

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
      count++;
    }
    if (t % prm.T_print == prm.T_print - 1) {
      double m_inv = 1.0 / prm.M;
      auto mem = eco.AverageMemLengths();
      auto a_sizes = eco.AverageAutomatonSizes();
      tout << t + 1 << ' ' << eco.CooperationLevel() << ' ' << eco.NumFriendlyRival() * m_inv
                    << ' ' << eco.NumEfficient() * m_inv << ' ' << eco.NumDefensible() * m_inv
                    << ' ' << eco.NumWSLSLike() * m_inv
                    << ' ' << eco.Diversity()
                    << ' ' << mem[0] << ' ' << mem[1]
                    << ' ' << a_sizes[0] << ' ' << a_sizes[1]
                    << std::endl;
      // IC(t, eco.species);
      GroupedEvoGame::Species s = eco.MostFrequentSpecies();
      if (s.strategy_id != prev_most_frequent.strategy_id) {
        tout_2 << t+1 << "\n" << nlohmann::json(s) << std::endl;
        prev_most_frequent = s;
      }
    }
  }
  tout.close();
  tout_2.close();

  {
    std::vector<std::pair<uint64_t,size_t>> v;
    for (auto pair: eco.alld_killer_counter[0]) {
      v.emplace_back(pair);
    }
    std::sort(v.begin(), v.end(),
              [] (const auto &x, const auto &y) {return x.second > y.second;});
    std::ofstream fout("alld_killer_migration.dat");
    for (auto pair: v) {
      if (pair.second <= 1) break;
      fout << pair.first << ' ' << pair.second << std::endl;
    }
  }
  {
    std::vector<std::pair<uint64_t,size_t>> v;
    for (auto pair: eco.alld_killer_counter[1]) {
      v.emplace_back(pair);
    }
    std::sort(v.begin(), v.end(),
              [] (const auto &x, const auto &y) {return x.second > y.second;});
    std::ofstream fout("alld_killer_mutation.dat");
    for (auto pair: v) {
      if (pair.second <= 1) break;
      fout << pair.first << ' ' << pair.second << std::endl;
    }
  }

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
    auto comp = [](const GroupedEvoGame::Species& x, const GroupedEvoGame::Species& y) { return x.strategy_id < y.strategy_id; };
    std::map<GroupedEvoGame::Species,size_t,decltype(comp)> species_count(comp);
    for (const auto& s: eco.species) {
      if (species_count.find(s) == species_count.end()) {
        species_count[s] = 1;
      }
      else {
        species_count[s] += 1;
      }
    }
    using pair_t = std::pair<GroupedEvoGame::Species,size_t>;
    std::vector<pair_t> v;
    for (const auto& pair: species_count) { v.emplace_back(pair); }
    std::sort(v.begin(), v.end(), [](const pair_t& lhs, const pair_t& rhs) { return lhs.second > rhs.second; } );

    nlohmann::json j = v;
    std::ofstream fout("final_state.json");
    fout << j.dump(2);
    fout.close();
  }

  MeasureElapsed("done");
  return 0;
}
