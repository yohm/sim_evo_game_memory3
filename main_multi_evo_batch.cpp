//
// Created by Yohsuke Murase on 2021/10/25.
//

#include <iostream>
#include <vector>
#include <array>
#include <chrono>
#include <regex>
#include "omp.h"
#include "mpi.h"
#include "GroupedEvoGame.hpp"
#include "icecream-cpp/icecream.hpp"
#include "caravan-lib/caravan.hpp"


using nlohmann::json;

json RunSimulation(const GroupedEvoGame::Parameters& prm) {
  std::chrono::system_clock::time_point start = std::chrono::system_clock::now();

  GroupedEvoGame eco(prm);

  uint64_t initial_species_id;
  bool is_measuring_lifetime = false;
  int64_t lifetime = -1;
  if (prm.initial_condition != "random") {
    initial_species_id = eco.species.at(0).strategy_id;
    is_measuring_lifetime = true;
  }

  double c_level_avg = 0.0, fr_fraction = 0.0, efficient_fraction = 0.0, defensible_fraction = 0.0;
  double avg_diversity = 0.0, avg_mem_0 = 0.0, avg_mem_1 = 0.0;
  std::array<double,2> avg_automaton_sizes = {0.0, 0.0};
  size_t count = 0ul;

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
  }

  std::chrono::system_clock::time_point end = std::chrono::system_clock::now();
  auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();

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
  output["elapsed_time"] = elapsed;
  return output;
}

void ExpandInputsAndEnqueue(const json& inputs, caravan::Queue& q, std::mt19937_64& rnd) {
  for (const auto& kv: inputs.items()) {
    if (
      (kv.key() != "strategy_space" && kv.value().is_array()) ||
      (kv.key() == "strategy_space" && kv.value().at(0).is_array())
    ) {
      for (const auto& x: kv.value()) {
        json j = inputs;
        j[kv.key()] = x;
        ExpandInputsAndEnqueue(j, q, rnd);
      }
      return;
    }
  }

  int num_runs = inputs.at("number_of_runs").get<int>();
  for (int i = 0; i < num_runs; i++) {
    std::uniform_int_distribution<uint64_t> uni;
    uint64_t seed = uni(rnd);
    json j = inputs;
    j["_seed"] = seed;
    q.Push(j);

  }
}


int main(int argc, char *argv[]) {
  #if defined(NDEBUG)
  icecream::ic.disable();
  #endif
  icecream::ic.prefix("[", omp_get_thread_num, "/" , omp_get_max_threads, "]: ");

  MPI_Init(&argc, &argv);

  int my_rank, mpi_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

  Eigen::initParallel();
  if( argc != 2 ) {
    std::cerr << "Error : invalid argument" << std::endl;
    std::cerr << "  Usage : " << argv[0] << " <parameter_json_file>" << std::endl;
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  json inputs;
  if (my_rank == 0) {
    std::ifstream fin(argv[1]);
    if (!fin) {
      std::cerr << "file " << argv[1] << " is not found" << std::endl;
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
    fin >> inputs;
  }

  std::function<void(caravan::Queue&)> on_init = [&inputs](caravan::Queue& q) {
    std::mt19937_64 rnd( inputs.at("_seed").get<uint64_t>() );
    inputs.erase("_seed");
    ExpandInputsAndEnqueue(inputs, q, rnd);
  };

  std::function<void(int64_t, const json&, const json&, caravan::Queue&)> on_result_receive =
    [](int64_t task_id, const json& input, const json& output, caravan::Queue& q) {
    std::cout << task_id << "\n" << input << "\n" << output << "\n";
  };

  std::function<json(const json&)> do_task = [](const json& input) {
    GroupedEvoGame::Parameters prm = input.get<GroupedEvoGame::Parameters>();
    return RunSimulation(prm);
  };

  caravan::Start(on_init, on_result_receive, do_task, MPI_COMM_WORLD);

  MPI_Finalize();

  return 0;
}