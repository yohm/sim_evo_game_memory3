#include <iostream>
#include <regex>
#include <cassert>
#include "MultiLevelEvoGameLowMutation.hpp"

#define myassert(x) do {                              \
if (!(x)) {                                           \
printf("Assertion failed: %s, file %s, line %d\n"   \
, #x, __FILE__, __LINE__);                   \
exit(1);                                            \
}                                                   \
} while (0)

MultiLevelEvoGameLowMutation::Parameters DefaultTestParameters() {
  MultiLevelEvoGameLowMutation::Parameters prm;
  prm.T_max = 1;
  prm.T_init = 0;
  prm.T_print = 1;
  prm.M = 50;
  prm.N = 2;
  prm.benefit = 2.0;
  prm.error_rate = 1.0e-3;
  prm.sigma = 10.0;
  prm.sigma_g = 10.0;
  prm.strategy_space = {3,3};
  prm.initial_condition = "random";
  prm._seed = 1234567890ull;
  return prm;
}

template <template<class,class,class...> class C, typename K, typename V, typename... Args>
V GetWithDef(const C<K,V,Args...>& m, K const& key, const V & defval) {
  typename C<K,V,Args...>::const_iterator it = m.find( key );
  if (it == m.end()) return defval;
  return it->second;
}

void PrintFixationProbHisto(uint64_t resident_id) {
  auto prm = DefaultTestParameters();
  MultiLevelEvoGameLowMutation eco(prm);

  MultiLevelEvoGameLowMutation::Species resident(resident_id, prm.error_rate);

  std::map<double,int> fixation_prob_histo;
  double sum = 0.0;
  size_t COUNT = 1000;
  for (size_t i = 0; i < COUNT; i++) {
    uint64_t mut_id = eco.WeightedSampleStrategySpace();
    // uint64_t mut_id = eco.UniformSampleStrategySpace();
    MultiLevelEvoGameLowMutation::Species mut(mut_id, eco.prm.error_rate);
    double f = eco.FixationProb(mut, resident);
    sum += f;
    double key = std::round(f * 10.0) / 10.0;
    fixation_prob_histo[key] = GetWithDef(fixation_prob_histo, key, 0) + 1;
  }
  double fixation_prob = sum / (double)COUNT;
  IC(fixation_prob_histo, fixation_prob);
}

int main(int argc, char* argv[]) {
  if (argc == 1) {
    std::cerr << "Testing MultiLevelEvoGameLowMutation class" << std::endl;
  }
  else if (argc == 2) {
    std::regex re_d(R"(\d+)"), re_c(R"([cd]{64})");
    if (std::regex_match(argv[1], re_d)) {
      uint64_t id = std::stoull(argv[1]);
      PrintFixationProbHisto(id);
    }
    else if (std::regex_match(argv[1], re_c)) {
      StrategyM3 str(argv[1]);
      PrintFixationProbHisto(str.ID());
    }
    else {
      std::map<std::string,StrategyM3> m = {
        {"ALLC", StrategyM3::ALLC()},
        {"ALLD", StrategyM3::ALLD()},
        {"TFT", StrategyM3::TFT()},
        {"WSLS", StrategyM3::WSLS()},
        {"TF2T", StrategyM3::TF2T()},
        {"TFT-ATFT", StrategyM3::TFT_ATFT()},
        {"CAPRI", StrategyM3::CAPRI()},
        {"CAPRI2", StrategyM3::CAPRI2()},
        {"AON2", StrategyM3::AON(2)},
        {"AON3", StrategyM3::AON(3)},
        };
      std::string key(argv[1]);
      if (m.find(key) != m.end()) {
        PrintFixationProbHisto(m.at(key).ID());
      }
      else {
        std::cerr << "Error: unknown strategy " << key << std::endl;
        std::cerr << "  supported strategies are [";
        for (const auto& kv: m) {
          std::cerr << kv.first << ", ";
        }
        std::cerr << "]" << std::endl;
        return 1;
      }
    }
  }
  else {
    throw std::runtime_error("invalid number of arguments");
  }

  return 0;
}