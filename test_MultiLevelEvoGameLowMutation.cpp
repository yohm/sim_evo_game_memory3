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

bool IsClose(double x, double y) {
  return std::abs(x-y) < 1.0e-2;
}

MultiLevelEvoGameLowMutation::Parameters DefaultTestParameters() {
  MultiLevelEvoGameLowMutation::Parameters prm;
  prm.T_max = 1;
  prm.T_init = 0;
  prm.T_print = 1;
  prm.M = 30;
  prm.N = 2;
  prm.benefit = 1.5;
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
  std::cout << static_cast<nlohmann::json>(prm) << std::endl;
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

void test_FixationProb() {
  auto prm = DefaultTestParameters();
  prm.N = 2;
  prm.M = 30;
  prm.sigma = 10.0;
  prm.sigma_g = 10.0;
  MultiLevelEvoGameLowMutation eco(prm);

  MultiLevelEvoGameLowMutation::Species allc(StrategyM3::ALLC().ID(), prm.error_rate);
  MultiLevelEvoGameLowMutation::Species alld(StrategyM3::ALLD().ID(), prm.error_rate);
  MultiLevelEvoGameLowMutation::Species tft(StrategyM3::TFT().ID(), prm.error_rate);
  MultiLevelEvoGameLowMutation::Species wsls(StrategyM3::WSLS().ID(), prm.error_rate);
  myassert( IsClose(eco.FixationProb(alld, allc), 1.0) );
  myassert( IsClose( eco.FixationProb(allc, alld), 0.0) );
  myassert( IsClose( eco.FixationProb(tft, allc), 0.0) );
  myassert( IsClose( eco.FixationProb(allc, tft), 0.45)  );
  myassert( IsClose( eco.FixationProb(tft, wsls), 0.0) );
  myassert( IsClose( eco.FixationProb(wsls, tft), 0.45)  );
  myassert( IsClose( eco.FixationProb(wsls, allc), 1.0)  );
  myassert( IsClose( eco.FixationProb(allc, wsls), 0.0)  );
}

void test_IntraFixationProb() {
  auto prm = DefaultTestParameters();
  prm.N = 2;
  prm.M = 30;
  prm.sigma = 10.0;
  prm.sigma_g = 10.0;
  MultiLevelEvoGameLowMutation eco(prm);

  auto intra_fp = [&eco](const MultiLevelEvoGameLowMutation::Species& mut, const MultiLevelEvoGameLowMutation::Species& res) -> double {
    double benefit = eco.prm.benefit;
    double error = eco.prm.error_rate;
    double pi_mut_mut = (benefit - 1.0) * mut.cooperation_level;
    double pi_res_res = (benefit - 1.0) * res.cooperation_level;
    auto payoffs = StrategyM3(mut.strategy_id).Payoffs(StrategyM3(res.strategy_id), benefit, error);
    double pi_mut_res = payoffs[0], pi_res_mut = payoffs[1];
    return eco.IntraGroupFixationProb(pi_mut_mut, pi_mut_res, pi_res_mut, pi_res_res);
  };

  MultiLevelEvoGameLowMutation::Species allc(StrategyM3::ALLC().ID(), prm.error_rate);
  MultiLevelEvoGameLowMutation::Species alld(StrategyM3::ALLD().ID(), prm.error_rate);
  MultiLevelEvoGameLowMutation::Species tft(StrategyM3::TFT().ID(), prm.error_rate);
  MultiLevelEvoGameLowMutation::Species wsls(StrategyM3::WSLS().ID(), prm.error_rate);

  myassert( IsClose(intra_fp(allc, alld), 0.0) );
  myassert( IsClose(intra_fp(alld, allc), 1.0) );
  myassert( IsClose(intra_fp(tft, allc), 0.5) );
  myassert( IsClose(intra_fp(allc, tft), 0.5) );
  myassert( IsClose(intra_fp(alld, wsls), 1.0) );
  myassert( IsClose(intra_fp(wsls, tft), 0.5) );
}

StrategyM3 ParseStrategy(const std::string& str) {
  std::regex re_d(R"(\d+)"), re_c(R"([cd]{64})");
  if (std::regex_match(str, re_d)) {
    uint64_t id = std::stoull(str);
    return StrategyM3{id};
  }
  else if (std::regex_match(str, re_c)) {
    return StrategyM3{str.data()};
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
    std::string key(str);
    if (m.find(key) != m.end()) {
      return m.at(key);
    }
    else {
      std::cerr << "Error: unknown strategy " << key << std::endl;
      std::cerr << "  supported strategies are [";
      for (const auto& kv: m) {
        std::cerr << kv.first << ", ";
      }
      std::cerr << "]" << std::endl;
      std::runtime_error("unknown strategy");
    }
  }
  return StrategyM3{0ull};
}

void PrintFixationProbs(const StrategyM3& mutant, const StrategyM3& resident) {
  auto prm = DefaultTestParameters();
  MultiLevelEvoGameLowMutation eco(prm);
  MultiLevelEvoGameLowMutation::Species res_species(resident.ID(), prm.error_rate);
  MultiLevelEvoGameLowMutation::Species mut_species(mutant.ID(), prm.error_rate);
  std::cout << "fixation prob: " << eco.FixationProb(mut_species, res_species) << '\n';

  double benefit = eco.prm.benefit;
  double error = eco.prm.error_rate;
  double pi_mut_mut = (benefit - 1.0) * mut_species.cooperation_level;
  double pi_res_res = (benefit - 1.0) * res_species.cooperation_level;
  auto payoffs = mutant.Payoffs(resident, benefit, error);
  double pi_mut_res = payoffs[0], pi_res_mut = payoffs[1];
  double intra_fixation_prob = eco.IntraGroupFixationProb(pi_mut_mut, pi_mut_res, pi_res_mut, pi_res_res);
  std::cout << "intra-fixation prob: " << intra_fixation_prob << '\n';

  double migration_prob = 1.0 / (1.0 + std::exp( prm.sigma_g * (pi_res_res - pi_mut_mut) ) );
  std::cout << "migration prob: " << migration_prob << std::endl;
  std::cout << "payoffs: [" << pi_mut_mut << ", " << pi_res_res << "]\n";
}

int main(int argc, char* argv[]) {
  if (argc == 1) {
    std::cerr << "Testing MultiLevelEvoGameLowMutation class" << std::endl;
    test_FixationProb();
    test_IntraFixationProb();
  }
  else if (argc == 2) {
    StrategyM3 strategy = ParseStrategy(argv[1]);
    PrintFixationProbHisto(strategy.ID());
  }
  else if (argc == 3) {
    StrategyM3 mutant = ParseStrategy(argv[1]);
    StrategyM3 resident = ParseStrategy(argv[2]);
    std::cout << "mutant: " << argv[1] <<'\n'
              << "resident: " << argv[2] << '\n';
    PrintFixationProbs(mutant, resident);
  }
  else {
    throw std::runtime_error("invalid number of arguments");
  }

  return 0;
}