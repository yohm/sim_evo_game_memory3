#include <iostream>
#include <regex>
#include <cassert>
#include "GroupedEvoGame.hpp"

#define myassert(x) do {                              \
if (!(x)) {                                           \
printf("Assertion failed: %s, file %s, line %d\n"   \
, #x, __FILE__, __LINE__);                   \
exit(1);                                            \
}                                                   \
} while (0)

GroupedEvoGame::Parameters DefaultTestParameters() {
  GroupedEvoGame::Parameters prm;
  prm.T_max = 1;
  prm.T_init = 0;
  prm.T_print = 1;
  prm.M = 100;
  prm.N = 3;
  prm.benefit = 2.0;
  prm.error_rate = 1.0e-3;
  prm.sigma_in = 10.0;
  prm.sigma_out = 10.0;
  prm.p_nu = 0.5;
  prm.strategy_space = {3,3};
  prm.initial_condition = "random";
  prm.weighted_sampling = 1;
  prm._seed = 1234567890ull;
  return prm;
}

void test_IntraGroupSelection() {
  auto prm = DefaultTestParameters();
  GroupedEvoGame eco(prm);

  GroupedEvoGame::Species allc(StrategyM3::ALLC().ID(), prm.error_rate);
  GroupedEvoGame::Species alld(StrategyM3::ALLD().ID(), prm.error_rate);
  GroupedEvoGame::Species capri(StrategyM3::CAPRI().ID(), prm.error_rate);
  GroupedEvoGame::Species aon3(StrategyM3::AON(3).ID(), prm.error_rate);
  GroupedEvoGame::Species wsls(StrategyM3::WSLS().ID(), prm.error_rate);

  IC( allc, alld, capri, aon3, wsls );
  IC(eco.IntraGroupFixationProb(allc, alld) );
  IC(eco.IntraGroupFixationProb(alld, allc) );
  IC(eco.IntraGroupFixationProb(alld, aon3) );
  IC(eco.IntraGroupFixationProb(alld, capri) );
  IC(eco.IntraGroupFixationProb(capri, aon3) );
  IC(eco.IntraGroupFixationProb(aon3, capri) );
}

void test_InterGroupSelection() {
  auto prm = DefaultTestParameters();
  GroupedEvoGame eco(prm);

  GroupedEvoGame::Species allc(StrategyM3::ALLC().ID(), prm.error_rate);
  GroupedEvoGame::Species alld(StrategyM3::ALLD().ID(), prm.error_rate);
  GroupedEvoGame::Species capri(StrategyM3::CAPRI().ID(), prm.error_rate);
  GroupedEvoGame::Species aon3(StrategyM3::AON(3).ID(), prm.error_rate);
  GroupedEvoGame::Species wsls(StrategyM3::WSLS().ID(), prm.error_rate);

  IC(eco.InterGroupImitationProb(allc, capri) );
  IC(eco.InterGroupImitationProb(capri, allc) );
  IC(eco.InterGroupImitationProb(alld, aon3) );
  IC(eco.InterGroupImitationProb(alld, capri) );
  IC(eco.InterGroupImitationProb(aon3, capri) );
  IC(eco.InterGroupImitationProb(capri, aon3) );
}

template <template<class,class,class...> class C, typename K, typename V, typename... Args>
V GetWithDef(const C<K,V,Args...>& m, K const& key, const V & defval) {
  typename C<K,V,Args...>::const_iterator it = m.find( key );
  if (it == m.end()) return defval;
  return it->second;
}

void test_AON3() {
  auto prm = DefaultTestParameters();
  prm.N = 3;
  prm.benefit = 1.2;
  prm.strategy_space = {2, 2};
  GroupedEvoGame eco(prm);

  GroupedEvoGame::Species aon3(StrategyM3::AON(3).ID(), prm.error_rate);
  GroupedEvoGame::Species capri(StrategyM3::CAPRI().ID(), prm.error_rate);

  auto histo_fixation_prob = [&eco](const GroupedEvoGame::Species& resident) {
    std::map<double,int> histo;
    double avg = 0.0;
    for (size_t i = 0; i < 1000; i++) {
      uint64_t mut_id = eco.SampleStrategySpace();
      // uint64_t mut_id = eco.UniformSampleStrategySpace();
      GroupedEvoGame::Species mut(mut_id, eco.prm.error_rate);
      double f = eco.IntraGroupFixationProb(mut, resident);
      avg += f;
      double key = std::round(f * 10.0) / 10.0;
      histo[key] = GetWithDef(histo, key, 0) + 1;
    }
    return std::make_pair(histo, avg/1000);
  };

  IC( histo_fixation_prob(aon3) );
  IC( histo_fixation_prob(capri) );
}

void PrintFixationProbs(uint64_t resident_id) {
  auto prm = DefaultTestParameters();
  prm.N = 3;
  prm.benefit = 2;
  GroupedEvoGame eco(prm);

  GroupedEvoGame::Species resident(resident_id, prm.error_rate);

  std::map<double,int> fixation_prob_histo;
  double sum = 0.0;
  size_t N = 1000;
  for (size_t i = 0; i < N; i++) {
    uint64_t mut_id = eco.SampleStrategySpace();
    GroupedEvoGame::Species mut(mut_id, eco.prm.error_rate);
    double f = eco.IntraGroupFixationProb(mut, resident);
    sum += f;
    double key = std::round(f * 10.0) / 10.0;
    fixation_prob_histo[key] = GetWithDef(fixation_prob_histo, key, 0) + 1;
  }
  double fixation_prob = sum / N;
  IC(fixation_prob_histo, fixation_prob);
}

int main(int argc, char* argv[]) {
  if (argc == 1) {
    std::cerr << "Testing GroupedEvoGame class" << std::endl;

    test_IntraGroupSelection();
    test_InterGroupSelection();
    test_AON3();
  }
  else if (argc == 2) {
    std::regex re_d(R"(\d+)"), re_c(R"([cd]{64})");
    if (std::regex_match(argv[1], re_d)) {
      uint64_t id = std::stoull(argv[1]);
      PrintFixationProbs(id);
    }
    else if (std::regex_match(argv[1], re_c)) {
      StrategyM3 str(argv[1]);
      PrintFixationProbs(str.ID());
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
        PrintFixationProbs(m.at(key).ID());
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