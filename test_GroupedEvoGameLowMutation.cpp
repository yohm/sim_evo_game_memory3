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

bool IsClose(double x, double y) {
  return std::abs(x-y) < 1.0e-2;
}

GroupedEvoGame::Parameters DefaultTestParameters() {
  GroupedEvoGame::Parameters prm;
  prm.T_max = 1;
  prm.T_init = 0;
  prm.T_print = 1;
  prm.M = 30;
  prm.N = 2;
  prm.benefit = 3;
  prm.error_rate = 1.0e-3;
  prm.sigma_in = 10.0;
  prm.sigma_out = 10.0;
  prm.strategy_space = {3,3};
  prm.initial_condition = "random";
  prm.weighted_sampling = 1;
  prm.parallel_update = 0;
  prm.p_nu = 0.0;
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
  GroupedEvoGame eco(prm);

  GroupedEvoGame::Species resident(resident_id, prm.error_rate);

  std::map<double,int> fixation_prob_histo;
  double sum = 0.0;
  size_t COUNT = 1000;
  for (size_t i = 0; i < COUNT; i++) {
    uint64_t mut_id = eco.SampleStrategySpace();
    GroupedEvoGame::Species mut(mut_id, eco.prm.error_rate);
    double f = eco.FixationProbLowMutation(mut, resident);
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
  prm.sigma_in = 10.0;
  prm.sigma_out = 10.0;
  GroupedEvoGame eco(prm);

  GroupedEvoGame::Species allc(StrategyM3::ALLC().ID(), prm.error_rate);
  GroupedEvoGame::Species alld(StrategyM3::ALLD().ID(), prm.error_rate);
  GroupedEvoGame::Species tft(StrategyM3::TFT().ID(), prm.error_rate);
  GroupedEvoGame::Species wsls(StrategyM3::WSLS().ID(), prm.error_rate);
  myassert( IsClose(eco.FixationProbLowMutation(alld, allc), 1.0) );
  myassert( IsClose( eco.FixationProbLowMutation(allc, alld), 0.0) );
  myassert( IsClose( eco.FixationProbLowMutation(tft, allc), 0.0) );
  myassert( IsClose( eco.FixationProbLowMutation(allc, tft), 0.45)  );
  myassert( IsClose( eco.FixationProbLowMutation(tft, wsls), 0.0) );
  myassert( IsClose( eco.FixationProbLowMutation(wsls, tft), 0.45)  );
  myassert( IsClose( eco.FixationProbLowMutation(wsls, allc), 1.0)  );
  myassert( IsClose( eco.FixationProbLowMutation(allc, wsls), 0.0)  );
}

void test_IntraFixationProb() {
  auto prm = DefaultTestParameters();
  prm.N = 2;
  prm.M = 30;
  prm.sigma_in = 10.0;
  prm.sigma_out = 10.0;
  GroupedEvoGame eco(prm);

  GroupedEvoGame::Species allc(StrategyM3::ALLC().ID(), prm.error_rate);
  GroupedEvoGame::Species alld(StrategyM3::ALLD().ID(), prm.error_rate);
  GroupedEvoGame::Species tft(StrategyM3::TFT().ID(), prm.error_rate);
  GroupedEvoGame::Species wsls(StrategyM3::WSLS().ID(), prm.error_rate);

  myassert( IsClose(eco.IntraGroupFixationProb(allc, alld), 0.0) );
  myassert( IsClose(eco.IntraGroupFixationProb(alld, allc), 1.0) );
  myassert( IsClose(eco.IntraGroupFixationProb(tft, allc), 0.5) );
  myassert( IsClose(eco.IntraGroupFixationProb(allc, tft), 0.5) );
  myassert( IsClose(eco.IntraGroupFixationProb(alld, wsls), 1.0) );
  myassert( IsClose(eco.IntraGroupFixationProb(wsls, tft), 0.5) );
}

StrategyM3 ParseStrategy(const std::string& str) {
  std::regex re_d(R"(\d+)"), re_c(R"([cd]{64})"), re_m1(R"(m1-(\d+))");
  std::smatch m;
  if (std::regex_match(str, re_d)) {
    uint64_t id = std::stoull(str);
    return StrategyM3{id};
  }
  else if (std::regex_match(str, re_c)) {
    return StrategyM3{str.data()};
  }
  else if (std::regex_match(str, m, re_m1)) {
    uint64_t i = std::strtoull(m[1].str().data(), nullptr, 10ull);
    StrategySpace ss(1, 1);
    uint64_t gid = ss.ToGlobalID(i);
    return StrategyM3{gid};
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
    if (m.find(str) != m.end()) {
      return m.at(str);
    }
    else {
      std::cerr << "Error: unknown strategy " << str << std::endl;
      std::cerr << "  supported strategies are [";
      for (const auto& kv: m) {
        std::cerr << kv.first << ", ";
      }
      std::cerr << "]" << "\n" << "Or\n  'm1-[0-15]'" << std::endl;

      throw std::runtime_error("unknown strategy");
    }
  }
  return StrategyM3{0ull};
}

void PrintFixationProbs(const StrategyM3& mutant, const StrategyM3& resident) {
  auto prm = DefaultTestParameters();
  IC(prm);

  GroupedEvoGame eco(prm);
  GroupedEvoGame::Species res_species(resident.ID(), prm.error_rate);
  GroupedEvoGame::Species mut_species(mutant.ID(), prm.error_rate);
  std::cout << "fixation probs [psi_A, psi_B]: " << eco.FixationProbLowMutation(mut_species, res_species) << ", "
            << eco.FixationProbLowMutation(res_species, mut_species) << '\n';

  double intra_fixation_prob = eco.IntraGroupFixationProb(mut_species, res_species);
  double intra_fixation_prob_res = eco.IntraGroupFixationProb(res_species, mut_species);
  std::cout << "intra-fixation probs [rho_A, rho_B]: " << intra_fixation_prob << ", " << intra_fixation_prob_res << '\n';

  double migration_prob = eco.InterGroupImitationProb(mut_species, res_species);
  double migration_prob_res = eco.InterGroupImitationProb(res_species, mut_species);
  std::cout << "migration probs: " << migration_prob << ", " << migration_prob_res << std::endl;
  double q_plus = intra_fixation_prob * migration_prob;
  double q_minus = intra_fixation_prob_res * migration_prob_res;
  std::cout << "[Q+, Q-]: " << q_plus << ", " << q_minus << std::endl;

  double ut = eco.UnconditionalFixationTimeLowMutation(mut_species, res_species);
  double ct = eco.ConditionalFixationTimeLowMutation(mut_species, res_species);
  std::cout << "unconditional fixation time: " << ut << "\n" << "conditional fixation time: " << ct << "\n";

  std::cout << "payoffs: [" << (prm.benefit-1.0)*mut_species.cooperation_level << ", " << (prm.benefit-1.0)*res_species.cooperation_level << "]\n";
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