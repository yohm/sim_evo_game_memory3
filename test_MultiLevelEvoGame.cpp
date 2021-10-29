#include <iostream>
#include <regex>
#include <cassert>
#include "MultiLevelEvoGame.hpp"

#define myassert(x) do {                              \
if (!(x)) {                                           \
printf("Assertion failed: %s, file %s, line %d\n"   \
, #x, __FILE__, __LINE__);                   \
exit(1);                                            \
}                                                   \
} while (0)

MultiLevelEvoGame::Parameters DefaultTestParameters() {
  MultiLevelEvoGame::Parameters prm;
  prm.T_max = 1;
  prm.T_init = 0;
  prm.T_print = 1;
  prm.M = 2;
  prm.N = 2;
  prm.benefit = 2.0;
  prm.error_rate = 1.0e-3;
  prm.sigma = 10.0;
  prm.sigma_g = 10.0;
  prm.p_intra = 0.9;
  prm.strategy_space = {3,3};
  prm.initial_condition = "random";
  prm._seed = 1234567890ull;
  return prm;
}

void test_IntraGroupSelection() {
  auto prm = DefaultTestParameters();
  MultiLevelEvoGame eco(prm);

  MultiLevelEvoGame::Species allc(StrategyM3::ALLC().ID(), prm.error_rate);
  MultiLevelEvoGame::Species alld(StrategyM3::ALLD().ID(), prm.error_rate);
  MultiLevelEvoGame::Species capri(StrategyM3::CAPRI().ID(), prm.error_rate);
  MultiLevelEvoGame::Species aon3(StrategyM3::AON(3).ID(), prm.error_rate);
  MultiLevelEvoGame::Species wsls(StrategyM3::WSLS().ID(), prm.error_rate);

  IC( allc, alld, capri, aon3, wsls );
  IC( eco.FixationProb(allc, alld) );
  IC( eco.FixationProb(alld, allc) );
  IC( eco.FixationProb(alld, aon3) );
  IC( eco.FixationProb(alld, capri) );
  IC( eco.FixationProb(capri, aon3) );
  IC( eco.FixationProb(aon3, capri) );
}

void test_InterGroupSelection() {
  auto prm = DefaultTestParameters();
  MultiLevelEvoGame eco(prm);

  MultiLevelEvoGame::Species allc(StrategyM3::ALLC().ID(), prm.error_rate);
  MultiLevelEvoGame::Species alld(StrategyM3::ALLD().ID(), prm.error_rate);
  MultiLevelEvoGame::Species capri(StrategyM3::CAPRI().ID(), prm.error_rate);
  MultiLevelEvoGame::Species aon3(StrategyM3::AON(3).ID(), prm.error_rate);
  MultiLevelEvoGame::Species wsls(StrategyM3::WSLS().ID(), prm.error_rate);

  IC( eco.SelectionProb(allc, capri) );
  IC( eco.SelectionProb(capri, allc) );
  IC( eco.SelectionProb(alld, aon3) );
  IC( eco.SelectionProb(alld, capri) );
  IC( eco.SelectionProb(aon3, capri) );
  IC( eco.SelectionProb(capri, aon3) );
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
  MultiLevelEvoGame eco(prm);

  MultiLevelEvoGame::Species aon3(StrategyM3::AON(3).ID(), prm.error_rate);
  MultiLevelEvoGame::Species capri(StrategyM3::CAPRI().ID(), prm.error_rate);

  auto histo_fixation_prob = [&eco](const MultiLevelEvoGame::Species& resident) {
    std::map<double,int> histo;
    double avg = 0.0;
    for (size_t i = 0; i < 1000; i++) {
      uint64_t mut_id = eco.WeightedSampleStrategySpace();
      // uint64_t mut_id = eco.UniformSampleStrategySpace();
      MultiLevelEvoGame::Species mut(mut_id, eco.prm.error_rate);
      double f = eco.FixationProb(mut, resident);
      avg += f;
      double key = std::round(f * 10.0) / 10.0;
      histo[key] = GetWithDef(histo, key, 0) + 1;
    }
    return std::make_pair(histo, avg/1000);
  };

  IC( histo_fixation_prob(aon3) );
  IC( histo_fixation_prob(capri) );

}

int main(int argc, char* argv[]) {
  if (argc == 1) {
    std::cerr << "Testing MultiLevelEvoGame class" << std::endl;

    test_IntraGroupSelection();
    test_InterGroupSelection();
    test_AON3();
  }

  return 0;
}