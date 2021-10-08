//
// Created by Yohsuke Murase on 2020/06/04.
//

#include <iostream>
#include <vector>
#include <array>
#include <map>
#include <random>
#include <cassert>
#include <fstream>
#include <Eigen/Dense>
#include "StrategyM3.hpp"
#include <nlohmann/json.hpp>

// memory-1 species in strategy space discretized with `D`
// memory-1 strategy is characterized by tuple (p_{cc}, p_{cd}, p_{dc}, p_{dd}),
// where each of which denotes cooperation probability conditioned by the last Alice's action and the other players' defectors.
// Each of which can take discrete values [0,1/D,2/D,...,D/D].
// ID of a species is given by an integer with base-(D+1).
// ID can take values [0, (D+1)^4-1]
// ID=0 : AllD, ID=ID_MAX-1 : AllC
class Cprobs {
 public:
  Cprobs(size_t id, size_t DIS) {
    double dinv = 1.0 / DIS;
    const size_t B = DIS+1;
    cc = (id % B) * dinv;
    cd = ((id/B) % B) * dinv; // A: c, B: d
    dc = ((id/B/B) % B) * dinv; // A: d, B: c
    dd = ((id/B/B/B) % B) * dinv;
  }
  double cc, cd, dc, dd;
};

class Mem1Species {
 public:
  static size_t N_M1_Species(size_t DIS) { return (DIS+1)*(DIS+1)*(DIS+1)*(DIS+1); }
  Mem1Species(size_t _id, size_t _DIS) : id(_id), DIS(_DIS), prob(_id, _DIS) {
    assert(id <= N_M1_Species(DIS));
  }
  size_t id;
  size_t DIS;
  Cprobs prob;

  std::string ToString() const {
    const size_t B = DIS+1;
    size_t cc = (id % B);
    size_t cd = ((id/B) % B);
    size_t dc = ((id/B/B) % B);
    size_t dd = ((id/B/B/B) % B);
    std::ostringstream oss;
    oss << cc << '-' << cd << '-' << dc << '-' << dd << '/' << DIS;
    if (IsDefensible()) { oss << "_D"; }
    if (IsEfficient()) { oss << "_E"; }
    return oss.str();
  }

  std::vector<double> StationaryState(const Mem1Species& Bstr, double error = 0.0) const {
    Eigen::Matrix<double,4,4> A;

    // state 0: cc, state 1: cd (lower bit is A's history), ... 3: dd
    // calculate transition probability from j to i

    for (size_t j = 0; j < 4; j++) {
      Action last_a = (j & 1ul) ? D : C;
      Action last_b = ((j>>1ul) & 1ul) ? D : C;

      auto cooperation_prob = [](Action a, Action b, const Mem1Species& str) -> double {
        if (a == C) {
          if (b == C) { return str.prob.cc; }
          else { return str.prob.cd; }
        }
        else {
          if (b == C) { return str.prob.dc; }
          else { return str.prob.dd; }
        }
      };
      double c_A = cooperation_prob(last_a, last_b, *this);
      double c_B = cooperation_prob(last_b, last_a, Bstr);

      c_A = (1.0 - error) * c_A + error * (1.0 - c_A);
      c_B = (1.0 - error) * c_B + error * (1.0 - c_B);
      A(0,j) = c_A * c_B;
      A(1,j) = (1.0-c_A) * c_B;
      A(2,j) = c_A * (1.0-c_B);
      A(3,j) = (1.0-c_A) * (1.0-c_B);
    }

    for(int i=0; i<4; i++) {
      A(i,i) -= 1.0;
    }
    for(int i=0; i<4; i++) {
      A(4-1,i) += 1.0;  // normalization condition
    }
    Eigen::VectorXd b(4);
    for(int i=0; i<4; i++) { b(i) = 0.0;}
    b(4-1) = 1.0;
    Eigen::VectorXd x = A.colPivHouseholderQr().solve(b);
    std::vector<double> ans(4, 0.0);
    for(int i=0; i<ans.size(); i++) {
      ans[i] = x(i);
    }

    return ans;
  }

  double CooperationProb(StateM3 s) const {
    if (s.a_1 == C) {
      if (s.b_1 == C) { return prob.cc; }
      else { return prob.cd; }
    }
    else {
      if (s.b_1 == C) { return prob.dc; }
      else { return prob.dd; }
    }
  }

  // returns false for mixed strategy
  bool IsDefensible() const {
    double tolerance = 1.0e-8;
    if (prob.cc > tolerance && prob.cc < 1.0 - tolerance) { return false; }
    if (prob.cd > tolerance && prob.cd < 1.0 - tolerance) { return false; }
    if (prob.dc > tolerance && prob.dc < 1.0 - tolerance) { return false; }
    if (prob.dd > tolerance && prob.dd < 1.0 - tolerance) { return false; }

    // N = 2^{2}
    const size_t N = 4;
    typedef std::array<std::array<int, N>, N> d_matrix_t;
    d_matrix_t d;

    // construct adjacency matrix
    const int INF = N; // N is large enough since the path length is between -N/4 to N/4.
    for (size_t i = 0; i < N; i++) {
      for (size_t j = 0; j < N; j++) {
        d[i][j] = INF;
      }
    }

    for (size_t i = 0; i < N; i++) {
      Action a0 = (i & 1) ? D : C; // current state
      Action b0 = (i & 2) ? D : C;
      Action act_a = C;
      if (a0 == C) {
        if (b0 == C) { act_a = (prob.cc > 0.5 ? C : D); }
        else { act_a = (prob.cd > 0.5 ? C : D); }
      }
      else {
        if (b0 == C) { act_a = (prob.dc > 0.5 ? C : D); }
        else { act_a = (prob.dd > 0.5 ? C : D); }
      }

      // Get possible next states
      std::array<Action, 2> act_bs = {C, D};
      for (auto act_b: act_bs) {
        size_t j = 0;
        if (act_a == D) { j += 1; }
        if (act_b == D) { j += 2; }
        if (j == 0 || j == 3) d[i][j] = 0;
        else if (j == 1) d[i][j] = 1;
        else if (j == 2) d[i][j] = -1;
      }
      if (d[i][i] < 0) { return false; }
    }

    for (size_t k = 0; k < N; k++) {
      for (size_t i = 0; i < N; i++) {
        for (size_t j = 0; j < N; j++) {
          d[i][j] = std::min(d[i][j], d[i][k] + d[k][j]);
        }
        if (d[i][i] < 0) { return false; }
      }
    }
    return true;
  }

  bool IsEfficient() const {
    auto ss = StationaryState(*this, 0.0001);
    return (ss[0] > 0.99);
  }
};


class Species { // either Mem1Species or StrategyN3M5
 public:
  Species(size_t ID, size_t DIS) : m1(Mem1Species(0, DIS)), m5(0ull){
    const size_t N_M1 = Mem1Species::N_M1_Species(DIS);
    if (ID < N_M1) {
      is_m1 = true;
      m1 = Mem1Species(ID, DIS);
      m5 = StrategyM3(0ull);
      name = m1.ToString();
    }
    else if (ID == N_M1) {
      is_m1 = false;
      m1 = Mem1Species(0, DIS);
      m5 = StrategyM3::CAPRI2();
      name = "CAPRI2";
    }
    else if (ID == N_M1 + 1) {
      is_m1 = false;
      m1 = Mem1Species(0, DIS);
      m5 = StrategyM3::CAPRI();
      name = "CAPRI";
    }
    else if (ID == N_M1 + 2) {
      is_m1 = false;
      m1 = Mem1Species(0, DIS);
      m5 = StrategyM3::TFT_ATFT();
      name = "TFT_ATFT";
    }
    else if (ID == N_M1 + 3) {
      is_m1 = false;
      m1 = Mem1Species(0, DIS);
      m5 = StrategyM3::AON(2);
      name = "AON2";
    }
    else if (ID == N_M1 + 4) {
      is_m1 = false;
      m1 = Mem1Species(0, DIS);
      m5 = StrategyM3::AON(3);
      name = "AON3";
    }
    else {
      throw std::runtime_error("must not happen");
    }
  };
  bool is_m1;
  Mem1Species m1;
  StrategyM3 m5;
  std::string name;
  std::string ToString() const { return name; }
  std::vector<double> StationaryState(const Species &sb, double error) const {
    if (is_m1 && sb.is_m1) {
      return m1.StationaryState(sb.m1, error);
    }
    std::cerr << "calculating stationary state: " << name << ' ' << sb.name << std::endl;

    typedef Eigen::Triplet<double> T;
    std::vector<T> tripletVec;

    for (size_t j = 0; j < 64; j++) {
      // calculate transition probability from j to i
      const StateM3 sj(j);
      double c_a = _CooperationProb(sj);
      double c_b = sb._CooperationProb(sj.SwapAB());
      // cooperation probability taking noise into account
      c_a = (1.0 - error) * c_a + error * (1.0 - c_a);
      c_b = (1.0 - error) * c_b + error * (1.0 - c_b);

      for (size_t t = 0; t < 4; t++) {
        Action act_a = (t & 1ul) ? D : C;
        Action act_b = (t & 2ul) ? D : C;
        size_t i = sj.NextState(act_a, act_b).ID();
        double p_a = (act_a == C) ? c_a : (1.0-c_a);
        double p_b = (act_b == C) ? c_b : (1.0-c_b);
        tripletVec.emplace_back(i, j, p_a * p_b);
      }
    }

    const size_t S = 64;
    Eigen::SparseMatrix<double> A(S, S);
    A.setFromTriplets(tripletVec.cbegin(), tripletVec.cend());

    // subtract unit matrix & normalization condition
    std::vector<T> iVec;
    for (int i = 0; i < S-1; i++) { iVec.emplace_back(i, i, -1.0); }
    for (int i = 0; i < S-1; i++) { iVec.emplace_back(S-1, i, 1.0); }
    Eigen::SparseMatrix<double> I(S, S);
    I.setFromTriplets(iVec.cbegin(), iVec.cend());
    A = A + I;
    // std::cerr << "  transition matrix has been created" << std::endl;

    Eigen::VectorXd b = Eigen::VectorXd::Zero(S);
    b(S-1) = 1.0;

    Eigen::BiCGSTAB<Eigen::SparseMatrix<double>, Eigen::IncompleteLUT<double> > solver;
    solver.compute(A);
    Eigen::VectorXd x = solver.solve(b);

    // std::cerr << "#iterations:     " << solver.iterations() << std::endl;
    // std::cerr << "estimated error: " << solver.error() << std::endl;

    std::vector<double> ans(S, 0.0);
    for (int i = 0; i < S; i++) { ans[i] = x[i]; }
    return ans;
  }
 private:
  double _CooperationProb(const StateM3 &s) const {
    if (is_m1) { return m1.CooperationProb(s); }
    else { return m5.ActionAt(s) == C ? 1.0 : 0.0; }
  }

 public:
  static std::vector<Species> Memory1Species(size_t discrete_level) {
    std::vector<Species> ans;
    size_t n = Mem1Species::N_M1_Species(discrete_level);
    for (size_t i = 0; i < n; i++) {
      ans.emplace_back(i, discrete_level);
    }
    return std::move(ans);
  }
  static std::vector<Species> ReactiveMem1Species(size_t discrete_level) {
    std::vector<Species> ans;
    size_t B = discrete_level + 1;
    size_t n = Mem1Species::N_M1_Species(discrete_level);
    size_t B2 = B * B;
    for (size_t i = 0; i < n/B2; i++) {
      size_t id = i * B2 + i;
      ans.emplace_back(id, discrete_level);
    }
    return std::move(ans);
  }
};


class Ecosystem {
 public:
  Ecosystem(const std::vector<Species> &species_pool, double error) :pool(species_pool), N_SPECIES(species_pool.size()), e(error) {
    CalculateSSCache();
  };
  size_t N_SPECIES;
  std::vector<Species> pool;
  const double e;
  typedef std::vector<double> ss_cache_t;
  std::vector<std::vector<ss_cache_t> > ss_cache;
  // ss_cache[i][i] stores the stationary state when PG game is played by (i,i)
  // ss_cache[i][j] stores the stationary state when PG game is played by (i,j)

  void CalculateSSCache() {
    ss_cache.resize(N_SPECIES);
    for (size_t i = 0; i < N_SPECIES; i++) {
      ss_cache[i].resize(N_SPECIES);
    }

#pragma omp parallel for schedule(dynamic,1)
    for (size_t I=0; I < N_SPECIES * N_SPECIES; I++) {
      size_t i = I / N_SPECIES;
      size_t j = I % N_SPECIES;
      ss_cache[i][j] = pool[i].StationaryState(pool[j], e);
    }
  }

  // payoff of species i and j when the game is played by (i,j)
  std::array<double,2> PayoffVersus(size_t i, size_t j, double benefit, double cost) const {
    return Payoffs(ss_cache[i][j], benefit, cost);
  }

  std::array<double,2> Payoffs(const ss_cache_t &ss, double benefit, double cost) const {
    std::array<double, 2> ans = {0.0, 0.0};
    if (ss.size() == 4) {
      for (size_t i = 0; i < 4; i++) {
        double pa = 0.0, pb = 0.0;
        if ((i & 1ul) == 0) {
          pa -= cost;
          pb += benefit;
        }
        if ((i & 2ul) == 0) {
          pb -= cost;
          pa += benefit;
        }
        ans[0] += ss[i] * pa;
        ans[1] += ss[i] * pb;
      }
    }
    else {
      assert(ss.size() == 64);
      for (size_t i = 0; i < 64; i++) {
        StateM3 s(i);
        double pa = 0.0, pb = 0.0;
        if (s.a_1 == C) {
          pa -= cost;
          pb += benefit;
        }
        if (s.b_1 == C) {
          pb -= cost;
          pa += benefit;
        }
        ans[0] += ss[i] * pa;
        ans[1] += ss[i] * pb;
      }
    }
    return ans;
  }

  // calculate the equilibrium distribution exactly by linear algebra
  std::vector<double> CalculateEquilibrium(double benefit, double cost, uint64_t N, double sigma) const {
    Eigen::MatrixXd A(N_SPECIES, N_SPECIES);
    #pragma omp parallel for
    for (size_t ii = 0; ii < N_SPECIES * N_SPECIES; ii++) {
      size_t i = ii / N_SPECIES;
      size_t j = ii % N_SPECIES;
      if (i == j) { A(i, j) = 0.0; continue; }
      double p = FixationProb(benefit, cost, N, sigma, i, j);
      // std::cerr << "Fixation prob of mutant (mutant,resident): " << p << " (" << pool[i].ToString() << ", " << pool[j].ToString() << ")" << std::endl;
      A(i, j) = p * (1.0 / N_SPECIES);
    }

    for (size_t j = 0; j < N_SPECIES; j++) {
      double p_sum = 0.0;
      for (size_t i = 0; i < N_SPECIES; i++) {
        p_sum += A(i, j);
      }
      assert(p_sum <= 1.0);
      A(j, j) = 1.0 - p_sum; // probability that the state doesn't change
    }

    size_t n_row = A.rows();

    // subtract Ax = x => (A-I)x = 0
    for (size_t i = 0; i < A.rows(); i++) {
      A(i, i) -= 1.0;
    }
    // normalization condition
    for (size_t i = 0; i < A.rows(); i++) {
      A(A.rows()-1, i) += 1.0;
    }

    Eigen::VectorXd b(A.rows());
    for(int i=0; i<A.rows()-1; i++) { b(i) = 0.0;}
    b(A.rows()-1) = 1.0;
    Eigen::VectorXd x = A.householderQr().solve(b);
    std::vector<double> ans(A.rows());
    double prob_total = 0.0;
    for(int i=0; i<ans.size(); i++) {
      ans[i] = x(i);
      prob_total += x(i);
      assert(x(i) > -0.000001);
    }
    assert(std::abs(prob_total - 1.0) < 0.00001);
    return ans;
  }

  double FixationProb(double benefit, double cost, uint64_t N, double sigma, size_t mutant_idx, size_t resident_idx) const {
    // \frac{1}{\rho} = \sum_{i=0}^{N-1} \exp\left( \sigma \sum_{j=1}^{i} \left[(N-j-1)s_{yy} + js_{yx} - (N-j)s_{xy} - (j-1)s_{xx} \right] \right) \\
    //                = \sum_{i=0}^{N-1} \exp\left( \frac{\sigma i}{2} \left[(-i+2N-3)s_{yy} + (i+1)s_{yx} - (-i+2N-1)s_{xy} - (i-1)s_{xx} \right] \right)

    double s_xx = PayoffVersus(mutant_idx, mutant_idx, benefit, cost)[0];
    double s_yy = PayoffVersus(resident_idx, resident_idx, benefit, cost)[0];
    auto xy = PayoffVersus(mutant_idx, resident_idx, benefit, cost);
    double s_xy = xy[0];
    double s_yx = xy[1];

    double num_games = (N-1);
    s_xx /= num_games;
    s_yy /= num_games;
    s_xy /= num_games;
    s_yx /= num_games;
    double rho_inv = 0.0;
    for (int i=0; i < N; i++) {
      double x = sigma * i * 0.5 * (
          (2*N-3-i) * s_yy
          + (i+1) * s_yx
          - (2*N-1-i) * s_xy
          - (i-1) * s_xx
      );
      rho_inv += std::exp(x);
    }
    return 1.0 / rho_inv;
  }

  std::vector<std::string> SpeciesNames() const {
    std::vector<std::string> ans;
    for(auto s: pool) {
      ans.emplace_back( s.ToString() );
    }
    return std::move(ans);
  }
  double CooperationLevelSpecies(size_t i) const {
    const ss_cache_t &ss = ss_cache[i][i];
    if (ss.size() == 4) {
      double level = 0.0;
      for (size_t s = 0; s < 4; s++) {
        size_t num_c = 2 - std::bitset<2>(s).count();
        level += ss[s] * (num_c / 2.0);
      };
      return level;
    }
    else {
      double level = 0.0;
      for (size_t s = 0; s < 64; s++) {
        StateM3 state(s);
        size_t num_c = 2ul;
        if (state.a_1 == D) num_c -= 1;
        if (state.b_1 == D) num_c -= 1;
        level += ss[s] * (num_c / 2.0);
      }
      return level;
    }
  }
  double CooperationLevel(const std::vector<double> &eq_rate) const {
    assert(eq_rate.size() == N_SPECIES);
    double ans = 0.0;
    for (size_t i = 0; i < N_SPECIES; i++) {
      double c_lev = CooperationLevelSpecies(i);
      ans += eq_rate[i] * c_lev;
    }
    return ans;
  }
};


int main(int argc, char *argv[]) {
  Eigen::initParallel();
  if( argc != 2 ) {
    std::cerr << "Error : invalid argument" << std::endl;
    std::cerr << "  Usage : " << argv[0] << " <parameter_json_file>" << std::endl;
    return 1;
  }

  if (false)  // debugging
  {
    Species wsls(0b1001, 1);
    Species cccd(0b0111, 1);
    Species capri2(0b10000, 1);
    std::cerr << wsls.ToString() << std::endl;
    std::cerr << capri2.ToString() << std::endl;
    auto x = capri2.StationaryState(wsls, 0.0001);
    std::cerr << x[0] << std::endl;
    // std::vector<Species> pool = Species::Memory1Species(1);
    std::vector<Species> pool;
    pool.push_back(wsls);
    pool.push_back(capri2);
    pool.push_back(cccd);
    Ecosystem eco(pool, 0.0001);
    double p01 = eco.FixationProb(4.0, 1.0, 64, 1.0, 0, 1);
    double p10 = eco.FixationProb(4.0, 1.0, 64, 1.0, 1, 0);
    double p02 = eco.FixationProb(4.0, 1.0, 64, 1.0, 0, 2);
    double p20 = eco.FixationProb(4.0, 1.0, 64, 1.0, 2, 0);
    double p12 = eco.FixationProb(4.0, 1.0, 64, 1.0, 1, 2);
    double p21 = eco.FixationProb(4.0, 1.0, 64, 1.0, 2, 1);
    auto eq = eco.CalculateEquilibrium(4.0, 1.0, 64, 1.0);
    std::cerr << cccd.ToString() << std::endl;
    std::cerr << ' ' << p02 << std::endl;
  }

  double cost = 1.0;
  nlohmann::json input;
  {
    std::ifstream fin(argv[1]);
    fin >> input;
  }

  const uint64_t Nmax = input.at("Nmax").get<uint64_t>();
  const double sigma = input.at("sigma").get<double>();
  const double e = input.at("error_rate").get<double>();
  const uint64_t discrete_level = input.at("discrete_level").get<uint64_t>();
  const std::string strategy_space = input.at("strategy_space").get<std::string>();
  if (strategy_space != "full" && strategy_space != "reactive") { throw std::runtime_error("unknown key for strategy space"); }

  std::vector<Species> pool = (strategy_space == "full" ? Species::Memory1Species(discrete_level) : Species::ReactiveMem1Species(discrete_level));
  size_t N_M1 = Mem1Species::N_M1_Species(discrete_level);
  for (const nlohmann::json& _s: input.at("additional")) {
    const std::string s = _s.get<std::string>();
    if (s == "CAPRI2") {
      pool.emplace_back(N_M1+0, discrete_level);
    }
    else if (s == "CAPRI") {
      pool.emplace_back(N_M1+1, discrete_level);
    }
    else if (s == "TFT-ATFT") {
      pool.emplace_back(N_M1 + 2, discrete_level);
    }
    else if (s == "AON2") {
      pool.emplace_back(N_M1 + 3, discrete_level);
    }
    else if (s == "AON3") {
      pool.emplace_back(N_M1 + 4, discrete_level);
    }
    else {
      std::cerr << "[Error] unknown strategy " << s << " is given." << std::endl;
      throw std::runtime_error("unknown species");
    }
  }

  Ecosystem eco(pool, e);

  {
    std::ofstream namout("species.txt");
    for(auto name: eco.SpeciesNames()) {
      namout << name << std::endl;
    }
    namout << std::endl;
  }

  auto SweepOverBeta = [&eco,cost,sigma](size_t N)->std::vector<std::pair<double,double>> {
    char fname1[100];
    sprintf(fname1, "abundance_%zu.dat", N);
    std::ofstream eqout(fname1);
    std::vector<std::pair<double,double>> c_levels;
    for (int i = 5; i <= 300; i+=5) {
      double benefit = 1.0 + i / 100.0;
      auto eq = eco.CalculateEquilibrium(benefit, cost, N, sigma);
      eqout << benefit << ' ';
      for (double x: eq) { eqout << x << ' '; }
      eqout << std::endl;
      double c_lev = eco.CooperationLevel(eq);
      c_levels.push_back(std::make_pair(benefit, c_lev));
    }
    return c_levels;
  };

  std::vector<std::vector<std::pair<double,double>>> ans;
  for (int N = 2; N <= Nmax; N++) {
    auto a = SweepOverBeta(N);
    ans.push_back(a);
  }

  std::ofstream fout("cooperation_level.dat");
  for (size_t i = 0; i < ans[0].size(); i++) {
    fout << ans[0][i].first;
    for (size_t j = 0; j < ans.size(); j++) {
      fout << ' ' << ans[j][i].second;
    }
    fout << "\n";
  }
  fout.close();

  return 0;
}