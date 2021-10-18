//
// Created by Yohsuke Murase on 2021/10/10.
//

#include <iostream>
#include <vector>
#include <array>
#include <map>
#include <random>
#include <cassert>
#include <fstream>
#include <chrono>
#include <Eigen/Dense>
#include <nlohmann/json.hpp>
#include <mpi.h>
#include "StrategyM3.hpp"
#include "StrategySpace.hpp"
#include "Scalapack.hpp"


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


class EvolutionaryGame {
 public:
  EvolutionaryGame(const StrategySpace& _space, double error) : space(_space), N_SPECIES(_space.Size()), e(error) {
  };
  const StrategySpace space;
  const size_t N_SPECIES;
  const double e;
  using ss_t = std::array<double,2>;  // probability of getting benefit & paying cost
  mutable std::map<std::pair<size_t,size_t>, ss_t> ss_cache;
  // ss_cache[(i,j)] stores the stationary state when PG game is played by (i,j)

  ss_t GetSS(size_t i, size_t j) const {
    auto key = (i < j) ? std::make_pair(i, j) : std::make_pair(j, i);
    auto found = ss_cache.find(key);
    ss_t ss;
    if (found != ss_cache.end()) {
      ss = found->second;
    }
    else {
      ss = CalculateSS(i, j);
      ss_cache[key] = ss;
    }
    return (i < j) ? ss : ss_t({ss[1], ss[0]});
  }

  ss_t CalculateSS(size_t i, size_t j) const {
    StrategyM3 si(space.ToGlobalID(i) );
    StrategyM3 sj(space.ToGlobalID(j) );
    auto p = si.StationaryState(e, &sj);
    ss_t ans = {0.0, 0.0};
    for (size_t n = 0; n < 64; n++) {
      StateM3 s(n);
      if (i > j) {
        if (s.a_1 == C) {
          ans[1] += p[n];
        }
        if (s.b_1 == C) {
          ans[0] += p[n];
        }
      }
      else if (i == j) {
        if (s.a_1 == C) {
          ans[1] += p[n];
        }
        if (s.b_1 == C) {
          ans[0] += p[n];
        }
      }
    }
    return ans;
  }

  // payoff of species i and j when the game is played by (i,j)
  std::array<double,2> PayoffVersus(size_t i, size_t j, double benefit, double cost) const {
    ss_t ss_ij = GetSS(i, j);
    ss_t ss_ji = {ss_ij[1], ss_ij[0]};
    return {
      ss_ij[0] * benefit - ss_ij[1] * cost,
      ss_ji[0] * benefit - ss_ji[1] * cost
    };
  }

  // calculate the equilibrium distribution exactly by linear algebra
  // [TODO] IMPLEMENT ME WITH MPI
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

    auto num_games = static_cast<double>(N-1);
    s_xx /= num_games;
    s_yy /= num_games;
    s_xy /= num_games;
    s_yx /= num_games;
    double rho_inv = 0.0;
    for (int i=0; i < N; i++) {
      double x = sigma * i * 0.5 * (
            (double)(2*N-3-i) * s_yy
          + (double)(i+1) * s_yx
          - (double)(2*N-1-i) * s_xy
          - (double)(i-1) * s_xx
      );
      rho_inv += std::exp(x);
    }
    return 1.0 / rho_inv;
  }

  double CooperationLevelSpecies(size_t i) const {
    return GetSS(i, i)[0];
  }
  // [TODO] IMPLEMENT ME WITH MPI
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


struct Param {
  uint64_t N_max;
  double sigma, error_rate, benefit_max, benefit_delta;
  int m0, m1;
  std::array<int, 2> proc_grid_size;
  int block_size;  // block size
  explicit Param(const nlohmann::json& j) {
    sigma = j.at("sigma").get<double>();
    error_rate = j.at("error_rate").get<double>();
    benefit_max = j.at("benefit_max").get<double>();
    benefit_delta = j.at("benefit_delta").get<double>();
    m0 = j.at("strategy_space").at(0).get<int>();
    m1 = j.at("strategy_space").at(1).get<int>();
    proc_grid_size[0] = j.at("process_grid_size").at(0).get<int>();
    proc_grid_size[1] = j.at("process_grid_size").at(1).get<int>();
    block_size = j.at("block_size").get<int>();
  }
};

Param LoadInputParameters(const char* input_json_path) {
  int my_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  std::vector<uint8_t> buf;
  using json = nlohmann::json;

  if (my_rank == 0) {
    json input;
    std::ifstream fin(input_json_path);
    fin >> input;
    buf = std::move(json::to_msgpack(input));
    uint64_t size = buf.size();
    MPI_Bcast(&size, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(buf.data(), buf.size(), MPI_BYTE, 0, MPI_COMM_WORLD);
  }
  else {
    uint64_t size;
    MPI_Bcast(&size, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
    buf.resize(size);
    MPI_Bcast(buf.data(), buf.size(), MPI_BYTE, 0, MPI_COMM_WORLD);
  }

  json j = json::from_msgpack(buf);
  return Param(j);
}


int main(int argc, char *argv[]) {
  Eigen::initParallel();

  MPI_Init(&argc, &argv);

  if( argc != 2 ) {
    std::cerr << "Error : invalid argument" << std::endl;
    std::cerr << "  Usage : " << argv[0] << " <parameter_json_file>" << std::endl;
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  Param p = LoadInputParameters(argv[1]);
  StrategySpace space(p.m0, p.m1);

  int n_procs;
  MPI_Comm_size(MPI_COMM_WORLD, &n_procs);
  if (n_procs != p.proc_grid_size[0] * p.proc_grid_size[1]) {
    std::cerr << "Error: invalid number of processes: " << n_procs << std::endl;
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  MatrixSolver solver(p.proc_grid_size, space.Size(), p.block_size);
  const bool is_root = (solver.MYROW == 0 && solver.MYCOL == 0);

  MeasureElapsed("initialize");

  EvolutionaryGame eco(space, p.error_rate);

  MeasureElapsed("sweep over beta_N");

  std::ofstream eqout;
  std::vector<std::map<double, double> > c_levels; // c_levels[N][beta]
  if (is_root) {
    eqout.open("abundance.dat");
    c_levels.resize(p.N_max+1);
  }

  for (int N = 2; N <= p.N_max; N++) {
    for (int i = 1; ; i++) {
      double benefit = 1.0 + p.benefit_delta * i;
      if (benefit > p.benefit_max + 1.0e-6) { break;}
      auto eq = eco.CalculateEquilibrium(benefit, 1.0, N, p.sigma);
      if (is_root) {
        eqout << N << ' ' << benefit << ' ';
        for (double x: eq) { eqout << x << ' '; }
        eqout << std::endl;
        c_levels[N][benefit] = eco.CooperationLevel(eq);
      }
    }
  }

  if (is_root) {
    eqout.close();
    std::ofstream fout("cooperation_level.dat");
    fout << -1;
    for (const auto& pair: c_levels[2]) {  // print header
      fout << ' ' << pair.first;
    }
    fout << "\n";
    for (int N = 2; N <= p.N_max; N++) {
      fout << N;
      for (const auto& pair: c_levels[N]) {
        fout << ' ' << pair.second;
      }
      fout << "\n";
    }
    fout.close();
  }

  MeasureElapsed("done");
  return 0;
}