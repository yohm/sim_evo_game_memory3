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
#include "icecream-cpp/icecream.hpp"
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
    // IC(i, j, ss_ij, ss_ji);
    return {
      ss_ij[0] * benefit - ss_ij[1] * cost,
      ss_ji[0] * benefit - ss_ji[1] * cost
    };
  }

  // calculate the equilibrium distribution exactly by linear algebra
  Scalapack::GMatrix CalculateEquilibrium(double benefit, double cost, uint64_t N, double sigma, size_t block_size) const {
    Scalapack::LMatrix A(N_SPECIES, N_SPECIES, block_size, block_size);
    // calculate off-diagonal elements
    for (int ii = 0; ii < A.SUB_ROWS * A.SUB_COLS; ii++) {
      int i = ii / A.SUB_COLS;
      int j = ii % A.SUB_COLS;
      auto IJ = A.ToGlobalCoordinate(i, j);
      int I = IJ[0], J = IJ[1];
      if (I >= N_SPECIES || J >= N_SPECIES) continue;
      if (I == J) continue;  // diagonal elements are calculated later
      double p = FixationProb(benefit, cost, N, sigma, I, J);
      A.Set(i, j, p * (1.0/N_SPECIES) );
    }
    A.DebugPrintAtRoot(std::cerr);

    // calculate diagonal elements
    Scalapack::LMatrix One(1, N_SPECIES, 1, block_size);
    One.SetAll(1.0);

    Scalapack::LMatrix D(1, N_SPECIES, 1, block_size);
    D.SetAll(1.0);

    // D = - One*A + One (calculate: 1.0 - \sum_j A_ij)
    Scalapack::CallPDGEMM(-1.0, One, A, 1.0, D);

    // std::cerr << "D:\n";
    // D.DebugPrintAtRoot(std::cerr);

    // set diagonal elements
    // subtract I since we solve Ax = x -> (A-I)x = 0
    auto gD = D.ConstructGlobalMatrix();
    for (int I = 0; I < N_SPECIES; I++) {
      A.SetByGlobalCoordinate(I, I, gD.At(0, I)-1.0);
    }

    // normalization condition (last row sums upto 1)
    for (int I = 0; I < N_SPECIES; I++) {
      auto local_pos = A.ToLocalCoordinate(N_SPECIES-1, I);
      auto proc_grid = local_pos.second;
      if (proc_grid[0] == Scalapack::MYROW && proc_grid[1] == Scalapack::MYCOL) {
        size_t i = local_pos.first[0], j = local_pos.first[1];
        A.Set(i, j, A.At(i, j)+1.0 );
      }
    }

    // std::cerr << "A (final):\n";
    // A.DebugPrintAtRoot(std::cerr);

    // B = (0 0 0 ... 1)
    Scalapack::LMatrix B(N_SPECIES, 1, block_size, 1);
    B.SetByGlobalCoordinate(N_SPECIES-1, 0, 1.0);

    // std::cerr << "B:\n";
    // B.DebugPrintAtRoot(std::cerr);

    // std::cerr << "PDGESV:\n";
    Scalapack::CallPDGESV(A, B);

    // std::cerr << "X:\n";
    // B.DebugPrintAtRoot(std::cerr);
    return B.ConstructGlobalMatrix();
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
  double CooperationLevel(Scalapack::GMatrix& eq_rate) const {
    assert(eq_rate.Size() == N_SPECIES);
    // for a better load-balancing, minimal block size is used
    Scalapack::LMatrix C(N_SPECIES, 1, 1, 1);
    for (int i = 0; i < C.SUB_ROWS; i++) {
      auto I = C.ToGlobalCoordinate(i, 0)[0];
      if (I < N_SPECIES) {
        double c_lev = CooperationLevelSpecies(I);
        C.Set(i, 0, eq_rate.At(I, 0) * c_lev);
      }
    }
    Scalapack::GMatrix gC = C.ConstructGlobalMatrix();
    double ans = 0.0;
    for (size_t I = 0; I < N_SPECIES; I++) {
      ans += gC.At(I, 0);
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
    N_max = j.at("N_max").get<uint64_t>();
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

  Scalapack::Initialize(p.proc_grid_size);
  const bool is_root = (Scalapack::MYROW == 0 && Scalapack::MYCOL == 0);

  if (is_root) MeasureElapsed("initialize");

  EvolutionaryGame eco(space, p.error_rate);

  if (is_root) MeasureElapsed("sweep over beta_N");

  std::ofstream eqout;
  std::vector<std::map<double, double> > c_levels; // c_levels[N][beta]
  if (is_root) {
    eqout.open("abundance.dat");
    c_levels.resize(p.N_max+1);
  }

  for (int N = 2; N <= p.N_max; N++) {
    for (int i = 1; ; i++) {
      double benefit = 1.0 + p.benefit_delta * i;
      if (is_root) std::cerr << "N,benefit: " << N << ' ' << benefit << std::endl;
      if (benefit > p.benefit_max + 1.0e-6) break;
      auto eq = eco.CalculateEquilibrium(benefit, 1.0, N, p.sigma, p.block_size);
      double pc = eco.CooperationLevel(eq);
      if (is_root) {
        eqout << N << ' ' << benefit << ' ';
        for (size_t i = 0; i < eq.Size(); i++) { eqout << eq.At(i, 0) << ' '; }
        eqout << std::endl;
        c_levels[N][benefit] = pc;
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

  if (is_root) MeasureElapsed("done");
  Scalapack::Finalize();
  return 0;
}