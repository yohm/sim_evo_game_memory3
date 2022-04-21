//
// Created by Yohsuke Murase on 2021/10/25.
//

#include <iostream>
#include <vector>
#include <array>
#include <chrono>
#include <regex>
#include "GroupedEvoGame.hpp"
#include "icecream-cpp/icecream.hpp"


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

constexpr size_t N = 16;
using vd_t = std::array<double,N>;
vd_t SolveByRungeKutta(std::function<vd_t(vd_t)>& func, const vd_t& init, double dt, size_t n_iter) {
  vd_t ht = init;
  for (size_t t = 0; t < n_iter; t++) {
#ifdef DEBUG
    if (t % 10000 == 9999) {
      std::cerr << t << ' ' << ht[0] << ' ' << ht[1] << ' ' << ht[2] << std::endl;
    }
#endif
    vd_t k1 = func(ht);
    vd_t arg2;
    for(int i = 0; i < k1.size(); i++) {
      k1[i] *= dt;
      arg2[i] = ht[i] + 0.5 * k1[i];
    }
    vd_t k2 = func(arg2);
    vd_t arg3;
    for(int i = 0; i < k2.size(); i++) {
      k2[i] *= dt;
      arg3[i] = ht[i] + 0.5 * k2[i];
    }
    vd_t k3 = func(arg3);
    vd_t arg4;
    for(int i = 0; i < k3.size(); i++) {
      k3[i] *= dt;
      arg4[i] = ht[i] + k3[i];
    }
    vd_t k4 = func(arg4);
    for(int i = 0; i < k4.size(); i++) {
      k4[i] *= dt;
    }
    vd_t delta;
    double sum = 0.0;
    for (int i = 0; i < delta.size(); i++) {
      delta[i] = (k1[i] + 2.0*k2[i] + 2.0*k3[i] + k4[i]) / 6.0;
      ht[i] += delta[i];
      sum += ht[i];
    }
    // normalize ht
    double sum_inv = 1.0 / sum;
    for (int i = 0; i < ht.size(); i++) { ht[i] *= sum_inv; }
  }
  return ht;
}


int main(int argc, char *argv[]) {
  Eigen::initParallel();
  if( argc < 9 ) {
    std::cerr << "Error : invalid argument" << std::endl;
    std::cerr << "  Usage: " << argv[0] << " <N> <M> <benefit> <error_rate> <sigma_in> <sigma_out> <mu> <Tmax>" << std::endl;
    return 1;
  }

  GroupedEvoGame::Parameters prm;
  prm.N = std::stoi(argv[1]);
  prm.M = std::stoi(argv[2]);
  prm.benefit = std::stod(argv[3]);
  prm.error_rate = std::stod(argv[4]);
  prm.sigma_in = std::stod(argv[5]);
  prm.sigma_out = std::stod(argv[6]);
  prm.p_nu = std::stod(argv[7]);
  prm.T_max = std::stol(argv[8]);
  prm.strategy_space = {1, 1};
  prm.T_init = 0;
  prm._seed = 0;
  prm.initial_condition = "random";
  prm.weighted_sampling = 0;
  prm.parallel_update = 0;

  std::cerr
    << "N: " << prm.N << std::endl
    << "M: " << prm.M << std::endl
    << "benefit: " << prm.benefit << std::endl
    << "error_rate: " << prm.error_rate << std::endl
    << "sigma_in: " << prm.sigma_in << std::endl
    << "sigma_out: " << prm.sigma_out << std::endl
    << "p_nu: " << prm.p_nu << std::endl
    << "T_max: " << prm.T_max << std::endl;

  GroupedEvoGame eco(prm);

  MeasureElapsed("simulation");

  std::vector<GroupedEvoGame::Species> species;
  StrategySpace ss = {1, 1};
  for (size_t i = 0; i < N; i++) {
    uint64_t gid = ss.ToGlobalID(i);
    species.emplace_back(gid, prm.error_rate);
  }

  std::vector<std::vector<double>> rho_AB(N, std::vector<double>(N, 0.0));
  for (size_t i = 0; i < N; i++) {
    for (size_t j = 0; j < N; j++) {
      rho_AB[i][j] = eco.IntraGroupFixationProb(species[i], species[j]);
    }
  }
  std::vector<std::vector<double>> delta_p_AB(N, std::vector<double>(N, 0.0));
  for (size_t i = 0; i < N; i++) {
    for (size_t j = 0; j < N; j++) {
      double p_plus  = eco.InterGroupImitationProb(species[i], species[j]) * eco.IntraGroupFixationProb(species[i], species[j]);
      double p_minus = eco.InterGroupImitationProb(species[j], species[i]) * eco.IntraGroupFixationProb(species[j], species[i]);
      delta_p_AB[i][j] = p_plus - p_minus;
    }
  }

  double nu = prm.p_nu;  // mutation rate
  double M = static_cast<double>(prm.M);
  std::function<vd_t(vd_t)> x_dot = [&rho_AB,&delta_p_AB,nu,M](const vd_t& x) {
    vd_t ans;
    for (size_t i = 0; i < N; i++) {
      double dx = 0.0;
      for (size_t j = 0; j < N; j++) {
        if (i == j) continue;
        dx += (1.0 - nu) * x[i] * x[j] * delta_p_AB[i][j] - nu * x[i] * rho_AB[j][i] / N + nu * x[j] * rho_AB[i][j] / N;
      }
      ans[i] = dx / M;
    }
    return ans;
  };

  vd_t x;
  {
    std::ofstream fout("timeseries.dat");
    for (double& xi : x) { xi = 1.0 / N; }
    size_t dt = prm.T_max / 500;
    for (size_t t = 0; t < prm.T_max; t++) {
      x = SolveByRungeKutta(x_dot, x, 0.01, 100);
      if (t % dt == 0) {
        for (double xi : x) { fout << xi << ' '; }
        fout << std::endl;
      }
    }
  }

  {
    std::ofstream fout("delta_p.dat");
    for (size_t i = 0; i < N; i++) {
      for (size_t j = 0; j < N; j++) {
        fout << delta_p_AB[i][j] << ' ';
      }
      fout << "\n";
    }
  }

  {
    std::ofstream fout("rho.dat");
    for (size_t i = 0; i < N; i++) {
      for (size_t j = 0; j < N; j++) {
        fout << rho_AB[i][j] << ' ';
      }
      fout << "\n";
    }
  }

  {
    std::ofstream fout("top_flows.dat");
    fout << std::fixed << std::setprecision(2);
    for (size_t i = 0; i < N; i++) {
      fout << i << ' ' << x[i] << std::endl;
      std::vector< std::pair<std::string,double> > inflows;
      std::vector< std::pair<std::string,double> > outflows;
      for (size_t j = 0; j < N; j++) {
        double mig = (1.0 - nu) * x[i] * x[j] * delta_p_AB[i][j];
        std::ostringstream key;
        key << "mig:" << j;
        if (mig < 0) { outflows.push_back({key.str(), -mig}); }
        else if (mig > 0) { inflows.push_back({key.str(), mig}); }

        double mut_out = nu * x[i] * rho_AB[j][i] / N;
        key.str(""); key.clear();
        key << "mut:" << j;
        outflows.push_back({key.str(), mut_out});

        double mut_in = nu * x[j] * rho_AB[i][j] / N;
        key.str(""); key.clear();
        key << "mut:" << j;
        inflows.push_back({key.str(), mut_in});
      }

      { // print inflow
        std::sort(inflows.begin(), inflows.end(), [](const auto& lhs, const auto& rhs) {
          return lhs.second > rhs.second;
        });
        IC(inflows);
        double sum = 0.0;
        for (const auto &pair: inflows) { sum += pair.second; }
        fout << "  inflow:" << "\n";
        for (auto& pair: inflows) {
          double v = pair.second / sum;
          if (v > 0.01) { fout << "    " << pair.first << "\t" << v << "\n"; }
          else break;
        }
      }
      { // print outflow
        std::sort(outflows.begin(), outflows.end(), [](const auto& lhs, const auto& rhs) {
          return lhs.second > rhs.second;
        });
        double sum = 0.0;
        for (const auto &pair: outflows) { sum += pair.second; }
        fout << "  outflow:" << "\n";
        for (auto& pair: outflows) {
          double v = pair.second / sum;
          if (v > 0.01) { fout << "    " << pair.first << "\t" << v << "\n"; }
          else break;
        }
      }
    }
  }

  {
    nlohmann::json output;
    double c_level = 0.0;
    for (size_t i = 0; i < N; i++) {
      c_level += x[i] * species[i].cooperation_level;
    }
    output["cooperation_level"] = c_level;
    std::ofstream fout("_output.json");
    fout << output.dump(2);
    fout.close();
  }

  MeasureElapsed("done");
  return 0;
}