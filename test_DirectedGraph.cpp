#include <iostream>
#include "DirectedGraph.hpp"

void PrintComponents(const components_t& components) {
  for(auto comp: components) {
    for( auto n: comp) {
      std::cout << n << ' ';
    }
    std::cout << "\n";
  }
}

bool CompareComponents(const components_t& comps, const components_t& expected) {
  components_t a1 = comps, a2 = expected;
  if( a1.size() != a2.size() ) { return false; }
  for(auto& x: a1) { std::sort(x.begin(), x.end()); }
  for(auto& x: a2) { std::sort(x.begin(), x.end()); }

  std::sort(a1.begin(), a1.end(), [](const comp_t& lhs, const comp_t& rhs) { return (lhs[0] < rhs[0]); });
  std::sort(a2.begin(), a2.end(), [](const comp_t& lhs, const comp_t& rhs) { return (lhs[0] < rhs[0]); });

  for(size_t i=0; i<a1.size(); i++) {
    if(a1[i].size() != a2[i].size()) { return false; }
    for(size_t j=0; j<a1[i].size(); j++) {
      if(a1[i][j] != a2[i][j]) { return false; }
    }
  }
  return true;
}

void test_g1() {
  DirectedGraph g1(5);
  g1.AddLink(1, 0);
  g1.AddLink(0, 2);
  g1.AddLink(2, 1);
  g1.AddLink(0, 3);
  g1.AddLink(3, 4);
  g1.AddLink(4, 4);
  // 1 -> 0 -> 3 -> 4 --
  // ^     --> 2    ^  |
  // |---------|    |---
  {
    components_t components;
    g1.SCCs(components);
    std::cout << "g1:" << std::endl;
    PrintComponents(components);
    bool b = CompareComponents(components, {{4},{3},{1,2,0}});
    assert(b);
  }
  {
    components_t sinks = g1.SinkSCCs();
    bool b = CompareComponents(sinks, {{4}});
    assert(b);
  }
  {
    assert( g1.HasLink(0,2) );
    assert( ! g1.HasLink(0,4) );
  }
  {
    assert( g1.Reachable(0, 4) );
    assert( ! g1.Reachable(4, 0) );
  }
  {
    std::vector<long> v;
    g1.BFS(0, [&v](long n) {
      v.push_back(n);
    });
    std::vector<long> expected = {0, 2, 3, 1, 4};
    assert( v == expected );
  }
  {
    std::vector<long> v;
    g1.DFS(0, [&v](long n) {
      v.push_back(n);
    });
    std::vector<long> expected = {0, 3, 4, 2, 1};
    assert( v == expected );
  }
}

void test_g2() {
  DirectedGraph g(4);
  g.AddLink(0, 1);
  g.AddLink(1, 2);
  g.AddLink(2, 3);
  {
    components_t components;
    g.SCCs(components);
    std::cout << "g2:" << std::endl;
    PrintComponents(components);
    bool b = CompareComponents(components, {{3},{2},{1},{0}});
    assert(b);
  }
  {
    components_t sinks = g.SinkSCCs();
    bool b = CompareComponents(sinks, {{3}});
    assert(b);
  }
}

void test_g3() {
  DirectedGraph g(7);
  g.AddLink(0, 1);
  g.AddLink(1, 2);
  g.AddLink(2, 0);
  g.AddLink(1, 3);
  g.AddLink(1, 4);
  g.AddLink(1, 6);
  g.AddLink(3, 5);
  g.AddLink(4, 5);
  {
    components_t components;
    g.SCCs(components);
    std::cout << "g3:" << std::endl;
    PrintComponents(components);
  }
  {
    components_t sinks = g.SinkSCCs();
    bool b = CompareComponents(sinks, {{5},{6}});
    assert(b);
  }
}

void test_g4() {
  DirectedGraph g(11);
  g.AddLink(0,1);g.AddLink(0,3);
  g.AddLink(1,2);g.AddLink(1,4);
  g.AddLink(2,0);g.AddLink(2,6);
  g.AddLink(3,2);
  g.AddLink(4,5);g.AddLink(4,6);
  g.AddLink(5,6);g.AddLink(5,7);g.AddLink(5,8);g.AddLink(5,9);
  g.AddLink(6,4);
  g.AddLink(7,9);
  g.AddLink(8,9);
  g.AddLink(9,8);
  {
    components_t components;
    g.SCCs(components);
    std::cout << "g4:" << std::endl;
    PrintComponents(components);
  }
  {
    components_t sinks = g.SinkSCCs();
    bool b = CompareComponents(sinks, {{8,9},{10}});
    assert(b);
  }
}

void test_g5() {
  DirectedGraph g(5);
  g.AddLink(0,1);
  g.AddLink(1,2);
  g.AddLink(2,3);
  g.AddLink(2,4);
  g.AddLink(3,0);
  g.AddLink(4,2);
  {
    components_t components;
    g.SCCs(components);
    std::cout << "g5:" << std::endl;
    PrintComponents(components);
  }
  {
    components_t sinks = g.SinkSCCs();
    bool b = CompareComponents(sinks, {{4,3,2,1,0}});
    assert(b);
  }
}

void test_transitionNodes1() {
  DirectedGraph g1(5);
  g1.AddLink(1, 0);
  g1.AddLink(0, 2);
  g1.AddLink(2, 1);
  g1.AddLink(0, 3);
  g1.AddLink(3, 3);
  g1.AddLink(3, 4);
  std::cout << "transition nodes in g1':" << std::endl;
  for( auto n: g1.TransitionNodes() ) {
    std::cout << n << ' ';
  }
  std::cout << std::endl;
}

int main( int argc, char* argv[]) {
  test_g1();
  test_g2();
  test_g3();
  test_g4();
  test_g5();
  test_transitionNodes1();
}