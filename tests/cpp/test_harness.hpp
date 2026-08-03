#pragma once
// Manual, minimal test harness -- no GoogleTest/Catch2 dependency, by
// deliberate choice (same "no dependency until proven necessary" philosophy
// already applied to benchmarking, see benchmarks/hardware_analysis/src/bench_gemm.c).
//
// One executable per test_*.cpp file: TEST(name){...} self-registers into a
// static list, FASTMLP_TEST_MAIN() generates main() which runs every
// registered test, catching one failing ASSERT_* without aborting the rest
// of the suite, and reports a pass/fail summary.

#include <cmath>
#include <cstdio>
#include <exception>
#include <string>
#include <vector>

namespace fastmlp_test {

struct TestFailure {
  std::string file;
  int line;
  std::string message;
};

using TestFn = void (*)();

struct TestCase {
  std::string name;
  TestFn fn;
};

inline std::vector<TestCase>& registry() {
  static std::vector<TestCase> tests;
  return tests;
}

struct Registrar {
  Registrar(const char* name, TestFn fn) { registry().push_back({name, fn}); }
};

}  // namespace fastmlp_test

#define FASTMLP_TEST_CONCAT_(a, b) a##b
#define FASTMLP_TEST_CONCAT(a, b) FASTMLP_TEST_CONCAT_(a, b)

#define TEST(name)                                                          \
  static void FASTMLP_TEST_CONCAT(fastmlp_test_fn_, name)();                \
  static ::fastmlp_test::Registrar FASTMLP_TEST_CONCAT(fastmlp_test_reg_, name)( \
      #name, &FASTMLP_TEST_CONCAT(fastmlp_test_fn_, name));                 \
  static void FASTMLP_TEST_CONCAT(fastmlp_test_fn_, name)()

#define ASSERT_TRUE(cond)                                                    \
  do {                                                                       \
    if (!(cond)) {                                                          \
      throw ::fastmlp_test::TestFailure{__FILE__, __LINE__,                 \
                                         "ASSERT_TRUE(" #cond ")"};          \
    }                                                                        \
  } while (0)

#define ASSERT_EQ(a, b)                                                      \
  do {                                                                       \
    if (!((a) == (b))) {                                                    \
      throw ::fastmlp_test::TestFailure{__FILE__, __LINE__,                 \
                                         "ASSERT_EQ(" #a ", " #b ")"};       \
    }                                                                        \
  } while (0)

#define ASSERT_NEAR(a, b, tol)                                               \
  do {                                                                       \
    double fastmlp_a_ = (a), fastmlp_b_ = (b), fastmlp_tol_ = (tol);        \
    if (std::fabs(fastmlp_a_ - fastmlp_b_) > fastmlp_tol_) {                \
      throw ::fastmlp_test::TestFailure{                                    \
          __FILE__, __LINE__,                                               \
          "ASSERT_NEAR(" #a ", " #b "): " + std::to_string(fastmlp_a_) +    \
              " vs " + std::to_string(fastmlp_b_) + " (tol " +              \
              std::to_string(fastmlp_tol_) + ")"};                          \
    }                                                                        \
  } while (0)

#define ASSERT_THROWS(expr)                                                  \
  do {                                                                       \
    bool fastmlp_threw_ = false;                                            \
    try {                                                                    \
      (expr);                                                               \
    } catch (...) {                                                         \
      fastmlp_threw_ = true;                                                \
    }                                                                        \
    if (!fastmlp_threw_) {                                                  \
      throw ::fastmlp_test::TestFailure{                                    \
          __FILE__, __LINE__, "ASSERT_THROWS(" #expr ") did not throw"};    \
    }                                                                        \
  } while (0)

#define FASTMLP_TEST_MAIN()                                                  \
  int main() {                                                              \
    int failed = 0;                                                        \
    for (auto& t : ::fastmlp_test::registry()) {                            \
      try {                                                                 \
        t.fn();                                                             \
        std::printf("[PASS] %s\n", t.name.c_str());                        \
      } catch (const ::fastmlp_test::TestFailure& f) {                      \
        std::printf("[FAIL] %s at %s:%d: %s\n", t.name.c_str(),             \
                     f.file.c_str(), f.line, f.message.c_str());            \
        failed++;                                                          \
      } catch (const std::exception& e) {                                   \
        std::printf("[FAIL] %s: unexpected exception: %s\n",                \
                     t.name.c_str(), e.what());                            \
        failed++;                                                          \
      }                                                                     \
    }                                                                        \
    std::printf("%zu test(s), %d failed\n",                                 \
                 ::fastmlp_test::registry().size(), failed);                \
    return failed == 0 ? 0 : 1;                                             \
  }
