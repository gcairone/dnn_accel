#pragma once
#include "fastmlp/module.hpp"

#include <vector>

namespace fastmlp {

// Base constructor walks root.for_each_in_tree(...) ONCE, caching a flat
// list of every Linear* in the whole tree. step()/zero_grad() iterate that
// cached list (skipping frozen layers), never re-walking the tree or
// allocating -- idea.md's zero-allocation-in-the-hot-path contract applies
// to the training loop as a whole, not just the forward/backward kernels.
class Optimizer {
 public:
  virtual ~Optimizer() = default;
  virtual void step() = 0;
  virtual void zero_grad() = 0;

 protected:
  explicit Optimizer(Module& root);
  std::vector<Linear*> params_;
};

class SGD : public Optimizer {
 public:
  SGD(Module& root, float lr, float momentum = 0.0f, float weight_decay = 0.0f);
  void step() override;
  void zero_grad() override;

 private:
  float lr_, momentum_, weight_decay_;
  std::vector<std::vector<float>> vel_W_, vel_b_;  // one per param, preallocated at construction
};

class Adam : public Optimizer {
 public:
  Adam(Module& root, float lr = 1e-3f, float beta1 = 0.9f, float beta2 = 0.999f,
       float eps = 1e-8f, float weight_decay = 0.0f);
  void step() override;
  void zero_grad() override;

 private:
  float lr_, beta1_, beta2_, eps_, weight_decay_;
  long t_ = 0;
  std::vector<std::vector<float>> m_W_, v_W_, m_b_, v_b_;
};

}  // namespace fastmlp
