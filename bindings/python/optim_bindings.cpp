// SGD/Adam bindings.
#include "mlp_wrapper.hpp"

#include "fastmlp/optimizer.hpp"

namespace fastmlp_py {

// Thin wrappers, not the core Optimizer types directly: fastmlp::SGD/Adam
// only holds raw Linear* pointers gathered from the tree at construction --
// if the Python model object were garbage-collected first, those pointers
// would dangle. Holding
// model_ here keeps the whole tree (and every Linear it owns) alive for as
// long as the optimizer is.
class SGD {
 public:
  SGD(std::shared_ptr<MLP> model, float lr, float momentum, float weight_decay)
      : model_(std::move(model)), impl_(model_->module(), lr, momentum, weight_decay) {}
  void step() { impl_.step(); }
  void zero_grad() { impl_.zero_grad(); }

 private:
  std::shared_ptr<MLP> model_;
  fastmlp::SGD impl_;
};

class Adam {
 public:
  Adam(std::shared_ptr<MLP> model, float lr, float beta1, float beta2, float eps, float weight_decay)
      : model_(std::move(model)), impl_(model_->module(), lr, beta1, beta2, eps, weight_decay) {}
  void step() { impl_.step(); }
  void zero_grad() { impl_.zero_grad(); }

 private:
  std::shared_ptr<MLP> model_;
  fastmlp::Adam impl_;
};

}  // namespace fastmlp_py

void register_optim(nb::module_& m) {
  using namespace fastmlp_py;
  using namespace nb::literals;

  nb::class_<SGD>(m, "SGD")
      .def(nb::init<std::shared_ptr<MLP>, float, float, float>(), "model"_a, "lr"_a,
           "momentum"_a = 0.0f, "weight_decay"_a = 0.0f)
      .def("step", &SGD::step)
      .def("zero_grad", &SGD::zero_grad);

  nb::class_<Adam>(m, "Adam")
      .def(nb::init<std::shared_ptr<MLP>, float, float, float, float, float>(), "model"_a,
           "lr"_a = 1e-3f, "beta1"_a = 0.9f, "beta2"_a = 0.999f, "eps"_a = 1e-8f,
           "weight_decay"_a = 0.0f)
      .def("step", &Adam::step)
      .def("zero_grad", &Adam::zero_grad);
}
