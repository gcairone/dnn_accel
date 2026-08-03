// The nanobind module entry point. MLP itself is defined in mlp_wrapper.hpp
// (shared with optim_bindings.cpp, since SGD/Adam bind against the same
// wrapper type); this file just registers it plus module-level exceptions.
#include "mlp_wrapper.hpp"

using namespace fastmlp_py;
using namespace nb::literals;

void register_optim(nb::module_& m);  // optim_bindings.cpp

NB_MODULE(_fastmlp, m) {
  nb::exception<fastmlp::ShapeError>(m, "ShapeError", PyExc_ValueError);
  nb::exception<fastmlp::StateError>(m, "StateError", PyExc_RuntimeError);

  nb::class_<MLP>(m, "MLP")
      .def(nb::init<std::vector<int>, const std::string&, int, std::vector<std::shared_ptr<MLP>>,
                     bool, std::optional<std::string>, std::optional<uint64_t>>(),
           "layers"_a, "activation"_a = "relu", "max_batch"_a, "inputs"_a = std::vector<std::shared_ptr<MLP>>{},
           "inference_only"_a = false, "name"_a = nb::none(), "seed"_a = nb::none())
      .def("__call__", &MLP::call)
      .def("backward", &MLP::backward, "grad_out"_a.noconvert())
      .def("freeze", &MLP::freeze)
      .def("unfreeze", &MLP::unfreeze)
      .def("freeze_tree", &MLP::freeze_tree)
      .def("unfreeze_tree", &MLP::unfreeze_tree)
      .def("zero_grad_tree", &MLP::zero_grad_tree)
      .def("save", &MLP::save, "path"_a, "inline_inputs"_a = true)
      .def("load_weights", &MLP::load_weights, "path"_a)
      .def_static("load", &MLP::load, "path"_a, "max_batch"_a = nb::none(), "inference_only"_a = false)
      .def_prop_ro("inputs", &MLP::inputs)
      .def_prop_ro("in_features", &MLP::in_features)
      .def_prop_ro("out_features", &MLP::out_features)
      .def_prop_ro("max_batch", &MLP::max_batch)
      .def_prop_ro("inference_only", &MLP::inference_only)
      .def_prop_ro("is_leaf", &MLP::is_leaf);

  register_optim(m);
}
