#pragma once
// The single translation layer between fastmlp_core and Python: every
// NumPy-contract check (dtype/contiguity via the ndarray
// template parameters, feature-dim/batch-size via explicit checks below)
// lives here, and nothing here leaks back into core/.
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>

#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

#include "fastmlp/errors.hpp"
#include "fastmlp/mlpf.hpp"
#include "fastmlp/module.hpp"

namespace nb = nanobind;

namespace fastmlp_py {

inline fastmlp::ActivationKind activation_from_str(const std::string& s) {
  auto k = fastmlp::activation_from_name(s);
  if (!k) {
    throw std::invalid_argument("fastmlp: unknown activation '" + s + "' (valid: " +
                                 fastmlp::activation_names_list() + ")");
  }
  return *k;
}

// `activation=` is either one name for every layer or a list with one name
// per layer; `output_activation=` overrides just the last layer of the
// one-name form (the common "relu hidden, linear output" case). Giving both a
// list and output_activation would be two sources of truth for the last
// layer, so that is an error rather than a precedence rule to remember.
using ActivationArg = std::variant<std::string, std::vector<std::string>>;

inline void resolve_activations(const ActivationArg& activation,
                                 const std::optional<std::string>& output_activation,
                                 fastmlp::ModuleConfig& cfg) {
  // < 2 sizes: let Module's own ShapeError report it.
  const size_t n_layers = cfg.layer_sizes.size() < 2 ? 0 : cfg.layer_sizes.size() - 1;
  if (const auto* names = std::get_if<std::vector<std::string>>(&activation)) {
    if (output_activation) {
      throw std::invalid_argument(
          "fastmlp: pass either a list of activations or output_activation, not both");
    }
    if (names->size() != n_layers) {
      throw std::invalid_argument("fastmlp: activation list has " + std::to_string(names->size()) +
                                   " entries, expected one per layer (" +
                                   std::to_string(n_layers) + ")");
    }
    for (const auto& n : *names) cfg.activations.push_back(activation_from_str(n));
    return;
  }
  cfg.activation = activation_from_str(std::get<std::string>(activation));
  if (output_activation && n_layers > 0) {
    cfg.activations.assign(n_layers, cfg.activation);
    cfg.activations.back() = activation_from_str(*output_activation);
  }
}

// Input arrays accept both writable and read-only NumPy arrays (Scalar is
// const); output arrays are handed back read-only, since they alias a
// buffer this module overwrites on the next forward() call -- mutating them
// from Python would be a footgun, not a feature.
using InArray = nb::ndarray<const float, nb::c_contig, nb::device::cpu>;
using OutArray = nb::ndarray<nb::numpy, const float>;

// Wraps one fastmlp::Module node plus everything the C++ core deliberately
// doesn't track itself: Python-side child lifetime (core Modules are
// non-owning, see module.hpp), a resolved depth-first leaf name list for
// the dict-call form, and the path a node was last saved to (needed so a
// parent's save(inline_inputs=False) can find its already-saved children --
// the child never receives that path from anywhere else).
//
// Deliberately NOT std::enable_shared_from_this: nanobind only engages a
// registered instance's internal weak_ptr the first time it converts that
// Python object into a std::shared_ptr<MLP> (e.g. as an SGD/Adam
// constructor argument) -- shared_from_this() throws std::bad_weak_ptr on
// any object that hasn't been through that conversion yet (e.g. an
// inference_only model with no optimizer, called immediately after
// construction). call() below uses nb::find(this) instead, which looks up
// this object's existing Python wrapper unconditionally.
class MLP {
 public:
  MLP(std::vector<int> layers, const ActivationArg& activation, int max_batch,
      std::vector<std::shared_ptr<MLP>> inputs, bool inference_only,
      std::optional<std::string> name, std::optional<uint64_t> seed,
      std::optional<std::string> output_activation)
      : child_refs_(std::move(inputs)), own_name_(std::move(name)) {
    std::vector<fastmlp::Module*> raw_inputs;
    raw_inputs.reserve(child_refs_.size());
    for (auto& c : child_refs_) raw_inputs.push_back(c->module_ptr_);

    fastmlp::ModuleConfig cfg;
    cfg.layer_sizes = std::move(layers);
    resolve_activations(activation, output_activation, cfg);
    cfg.max_batch = max_batch;
    cfg.inference_only = inference_only;
    cfg.seed = seed;

    owned_module_ = std::make_unique<fastmlp::Module>(cfg, raw_inputs);
    module_ptr_ = owned_module_.get();
    compute_leaves();
  }

  // MLP::load()'s helper: wraps an already-constructed subtree owned by a
  // shared LoadResult. No per-child Python wrappers exist for a loaded
  // tree (nothing reconstructs them), so .inputs() is empty and leaf names
  // are always auto-generated.
  static std::shared_ptr<MLP> wrap_loaded(fastmlp::Module* root,
                                           std::shared_ptr<fastmlp::mlpf::LoadResult> owner) {
    auto self = std::shared_ptr<MLP>(new MLP());
    self->module_ptr_ = root;
    self->load_result_ = std::move(owner);
    self->compute_leaves_from_module();
    return self;
  }

  nb::object call(nb::args args) {
    int batch = -1;
    std::vector<const float*> leaf_ptrs(leaves_.size(), nullptr);

    auto bind_one = [&](size_t idx, nb::handle obj, const std::string& label) {
      InArray arr;
      // convert=false: reject a dtype/layout mismatch outright. The default
      // try_cast would silently build a converted copy and hand back a
      // pointer into a temporary array with no registered cleanup.
      if (!nb::try_cast<InArray>(obj, arr, false)) {
        throw std::invalid_argument("fastmlp: input '" + label +
                                     "' must be a float32, C-contiguous NumPy array");
      }
      if (arr.ndim() != 2) {
        throw std::invalid_argument("fastmlp: input '" + label + "' must be 2D (batch, features)");
      }
      int this_batch = (int)arr.shape(0);
      int features = (int)arr.shape(1);
      int expected = leaves_[idx]->in_features();
      if (features != expected) {
        throw std::invalid_argument("fastmlp: input '" + label + "' has " +
                                     std::to_string(features) + " features, expected " +
                                     std::to_string(expected));
      }
      if (batch == -1) {
        batch = this_batch;
      } else if (this_batch != batch) {
        throw std::invalid_argument("fastmlp: all inputs must share the same batch size (got " +
                                     std::to_string(batch) + " and " + std::to_string(this_batch) +
                                     ")");
      }
      leaf_ptrs[idx] = arr.data();
    };

    if (args.size() == 1 && nb::isinstance<nb::dict>(args[0])) {
      nb::dict d = nb::borrow<nb::dict>(args[0]);
      if (d.size() != leaves_.size()) {
        throw std::invalid_argument("fastmlp: expected " + std::to_string(leaves_.size()) +
                                     " leaf inputs, dict has " + std::to_string(d.size()));
      }
      for (size_t i = 0; i < leaves_.size(); i++) {
        nb::object key = nb::cast(leaf_names_[i]);
        if (!d.contains(key)) {
          throw std::invalid_argument("fastmlp: missing leaf input '" + leaf_names_[i] + "'");
        }
        bind_one(i, d[key], leaf_names_[i]);
      }
    } else {
      if (args.size() != leaves_.size()) {
        throw std::invalid_argument("fastmlp: expected " + std::to_string(leaves_.size()) +
                                     " positional inputs (flattened depth-first leaf order), got " +
                                     std::to_string(args.size()));
      }
      for (size_t i = 0; i < leaves_.size(); i++) {
        bind_one(i, args[i], leaf_names_[i]);
      }
    }

    if (batch <= 0 || batch > module_ptr_->max_batch()) {
      throw std::invalid_argument("fastmlp: batch size " + std::to_string(batch) +
                                   " out of range (max_batch=" +
                                   std::to_string(module_ptr_->max_batch()) + ")");
    }

    const float* out = module_ptr_->forward(leaf_ptrs, batch);
    size_t shape[2] = {(size_t)batch, (size_t)module_ptr_->out_features()};
    return nb::cast(OutArray(out, 2, shape, nb::find(this)));
  }

  void backward(InArray grad_out) {
    if (grad_out.ndim() != 2) {
      throw std::invalid_argument("fastmlp: grad_out must be 2D (batch, out_features)");
    }
    int batch = (int)grad_out.shape(0);
    int features = (int)grad_out.shape(1);
    if (features != module_ptr_->out_features()) {
      throw std::invalid_argument("fastmlp: grad_out has " + std::to_string(features) +
                                   " features, expected " +
                                   std::to_string(module_ptr_->out_features()));
    }
    module_ptr_->backward(grad_out.data(), batch);
  }

  void freeze() { module_ptr_->freeze(); }
  void unfreeze() { module_ptr_->unfreeze(); }
  void freeze_tree() { module_ptr_->freeze_tree(); }
  void unfreeze_tree() { module_ptr_->unfreeze_tree(); }
  void zero_grad_tree() { module_ptr_->zero_grad_tree(); }

  void save(const std::string& path, bool inline_inputs) {
    std::vector<std::string> child_paths;
    if (!inline_inputs && !module_ptr_->is_leaf()) {
      if (child_refs_.size() != module_ptr_->inputs().size()) {
        throw std::runtime_error(
            "fastmlp: save(inline_inputs=False) needs a Python MLP object for every child "
            "(not available on a tree returned by MLP.load())");
      }
      child_paths.reserve(child_refs_.size());
      for (auto& c : child_refs_) {
        if (!c->last_saved_path_) {
          throw std::runtime_error(
              "fastmlp: every child must be saved with its own .save(path) before its parent "
              "can be saved with inline_inputs=False");
        }
        child_paths.push_back(*c->last_saved_path_);
      }
    }
    fastmlp::mlpf::save(*module_ptr_, path, inline_inputs, child_paths);
    last_saved_path_ = path;
  }

  void load_weights(const std::string& path) { fastmlp::mlpf::load_weights_into(*module_ptr_, path); }

  static std::shared_ptr<MLP> load(const std::string& path, std::optional<int> max_batch,
                                    bool inference_only) {
    fastmlp::mlpf::LoadOptions opts;
    opts.max_batch_override = max_batch;
    opts.inference_only = inference_only;
    auto result = std::make_shared<fastmlp::mlpf::LoadResult>(fastmlp::mlpf::load(path, opts));
    fastmlp::Module* root = &result->root();
    return wrap_loaded(root, std::move(result));
  }

  std::vector<std::shared_ptr<MLP>> inputs() const { return child_refs_; }
  int in_features() const { return module_ptr_->in_features(); }
  int out_features() const { return module_ptr_->out_features(); }
  int max_batch() const { return module_ptr_->max_batch(); }
  bool inference_only() const { return module_ptr_->inference_only(); }
  bool is_leaf() const { return module_ptr_->is_leaf(); }
  std::vector<std::string> activations() const {
    std::vector<std::string> out;
    for (const auto& l : module_ptr_->layers()) out.push_back(fastmlp::activation_name(l.activation()));
    return out;
  }

  fastmlp::Module& module() { return *module_ptr_; }

 private:
  MLP() = default;

  void compute_leaves() {
    leaves_.clear();
    leaf_explicit_names_.clear();
    if (module_ptr_->is_leaf()) {
      leaves_.push_back(module_ptr_);
      leaf_explicit_names_.push_back(own_name_);
    } else {
      for (auto& c : child_refs_) {
        leaves_.insert(leaves_.end(), c->leaves_.begin(), c->leaves_.end());
        leaf_explicit_names_.insert(leaf_explicit_names_.end(), c->leaf_explicit_names_.begin(),
                                     c->leaf_explicit_names_.end());
      }
    }
    resolve_leaf_names();
  }

  void compute_leaves_from_module() {
    leaves_.clear();
    module_ptr_->for_each_in_tree([this](fastmlp::Module& m) {
      if (m.is_leaf()) leaves_.push_back(&m);
    });
    leaf_explicit_names_.assign(leaves_.size(), std::nullopt);
    resolve_leaf_names();
  }

  void resolve_leaf_names() {
    leaf_names_.resize(leaves_.size());
    for (size_t i = 0; i < leaves_.size(); i++) {
      leaf_names_[i] =
          leaf_explicit_names_[i] ? *leaf_explicit_names_[i] : ("leaf_" + std::to_string(i));
    }
  }

  std::unique_ptr<fastmlp::Module> owned_module_;               // set unless loaded via load()
  std::shared_ptr<fastmlp::mlpf::LoadResult> load_result_;      // set only when loaded via load()
  fastmlp::Module* module_ptr_ = nullptr;                       // always points at the live Module

  std::vector<std::shared_ptr<MLP>> child_refs_;  // empty for a loaded tree -- see wrap_loaded()
  std::vector<fastmlp::Module*> leaves_;
  std::vector<std::optional<std::string>> leaf_explicit_names_;
  std::vector<std::string> leaf_names_;

  std::optional<std::string> own_name_;
  std::optional<std::string> last_saved_path_;
};

}  // namespace fastmlp_py
