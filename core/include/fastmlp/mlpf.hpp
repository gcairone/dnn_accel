#pragma once
#include "fastmlp/module.hpp"

#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace fastmlp::mlpf {

// .mlpf format (idea.md, "Model File Format"): one line of human-readable
// JSON ("MLPF1" magic + header), followed immediately by a raw
// concatenated float32 weight blob, one layer after another in declaration
// order (own layers first, then children recursively, depth-first).
//
// The JSON reader/writer here is hand-rolled and restricted to exactly the
// .mlpf vocabulary (strings, ints, arrays of ints, arrays of
// strings-or-nested-objects for "inputs") -- not a general JSON library,
// by deliberate choice (kept dependency-free, consistent with the
// project's other "no dependency until proven necessary" decisions).

struct LoadOptions {
  std::optional<int> max_batch_override = std::nullopt;
  bool inference_only = false;
};

// Owns the WHOLE loaded tree (root + every descendant, whether inline or
// loaded from a separate multi-file path) as one unit -- Module itself
// never owns its children (see module.hpp), so something has to keep every
// node alive together. nodes.back() is always the root (children are
// always constructed, and therefore pushed, before their parent).
struct LoadResult {
  std::vector<std::unique_ptr<Module>> nodes;
  Module& root() const { return *nodes.back(); }
};

LoadResult load(const std::string& path, const LoadOptions& opts = {});

// Saves `root`'s own layer weights plus, when inline_inputs is true,
// every descendant's weights too (recursively, in the same file). When
// inline_inputs is false, only `root`'s own weights are written and
// `child_paths` (one entry per root.inputs(), already-saved separately by
// the caller -- see usages.md's enc_x.save(...) pattern) is recorded in
// the header as path references instead.
void save(const Module& root, const std::string& path, bool inline_inputs = true,
          const std::vector<std::string>& child_paths = {});

// Loads weights into an ALREADY-CONSTRUCTED, shape-matching tree (the
// pattern usages.md actually exercises: enc_x.load_weights(...),
// model.load_weights(...)). Validates the entire tree's shapes against the
// file FIRST, before writing any weights, so a mismatch anywhere in a
// single file never leaves a partial load behind. (Note: this atomicity is
// per-file; a multi-file tree where a later file fails after an earlier
// file's weights were already applied is a documented, accepted gap --
// full cross-file atomicity isn't worth the extra complexity here.)
// Throws ShapeError on any shape/layer-count/child-count mismatch.
void load_weights_into(Module& root, const std::string& path);

}  // namespace fastmlp::mlpf
