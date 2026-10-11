#include "fastmlp/mlpf.hpp"

#include <cctype>
#include <fstream>
#include <memory>
#include <sstream>

#include "fastmlp/errors.hpp"

namespace fastmlp::mlpf {

namespace {

// --- activation <-> string ------------------------------------------

ActivationKind activation_from_string(const std::string& s) {
  auto k = activation_from_name(s);
  if (!k) throw ShapeError("mlpf: unknown activation '" + s + "'");
  return *k;
}

// --- minimal hand-rolled JSON, restricted to the .mlpf vocabulary ----

struct ParsedHeader;

// A ParsedHeader can't contain a by-value ParsedHeader member (recursive
// size), so the inline case goes through a unique_ptr indirection.
struct ParsedChildRef {
  bool is_path;
  std::string path;                          // valid iff is_path
  std::unique_ptr<ParsedHeader> inline_header;  // valid iff !is_path
};

struct ParsedHeader {
  std::string name;
  // One entry = every layer (how MLPF1 always wrote it, and still writes a
  // uniform stack); otherwise one per layer.
  std::vector<std::string> activations;
  int max_batch = 0;
  std::vector<int> layers;
  std::vector<ParsedChildRef> children;
};

class JsonCursor {
 public:
  // s_ owns a copy, not a reference: callers often pass a temporary
  // (e.g. line.substr(...)), and a reference member bound to that would
  // dangle the instant the constructor call's full-expression ends.
  explicit JsonCursor(std::string s) : s_(std::move(s)) {}

  void skip_ws() {
    while (pos_ < s_.size() && std::isspace((unsigned char)s_[pos_])) pos_++;
  }
  char peek() {
    skip_ws();
    return pos_ < s_.size() ? s_[pos_] : '\0';
  }
  void expect(char c) {
    skip_ws();
    if (pos_ >= s_.size() || s_[pos_] != c) {
      throw ShapeError(std::string("mlpf: malformed header, expected '") + c + "'");
    }
    pos_++;
  }
  bool try_char(char c) {
    skip_ws();
    if (pos_ < s_.size() && s_[pos_] == c) {
      pos_++;
      return true;
    }
    return false;
  }
  std::string parse_string() {
    expect('"');
    std::string out;
    while (pos_ < s_.size() && s_[pos_] != '"') {
      if (s_[pos_] == '\\' && pos_ + 1 < s_.size()) {
        pos_++;
      }
      out.push_back(s_[pos_]);
      pos_++;
    }
    expect('"');
    return out;
  }
  long parse_int() {
    skip_ws();
    size_t start = pos_;
    if (pos_ < s_.size() && (s_[pos_] == '-' || s_[pos_] == '+')) pos_++;
    while (pos_ < s_.size() && std::isdigit((unsigned char)s_[pos_])) pos_++;
    if (pos_ == start) throw ShapeError("mlpf: expected an integer in header");
    return std::stol(s_.substr(start, pos_ - start));
  }

 private:
  std::string s_;
  size_t pos_ = 0;
};

ParsedHeader parse_object(JsonCursor& cur) {
  ParsedHeader h;
  cur.expect('{');
  if (cur.try_char('}')) return h;
  while (true) {
    std::string key = cur.parse_string();
    cur.expect(':');
    if (key == "name") {
      h.name = cur.parse_string();
    } else if (key == "dtype") {
      std::string dtype = cur.parse_string();
      if (dtype != "float32") throw ShapeError("mlpf: unsupported dtype '" + dtype + "'");
    } else if (key == "activation") {
      if (cur.peek() == '[') {
        cur.expect('[');
        while (true) {
          h.activations.push_back(cur.parse_string());
          if (cur.try_char(',')) continue;
          cur.expect(']');
          break;
        }
      } else {
        h.activations.push_back(cur.parse_string());
      }
    } else if (key == "max_batch") {
      h.max_batch = (int)cur.parse_int();
    } else if (key == "layers") {
      cur.expect('[');
      if (!cur.try_char(']')) {
        while (true) {
          h.layers.push_back((int)cur.parse_int());
          if (cur.try_char(',')) continue;
          cur.expect(']');
          break;
        }
      }
    } else if (key == "inputs") {
      cur.expect('[');
      if (!cur.try_char(']')) {
        while (true) {
          ParsedChildRef c;
          if (cur.peek() == '"') {
            c.is_path = true;
            c.path = cur.parse_string();
          } else {
            c.is_path = false;
            c.inline_header = std::make_unique<ParsedHeader>(parse_object(cur));
          }
          h.children.push_back(std::move(c));
          if (cur.try_char(',')) continue;
          cur.expect(']');
          break;
        }
      }
    } else {
      throw ShapeError("mlpf: unrecognized header key '" + key + "'");
    }
    if (cur.try_char(',')) continue;
    cur.expect('}');
    break;
  }
  return h;
}

ParsedHeader parse_header_line(std::istream& in) {
  std::string line;
  std::getline(in, line);
  static const std::string kMagic = "MLPF1";
  if (line.size() < kMagic.size() || line.compare(0, kMagic.size(), kMagic) != 0) {
    throw ShapeError("mlpf: bad magic/version, expected MLPF1");
  }
  JsonCursor cur(line.substr(kMagic.size()));
  return parse_object(cur);
}

void write_string(std::ostream& out, const std::string& s) {
  out << '"';
  for (char c : s) {
    if (c == '"' || c == '\\') out << '\\';
    out << c;
  }
  out << '"';
}

std::vector<int> reconstruct_layer_sizes(const Module& node) {
  std::vector<int> sizes;
  sizes.push_back(node.in_features());
  for (auto& l : node.layers()) sizes.push_back(l.out_features());
  return sizes;
}

void write_header_object(std::ostream& out, const Module& node, const std::string& name,
                          bool inline_children, const std::vector<std::string>* child_paths) {
  out << '{';
  out << "\"name\":";
  write_string(out, name);
  out << ",\"dtype\":\"float32\"";
  // A uniform stack keeps the plain-string form, so such files stay
  // byte-identical to what pre-per-layer versions wrote (and readable by them).
  out << ",\"activation\":";
  const auto& layers = node.layers();
  bool uniform = true;
  for (auto& l : layers) uniform = uniform && l.activation() == layers.front().activation();
  if (uniform) {
    write_string(out, activation_name(layers.front().activation()));
  } else {
    out << '[';
    for (size_t i = 0; i < layers.size(); i++) {
      if (i) out << ',';
      write_string(out, activation_name(layers[i].activation()));
    }
    out << ']';
  }
  out << ",\"max_batch\":" << node.max_batch();
  out << ",\"layers\":[";
  auto sizes = reconstruct_layer_sizes(node);
  for (size_t i = 0; i < sizes.size(); i++) out << (i ? "," : "") << sizes[i];
  out << "],\"inputs\":[";
  if (!node.is_leaf()) {
    for (size_t i = 0; i < node.inputs().size(); i++) {
      if (i) out << ",";
      if (inline_children) {
        write_header_object(out, *node.inputs()[i], name, true, nullptr);
      } else {
        write_string(out, (*child_paths)[i]);
      }
    }
  }
  out << "]}";
}

void write_weights_blob(std::ostream& out, const Module& node, bool inline_children) {
  for (auto& l : node.layers()) {
    out.write((const char*)l.weights(), sizeof(float) * (size_t)l.out_features() * l.in_features());
    out.write((const char*)l.bias(), sizeof(float) * l.out_features());
  }
  if (inline_children) {
    for (auto* child : node.inputs()) write_weights_blob(out, *child, true);
  }
}

// Used by load_weights_into(): validates the WHOLE tree (including opening
// path-referenced files just far enough to read their header) before any
// weight bytes are written anywhere, so a mismatch anywhere never leaves a
// partial load behind.
void validate_shapes_match(const Module& node, const ParsedHeader& h) {
  if (reconstruct_layer_sizes(node) != h.layers) {
    throw ShapeError("mlpf: layer shape mismatch against constructed tree");
  }
  if (h.children.size() != node.inputs().size()) {
    throw ShapeError("mlpf: child count mismatch against constructed tree");
  }
  for (size_t i = 0; i < h.children.size(); i++) {
    if (h.children[i].is_path) {
      std::ifstream child_in(h.children[i].path, std::ios::binary);
      if (!child_in) {
        throw ShapeError("mlpf: cannot open referenced file: " + h.children[i].path);
      }
      ParsedHeader child_h = parse_header_line(child_in);
      validate_shapes_match(*node.inputs()[i], child_h);
    } else {
      validate_shapes_match(*node.inputs()[i], *h.children[i].inline_header);
    }
  }
}

// Used by load(): construct_shapes() already fully handled every
// path-referenced child (construction AND weights, via its own nested
// load() call) before this runs -- so this only reads THIS blob's own
// layers plus INLINE children, and explicitly skips path children.
void read_weights_after_construct(Module& node, const ParsedHeader& h, std::ifstream& blob_in) {
  for (auto& l : node.layers()) {
    blob_in.read((char*)l.weights(), sizeof(float) * (size_t)l.out_features() * l.in_features());
    blob_in.read((char*)l.bias(), sizeof(float) * l.out_features());
    if (!blob_in) throw ShapeError("mlpf: truncated weight blob");
  }
  for (size_t i = 0; i < h.children.size(); i++) {
    if (!h.children[i].is_path) {
      read_weights_after_construct(*node.inputs()[i], *h.children[i].inline_header, blob_in);
    }
  }
}

// Used by load_weights_into(): the tree already exists, so path children's
// weights must be loaded via a recursive load_weights_into() call (nothing
// else has done it, unlike the load()/construct_shapes() pathway above).
void read_weights_into_existing(Module& node, const ParsedHeader& h, std::ifstream& blob_in) {
  for (auto& l : node.layers()) {
    blob_in.read((char*)l.weights(), sizeof(float) * (size_t)l.out_features() * l.in_features());
    blob_in.read((char*)l.bias(), sizeof(float) * l.out_features());
    if (!blob_in) throw ShapeError("mlpf: truncated weight blob");
  }
  for (size_t i = 0; i < h.children.size(); i++) {
    if (h.children[i].is_path) {
      load_weights_into(*node.inputs()[i], h.children[i].path);
    } else {
      read_weights_into_existing(*node.inputs()[i], *h.children[i].inline_header, blob_in);
    }
  }
}

Module* construct_shapes(const ParsedHeader& h, std::vector<std::unique_ptr<Module>>& out_nodes,
                          const LoadOptions& opts) {
  std::vector<Module*> child_ptrs;
  for (auto& c : h.children) {
    if (c.is_path) {
      LoadResult sub = load(c.path, opts);
      Module* child_root = &sub.root();
      for (auto& n : sub.nodes) out_nodes.push_back(std::move(n));
      child_ptrs.push_back(child_root);
    } else {
      child_ptrs.push_back(construct_shapes(*c.inline_header, out_nodes, opts));
    }
  }

  ModuleConfig cfg;
  cfg.layer_sizes = h.layers;
  if (h.activations.empty()) throw ShapeError("mlpf: header has no activation");
  if (h.activations.size() == 1) {
    cfg.activation = activation_from_string(h.activations.front());
  } else {
    // Module itself rejects a count that doesn't match the layer count.
    for (auto& a : h.activations) cfg.activations.push_back(activation_from_string(a));
  }
  cfg.max_batch = opts.max_batch_override.value_or(h.max_batch);
  cfg.inference_only = opts.inference_only;

  auto node = std::make_unique<Module>(cfg, child_ptrs);
  Module* raw = node.get();
  out_nodes.push_back(std::move(node));
  return raw;
}

}  // namespace

LoadResult load(const std::string& path, const LoadOptions& opts) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw ShapeError("mlpf::load: cannot open file: " + path);
  ParsedHeader h = parse_header_line(in);

  LoadResult result;
  Module* root = construct_shapes(h, result.nodes, opts);
  read_weights_after_construct(*root, h, in);
  return result;
}

void save(const Module& root, const std::string& path, bool inline_inputs,
          const std::vector<std::string>& child_paths) {
  if (!inline_inputs && !root.is_leaf() && child_paths.size() != root.inputs().size()) {
    throw ShapeError(
        "mlpf::save: child_paths.size() must equal root.inputs().size() when inline_inputs=false");
  }
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out) throw ShapeError("mlpf::save: cannot open file for writing: " + path);

  out << "MLPF1";
  write_header_object(out, root, "model", inline_inputs, inline_inputs ? nullptr : &child_paths);
  out << "\n";
  write_weights_blob(out, root, inline_inputs);
}

void load_weights_into(Module& root, const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw ShapeError("mlpf::load_weights_into: cannot open file: " + path);
  ParsedHeader h = parse_header_line(in);

  validate_shapes_match(root, h);  // whole tree (incl. path-referenced files), before writing anything
  read_weights_into_existing(root, h, in);
}

}  // namespace fastmlp::mlpf
