#include <algorithm>
#include <format>
#include <iostream>

#include "equation.h"
#include "expression.h"
#include "helpers/helpers.h"
#include "sqoperator.h"
#include "tensor.h"
#include "wicked-def.h"

Equation::Equation(const SymbolicTerm &lhs, const SymbolicTerm &rhs,
                   scalar_t factor)
    : lhs_(lhs), rhs_(rhs), factor_(factor) {}

const SymbolicTerm &Equation::lhs() const { return lhs_; }

const SymbolicTerm &Equation::rhs() const { return rhs_; }

scalar_t Equation::rhs_factor() const { return factor_; }

Expression Equation::rhs_expression() const {
  Expression expr;
  expr.add(rhs(), rhs_factor());
  return expr;
}

bool Equation::operator==(Equation const &other) const {
  return ((lhs() == other.lhs()) and (rhs() == other.rhs()) and
          (rhs_factor() == other.rhs_factor()));
}

std::string Equation::str() const {
  std::vector<std::string> str_vec;
  str_vec.push_back(lhs_.str());
  str_vec.push_back("+=");
  str_vec.push_back(factor_.str());
  str_vec.push_back(rhs_.str());
  return (join(str_vec, " "));
}

std::string Equation::latex() const { return str(); }

std::string get_unique_index(const std::string &s,
                             std::map<std::string, std::string> &index_map,
                             std::vector<char> &unused_indices);

std::string
get_unique_tensor_indices(const Tensor &t,
                          std::map<std::string, std::string> &index_map,
                          std::vector<char> &unused_indices);

std::string get_unique_index(const std::string &s,
                             std::map<std::string, std::string> &index_map,
                             std::vector<std::string> &unused_indices) {
  // is this index (something like "i" or "o2") in the map? If not, figure out
  // what it corresponds to.
  if (index_map.count(s) == 0) {
    // a character
    if (s.size() == 1) {
      if (std::find(unused_indices.begin(), unused_indices.end(), s) !=
          unused_indices.end()) {
        // if this character is unused, use it
        index_map[s] = s;
      } else {
        // if it is used, grab the first available index
        index_map[s] = unused_indices.back();
      }
    } else {
      index_map[s] = unused_indices.back();
    }
    // erase this character from the available characters to avoid reusing
    unused_indices.erase(
        std::remove(unused_indices.begin(), unused_indices.end(), index_map[s]),
        unused_indices.end());
  }
  return index_map[s];
}

std::string
get_unique_tensor_indices(const Tensor &t,
                          std::map<std::string, std::string> &index_map,
                          std::vector<std::string> &unused_indices) {
  std::string indices;
  for (const auto &l : t.upper()) {
    indices += get_unique_index(l.latex(), index_map, unused_indices);
  }
  for (const auto &l : t.lower()) {
    indices += get_unique_index(l.latex(), index_map, unused_indices);
  }
  return indices;
}

std::string Equation::compile(const std::string &format) const {
  if (format == "ambit") {
    std::vector<std::string> str_vec;
    str_vec.push_back(lhs_.compile(format) + " += " + factor_.compile(format));
    str_vec.push_back(rhs_.compile(format));
    return (join(str_vec, " * ") + ";");
  }

  if (format == "einsum") {
    std::vector<std::string> str_vec;
    const auto &lhs_tensor = lhs().tensors()[0];

    std::string lhs_tensor_label = lhs_tensor.label();
    for (const auto &l : lhs_tensor.upper()) {
      lhs_tensor_label += orbital_subspaces->label(l.space());
    }
    for (const auto &l : lhs_tensor.lower()) {
      lhs_tensor_label += orbital_subspaces->label(l.space());
    }

    str_vec.push_back(lhs_tensor_label +
                      " += " + std::format("{:.9f}", rhs_factor().to_double()) +
                      " * np.einsum(");

    std::map<std::string, std::string> index_map;
    std::vector<std::string> unused_indices = {
        "Z", "Y", "X", "W", "V", "U", "T", "S", "R", "Q", "P", "O", "N",
        "M", "L", "K", "J", "I", "H", "G", "F", "E", "D", "C", "B", "A",
        "z", "y", "x", "w", "v", "u", "t", "s", "r", "q", "p", "o", "n",
        "m", "l", "k", "j", "i", "h", "g", "f", "e", "d", "c", "b", "a"};

    std::vector<std::string> indices_vec;
    for (const auto &t : rhs().tensors()) {
      std::string tensor_indices =
          get_unique_tensor_indices(t, index_map, unused_indices);
      indices_vec.push_back(tensor_indices);
    }

    std::vector<std::string> args_vec;
    args_vec.push_back(
        "\"" + join(indices_vec, ",") + "->" +
        get_unique_tensor_indices(lhs_tensor, index_map, unused_indices) +
        "\"");
    for (const auto &t : rhs().tensors()) {
      std::string t_label = t.label() + "[\"";
      for (const auto &l : t.upper()) {
        t_label += orbital_subspaces->label(l.space());
      }
      for (const auto &l : t.lower()) {
        t_label += orbital_subspaces->label(l.space());
      }
      t_label += "\"]";
      args_vec.push_back(t_label);
    }
    str_vec.push_back(join(args_vec, ","));
    str_vec.push_back(",optimize=\"optimal\")");
    return join(str_vec, "");
  }

  if (format == "orca_age") {
    std::string ret;

    std::string lhs_str;

    const auto &lhs_tensor = lhs().tensors()[0];
    std::string lhs_tensor_space;

    // Helper functions
    auto get_spaces = [](const auto &t) -> std::string {
      std::string spaces;
      for (const auto &l : t.upper()) {
        spaces += orbital_subspaces->label(l.space());
      }
      for (const auto &l : t.lower()) {
        spaces += orbital_subspaces->label(l.space());
      }
      return spaces;
    };
    auto wicked2orca_permutation = [](const auto &t) -> std::vector<int> {

      const int n = t.indices().size();
      std::vector<int> indices(n);
      if (n == 2) {
        // NOTE: For some reason singles don't follow this convention
        // returns [0, 1] i.e., noop
        std::iota(indices.begin(), indices.end(), 0);
      } else {
         // returns e.g: [2, 0, 3, 1] and [3, 0, 4, 1, 5, 2]
         for (int i = 0; i < indices.size() / 2; ++i) {
           indices[2 * i] = indices.size() / 2 + i;
           indices[2 * i + 1] = i;
         }
       }
      return indices;
    };
    // TODO: For now hard-coded, could be made generic
    // Also only supports singles and doubles
    std::unordered_map<std::string, std::string> wicked2orca_spaces{
        {"ca", "it"},     {"cv", "ia"},     {"av", "ta"},     {"ccvv", "ijab"},
        {"ccaa", "ijtu"}, {"aavv", "tuab"}, {"cavv", "itab"}, {"ccav", "ijta"},
        {"caaa", "ituv"}, {"aaav", "tuva"}, {"caav", "itua"}};
    std::map<std::string, std::string> index_map;
    std::vector<std::string> unused_indices = {
        "z", "y", "x", "w", "v", "u", "t", "s", "r", "q", "p", "o", "n",
        "m", "l", "k", "j", "i", "h", "g", "f", "e", "d", "c", "b", "a"};

    // Handle LHS
    if (lhs().tensors()[0].indices().size() == 0) {
      lhs_str = "CorrelationEnergy";
    } else {
      lhs_str = "S" + wicked2orca_spaces[get_spaces(lhs().tensors()[0])];
    }
    std::string lhs_indices =
        get_unique_tensor_indices(lhs_tensor, index_map, unused_indices);
    lhs_str += "(";
    std::vector<int> permutation = wicked2orca_permutation(lhs().tensors()[0]);
    for (int i = 0; i < lhs_indices.size(); ++i) {
      lhs_str += lhs_indices[permutation[i]];
      if (i != lhs_indices.size() - 1) {
        lhs_str += ",";
      }
    }
    lhs_str += ")";

    // Handle RHS
    std::vector<std::string> rhs_vec;
    bool has_two_body_integral = false;
    for (const auto &t : rhs().tensors()) {
      std::string rhs_str;
      bool is_onebody_hamiltonian = t.label() == "H" && t.indices().size() == 2;
      bool is_twobody_hamiltonian = t.label() == "H" && t.indices().size() == 4;
      if (is_onebody_hamiltonian) {
        rhs_str += "FT";
      } else if (is_twobody_hamiltonian) {
        rhs_str += "I";
        has_two_body_integral = true;
      } else {
        rhs_str += t.label() + wicked2orca_spaces[get_spaces(t)];
      }
      std::string t_indices =
          get_unique_tensor_indices(t, index_map, unused_indices);
      permutation = wicked2orca_permutation(t);
      rhs_str += "(";
      for (int i = 0; i < t_indices.size(); ++i) {
        // Contracted indices need to be capitalized
        char& idx = t_indices[permutation[i]];
        bool is_contracted_idx = std::find(lhs_indices.begin(), lhs_indices.end(), t_indices[permutation[i]]) == lhs_indices.end();
        if (is_contracted_idx) {
          idx = std::toupper(idx, std::locale());
        }
        rhs_str += t_indices[permutation[i]];
        if (i != t_indices.size() - 1) {
          rhs_str += ",";
        }
      }
      rhs_str += ") ";
      rhs_vec.push_back(rhs_str);
    }
    ret += lhs_str + " += " + std::format("{: .9f}", rhs_factor().to_double()) +
           " " + join(rhs_vec, "");
    if (has_two_body_integral) {
      // Add exchange term
      std::vector<std::string> exchange_vec(rhs_vec);
      for (auto& t : exchange_vec) {
        // Find term corresponding to two body integral
        if (t.starts_with("I(")) {
          std::vector<std::string> indices = split(t.substr(2, t.size() - 4));
          std::swap(indices[1], indices[3]);
          std::string new_t;
          new_t = "I(";
          for (int i = 0; i < indices.size(); ++i) {
            new_t += indices[i];
            if (i != indices.size() - 1) {
              new_t += ",";
            }
          }
          new_t += ") ";
          t = new_t;
        }
      }
      ret += '\n' + lhs_str + " += " + std::format("{: .9f}", -rhs_factor().to_double()) +
             " " + join(exchange_vec, "");
    }
    
    return ret;
  }
  std::string msg = "Equation::compile() - the argument '" + format +
                    "' is not valid. Choices are 'ambit' or 'einsum'";
  throw std::runtime_error(msg);
  return "";
}

std::ostream &operator<<(std::ostream &os, const Equation &eterm) {
  os << eterm.str();
  return os;
}
