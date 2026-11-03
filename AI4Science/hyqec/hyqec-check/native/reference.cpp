// Independent reference algorithm from SOUNDNESS.md and README prose only.
// Deterministic choices here are LOCAL POLICY, pending BNB_CERT_SPEC.md.
#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>
#include <omp.h>

namespace {
using Word = uint64_t;
using Mask = uint8_t;
constexpr Mask identity = 1, free_domain = 15;
// Label code = x + 2z: I=0, X=1, Z=2, Y=3.
int label(char c) {
    auto p = std::string("IXZY").find(c);
    if (p == std::string::npos) throw std::runtime_error("invalid Pauli");
    return static_cast<int>(p);
}
Mask domain(char c) {
    if (c == '*') return free_domain;
    if (c == 'x') return identity | (1 << 1);
    if (c == 'z') return identity | (1 << 2);
    if (c == 'y') return identity | (1 << 3);
    return 1 << label(c);
}
bool anticommutes(int a, int b) { return a != 0 && b != 0 && a != b; }
bool undecided(Mask a) { return std::popcount(a) > 1; }
struct Entry { int qubit, pauli; };
struct Model {
    int n = 0, cap = 0, threads = 0;
    std::vector<std::vector<Entry>> generators, logicals;
    std::vector<std::array<std::vector<Word>, 4>> toggle;
    std::vector<std::vector<Mask>> roots;
    size_t syndrome_words() const { return (generators.size() + 63) / 64; }
    bool logical(const std::vector<Mask>& state) const {
        for (const auto& row : logicals) {
            bool parity = false;
            for (auto e : row) {
                Mask a = state[e.qubit];
                if (!undecided(a)) parity ^= anticommutes(std::countr_zero(a), e.pauli);
            }
            if (parity) return true;
        }
        return false;
    }
};
struct Candidate {
    int generator;
    int children = 0;
    std::vector<int> qubits;
};
struct Statistics {
    Word nodes = 0, dead = 0, leaves = 0, bound = 0, branches = 0;
    int best = std::numeric_limits<int>::max();
    std::string witness;
};
void increment(Word& value) {
    if (value == std::numeric_limits<Word>::max()) throw std::runtime_error("count overflow");
    ++value;
}

// Straightforward state-copy DFS: a semantic oracle, not the large-tree engine.
void visit(const Model& model, const std::vector<Mask>& state,
           const std::vector<Word>& syndrome, int weight, Statistics& stats) {
    increment(stats.nodes);
    if (weight > model.cap) { increment(stats.bound); return; }
    std::vector<Candidate> active;
    for (size_t block = 0; block < syndrome.size(); ++block) {
        Word remaining = syndrome[block];
        while (remaining) {
            int g = static_cast<int>(block * 64 + std::countr_zero(remaining));
            remaining &= remaining - 1;
            Candidate candidate{g, 0, {}};
            for (auto entry : model.generators[g]) {
                Mask a = state[entry.qubit];
                if (!undecided(a)) continue;
                Mask options = a & (free_domain ^ identity ^ (1 << entry.pauli));
                if (options) {
                    candidate.qubits.push_back(entry.qubit);
                    candidate.children += std::popcount(options);
                }
            }
            if (candidate.qubits.empty()) { increment(stats.dead); return; }
            active.push_back(std::move(candidate));
        }
    }
    if (active.empty()) {
        increment(stats.leaves);
        if (model.logical(state) && weight < stats.best) {
            stats.best = weight;
            stats.witness.clear();
            for (Mask a : state) stats.witness += undecided(a) ? 'I' : "IXZY"[std::countr_zero(a)];
        }
        return;
    }
    std::vector<size_t> order(active.size());
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) {
        return active[a].qubits.size() < active[b].qubits.size();
    });
    std::vector<bool> packed(model.n, false);
    int bound = weight;
    for (size_t i : order) {
        bool intersects = false;
        for (int q : active[i].qubits) if (packed[q]) { intersects = true; break; }
        if (!intersects) {
            if (++bound > model.cap) { increment(stats.bound); return; }
            for (int q : active[i].qubits) packed[q] = true;
        }
    }
    increment(stats.branches);
    auto chosen = std::min_element(active.begin(), active.end(), [](const auto& a, const auto& b) {
        return a.children < b.children;
    });
    auto next = state;
    for (auto entry : model.generators[chosen->generator]) {
        Mask a = next[entry.qubit];
        if (!undecided(a)) continue;
        Mask options = a & (free_domain ^ identity ^ (1 << entry.pauli));
        for (int option = 1; option <= 3; ++option) {
            if (!(options & (1 << option))) continue;
            next[entry.qubit] = 1 << option;
            auto child_syndrome = syndrome;
            for (size_t b = 0; b < syndrome.size(); ++b)
                child_syndrome[b] ^= model.toggle[entry.qubit][option][b];
            visit(model, next, child_syndrome, weight + 1, stats);
        }
        // Exclude all previous anticommuting options, retaining commuting ones.
        next[entry.qubit] = a & (identity | (1 << entry.pauli));
    }
}
std::vector<Entry> read_pauli(int n) {
    std::string row;
    if (!(std::cin >> row) || row.size() != static_cast<size_t>(n))
        throw std::runtime_error("invalid Pauli row length");
    std::vector<Entry> result;
    for (int q = 0; q < n; ++q) if (int p = label(row[q])) result.push_back({q, p});
    return result;
}
Model read_model() {
    Model model;
    int m, l, r;
    if (!(std::cin >> model.n >> m >> l >> model.cap >> model.threads >> r)
        || model.n < 1 || model.n > 100000 || m < 0 || l < 0 || r < 0
        || m > 100000 || l > 200000 || r > 300000
        || model.cap < 0 || model.cap > model.n || model.threads < 1 || model.threads > 192)
        throw std::runtime_error("invalid reference model header");
    for (int i = 0; i < m; ++i) model.generators.push_back(read_pauli(model.n));
    for (int i = 0; i < l; ++i) model.logicals.push_back(read_pauli(model.n));
    model.toggle.resize(model.n);
    for (auto& qubit : model.toggle)
        for (auto& option : qubit) option.resize(model.syndrome_words());
    for (int g = 0; g < m; ++g)
        for (auto e : model.generators[g])
            for (int p = 1; p <= 3; ++p)
                if (anticommutes(e.pauli, p)) model.toggle[e.qubit][p][g / 64] |= Word(1) << (g % 64);
    for (int i = 0; i < r; ++i) {
        std::string raw;
        if (!(std::cin >> raw) || raw.size() != static_cast<size_t>(model.n))
            throw std::runtime_error("invalid root domain length");
        std::vector<Mask> state;
        for (char c : raw) state.push_back(domain(c));
        model.roots.push_back(std::move(state));
    }
    std::string excess;
    if (std::cin >> excess) throw std::runtime_error("trailing reference protocol tokens");
    return model;
}
}

int main() {
    try {
        auto model = read_model();
        auto started = std::chrono::steady_clock::now();
        std::vector<Statistics> per_root(model.roots.size());
        std::vector<std::string> errors(model.roots.size());
        #pragma omp parallel for schedule(dynamic) num_threads(model.threads)
        for (size_t i = 0; i < model.roots.size(); ++i) {
            try {
                const auto& state = model.roots[i];
                std::vector<Word> syndrome(model.syndrome_words());
                int weight = 0;
                for (int q = 0; q < model.n; ++q) {
                    if (undecided(state[q]) || state[q] == identity) continue;
                    ++weight;
                    int p = std::countr_zero(state[q]);
                    for (size_t b = 0; b < syndrome.size(); ++b) syndrome[b] ^= model.toggle[q][p][b];
                }
                visit(model, state, syndrome, weight, per_root[i]);
            } catch (const std::exception& error) { errors[i] = error.what(); }
        }
        for (const auto& error : errors) if (!error.empty()) throw std::runtime_error(error);
        double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
        std::cout << "{\"mode\":\"reference-search\",\"certificate_verified\":false,\"wall_seconds\":"
                  << elapsed << ",\"roots\":[";
        for (size_t i = 0; i < per_root.size(); ++i) {
            auto& s = per_root[i];
            if (i) std::cout << ',';
            std::cout << "{\"nodes\":" << s.nodes << ",\"dead\":" << s.dead << ",\"leaves\":"
                      << s.leaves << ",\"bound\":" << s.bound << ",\"branches\":" << s.branches
                      << ",\"witness\":\"" << s.witness << "\"}";
        }
        std::cout << "]}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "reference engine error: " << error.what() << '\n';
        return 2;
    }
}
