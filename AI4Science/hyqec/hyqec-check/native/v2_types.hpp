// Justification data only: no decision or verification algorithms are shared.
#pragma once
#include <algorithm>
#include <compare>
#include <cstdint>
#include <utility>
#include <vector>
namespace cov2 {
struct Action {
    int q, l;
    auto operator<=>(const Action&) const = default;
};
struct Step {
    enum Kind { elim, wiped, odd } kind;
    int g=-1, v=-1;
};
using Derivation = std::vector<Step>;
struct Reason {
    enum Kind { packed, kill, zgen, odd, derivation } kind;
    int index=-1;
    std::vector<int> killers;
    Derivation steps;
    // KILL only: if >= 0, the killer list is Justification::kill_lists[shared] instead of `killers`.
    int shared=-1;
};
// A map kept as a vector sorted by key, with the std::map operations J needs. Appending keys in
// increasing order (emplace_hint at end) costs no search and no per-entry allocation.
template<class K,class V> class FlatMap {
    std::vector<std::pair<K,V>> items;
    auto lower(const K& k) { return std::lower_bound(items.begin(),items.end(),k,[](const auto& e,const K& x){return e.first<x;}); }
    auto lower(const K& k) const { return std::lower_bound(items.begin(),items.end(),k,[](const auto& e,const K& x){return e.first<x;}); }
public:
    using iterator=typename std::vector<std::pair<K,V>>::iterator;
    using const_iterator=typename std::vector<std::pair<K,V>>::const_iterator;
    iterator begin() { return items.begin(); }
    iterator end() { return items.end(); }
    const_iterator begin() const { return items.begin(); }
    const_iterator end() const { return items.end(); }
    size_t size() const { return items.size(); }
    bool empty() const { return items.empty(); }
    void reserve(size_t n) { items.reserve(n); }
    // Takes entries with distinct keys in any order.
    void assign(std::vector<std::pair<K,V>> entries) {
        items=std::move(entries);
        std::sort(items.begin(),items.end(),[](const auto& a,const auto& b){return a.first<b.first;});
    }
    const_iterator find(const K& k) const { auto at=lower(k); return at!=items.end()&&at->first==k?at:items.end(); }
    iterator find(const K& k) { auto at=lower(k); return at!=items.end()&&at->first==k?at:items.end(); }
    V& operator[](const K& k) { auto at=lower(k); if(at==items.end()||k<at->first) at=items.insert(at,{k,V{}}); return at->second; }
    size_t erase(const K& k) { auto at=find(k); if(at==items.end()) return 0; items.erase(at); return 1; }
    iterator emplace_hint(const_iterator,const K& k,V v) {
        if(items.empty()||items.back().first<k) { items.emplace_back(k,std::move(v)); return items.end()-1; }
        auto at=lower(k); if(at!=items.end()&&at->first==k) return at;
        return items.insert(at,{k,std::move(v)});
    }
};
struct Exception {
    std::vector<Action> actions;
    std::vector<uint64_t> y;
};
struct Justification {
    std::vector<int> packing;
    Derivation steps;
    bool z_anchor=false;
    int anchor=-1;
    std::vector<int> killers;
    FlatMap<Action,Reason> reasons;
    std::vector<std::vector<int>> kill_lists;
    // J_S1F (spec 9.2.1), used at S1 filter prunes instead of the anchor and reasons: the packed
    // variables I, one killer list per entry of I aligned with D_i, and the zero list ZL.
    bool filter=false;
    std::vector<int> filter_vars;
    std::vector<std::vector<int>> filter_killers;
    std::vector<int> filter_zeros;
    std::vector<uint64_t> kappa;
    std::vector<Exception> exceptions;
};
}
