// Spec-only pure deciders. Their outputs never authorize a prune until VERIFY.
namespace v2decide {
using cov2::Action;
using cov2::Step;
using cov2::Derivation;
using cov2::Justification;
using cov2::Reason;
// A in increasing index order, from the active bitset.
std::vector<int> active_list(const State& s) {
    std::vector<int> a; a.reserve(s.active_count);
    for(size_t w=0;w<s.active.size();++w) for(U bits=s.active[w];bits;bits&=bits-1) a.push_back(static_cast<int>(64*w+std::countr_zero(bits)));
    return a;
}
// Calls f(q) for every candidate qubit q of generator g, in support order.
template<class F> void for_candidates(const State& s,int g,F f) {
    for(auto e:s.model->rows[g]) if(candidate(s.lab[e.q],e.p)) f(e.q);
}
struct Packing { int np; std::vector<int> pk; };
Packing packing(const State& s) {
    int r=s.model->cap-s.weight;
    auto a=active_list(s);
    std::array<int,96> order; int length=static_cast<int>(std::min<size_t>(a.size(),96));
    std::copy(a.begin(),a.begin()+length,order.begin());
    std::sort(order.begin(),order.begin()+length,[&](int g,int h){return std::pair(s.cnt[g],g)<std::pair(s.cnt[h],h);});
    std::array<U,64> used; std::fill_n(used.begin(),(s.model->n+63)/64,U(0)); // n <= 4096
    std::vector<int> pk; pk.reserve(static_cast<size_t>(std::max(r+1,0)));
    for(int k=0;k<length;++k) {
        int g=order[k];
        if(static_cast<int>(pk.size())==r+1) break;
        bool overlaps=false; for_candidates(s,g,[&](int q){overlaps|=(used[q/64]>>(q%64))&1;});
        if(overlaps) continue;
        pk.push_back(g); for_candidates(s,g,[&](int q){used[q/64]|=U(1)<<(q%64);});
    }
    int lb=static_cast<int>(pk.size()), R=static_cast<int>(a.size());
    if(lb>=r||R>64) return {lb,pk};
    U all=R==64?~U(0):(U(1)<<R)-1;
    std::vector<U> adj(R), masks(s.model->n); std::vector<int> Q; Q.reserve(s.model->n);
    for(int v=0;v<R;++v) for_candidates(s,a[v],[&](int q) {
        if(!masks[q]) Q.push_back(q);
        masks[q]|=U(1)<<v;
    });
    for(int v=0;v<R;++v) {
        for_candidates(s,a[v],[&](int q){adj[v]|=masks[q];});
        adj[v]&=~(U(1)<<v);
    }
    auto mindeg=[&](U X) {
        int u=-1, best=65;
        for(U bits=X;bits;bits&=bits-1) {
            int v=std::countr_zero(bits), d=std::popcount(adj[v]&X);
            if(d<best) { best=d; u=v; if(!d) break; }
        }
        return u;
    };
    auto selected=[&](U bits) {
        std::vector<int> out;
        while(bits&&static_cast<int>(out.size())<r+1) { int v=std::countr_zero(bits); bits&=bits-1; out.push_back(a[v]); }
        return out;
    };
    U X=all, sg=0;
    while(X) { int u=mindeg(X); sg|=U(1)<<u; X&=~((U(1)<<u)|adj[u]); }
    int ng=std::popcount(sg);
    if(ng>lb) { pk=selected(sg); if(ng>=r) return {ng,pk}; lb=ng; }
    U uncovered=all; int cover=0;
    while(uncovered&&cover<r) {
        int best=0, qbest=-1;
        for(int q:Q) { int n=std::popcount(masks[q]&uncovered); if(n>best) {best=n;qbest=q;} }
        if(best<=1) {cover+=std::popcount(uncovered);uncovered=0;break;}
        uncovered&=~masks[qbest]; ++cover;
    }
    int ub=uncovered?r:cover, target=s.model->rule_set==RuleSet::s0?r:r-1;
    if(ub<target||ub<=lb) return {lb,pk};
    int best=lb; U bs=0;
    std::function<void(U,int,U)> search=[&](U x,int cur,U chosen) {
        if(best>=ub) return;
        if(!x) {if(cur>best){best=cur;bs=chosen;}return;}
        if(cur+std::popcount(x)<=best) return;
        int u=mindeg(x), d=std::popcount(adj[u]&x);
        search(x&~((U(1)<<u)|adj[u]),cur+1,chosen|(U(1)<<u));
        if(d>0) search(x&~(U(1)<<u),cur,chosen);
    };
    search(all,0,0);
    return bs?Packing{best,selected(bs)}:Packing{lb,pk};
}
// A value (q, l) flips exactly the generators of q's column whose label at q differs from l.
struct Value { Action a; };
struct System {
    const State& s;
    const Model& m;
    std::vector<int> pk;
    std::vector<std::vector<Value>> d;
    std::vector<U> F;
    std::vector<bool> packed;
    System(const State& state,const std::vector<int>& packing):s(state),m(*s.model),pk(packing),F(m.m),packed(m.m) {
        d.reserve(pk.size());
        for(size_t i=0;i<pk.size();++i) {
            int g=pk[i]; packed[g]=true; std::vector<Value> vals; vals.reserve(2*m.rows[g].size());
            for(auto e:m.rows[g]) if(candidate(s.lab[e.q],e.p))
                for(int l=1;l<=3;++l) if(l!=e.p&&(s.lab[e.q]==4||s.lab[e.q]==4+l)) {
                    for(auto c:m.columns[e.q]) if(c.p!=l) F[c.q]|=U(1)<<i;
                    vals.push_back({{e.q,l}});
                }
            d.push_back(std::move(vals));
        }
    }
    std::vector<int> flips(Action a) const {
        std::vector<int> out;
        for(auto e:m.columns[a.q]) if(e.p!=a.l) out.push_back(e.q);
        return out;
    }
    bool flip(const Value& c,int g) const {
        const auto& column=m.columns[c.a.q];
        auto e=std::lower_bound(column.begin(),column.end(),g,[](Entry e,int g){return e.q<g;});
        return e!=column.end()&&e->q==g&&e->p!=c.a.l;
    }
    std::vector<Action> actions() const {
        std::vector<Action> out;
        for(int q=0;q<m.n;++q) for(int l=1;l<=3;++l) if(s.lab[q]==4||s.lab[q]==4+l) out.push_back({q,l});
        return out;
    }
    size_t choices() const {size_t n=0;for(const auto& v:d)n+=v.size();return n;}
};
struct Unit {
    std::vector<int> zeros,wiped;
    std::vector<std::pair<int,int>> needs;
    std::vector<std::vector<int>> first;
    // Indices into System::d, in the modified domain's canonical order.
    std::vector<std::vector<size_t>> values;
    std::vector<U> F;
    std::vector<uint8_t> target;
    // Full counts of zero generators and need entries in ODD (the lists keep at most 96).
    int zero_count=0, need_count=0;
    bool refuted() const {return !zeros.empty()||!wiped.empty();}
};
Unit unit(const System& c,const std::vector<int>& ef={},int qe=-1) {
    Unit u; u.target=c.s.syndrome; u.first.reserve(c.d.size()); u.values.reserve(c.d.size());
    for(int g:ef) u.target[g]^=1;
    // Without an excluded qubit the flipper sets are the System's.
    if(qe<0) u.F=c.F;
    else {
        u.F.assign(c.m.m,0);
        for(size_t i=0;i<c.d.size();++i) for(const auto& v:c.d[i]) if(v.a.q!=qe)
            for(auto e:c.m.columns[v.a.q]) if(e.p!=v.a.l) u.F[e.q]|=U(1)<<i;
    }
    std::vector<int> odd; odd.reserve(static_cast<size_t>(c.s.active_count)+ef.size());
    for(size_t w=0;w<c.s.active.size();++w) for(U bits=c.s.active[w];bits;bits&=bits-1) {
        int g=static_cast<int>(64*w+std::countr_zero(bits)); if(!c.packed[g]&&u.target[g]) odd.push_back(g);
    }
    for(int g:ef) if(!c.s.syndrome[g]&&!c.packed[g]) odd.push_back(g);
    for(int g:odd) {
        if(!u.F[g]) {++u.zero_count; if(u.zeros.size()<96) u.zeros.push_back(g);}
        else if(std::popcount(u.F[g])==1) {++u.need_count; if(u.needs.size()<96) u.needs.emplace_back(g,std::countr_zero(u.F[g]));}
    }
    for(size_t i=0;i<c.d.size();++i) {
        std::vector<int> fk; std::vector<size_t> values; bool alive=false;
        fk.reserve(c.d[i].size()); values.reserve(c.d[i].size());
        for(size_t k=0;k<c.d[i].size();++k) {
            const auto& v=c.d[i][k];
            if(v.a.q==qe) continue; // Excluded values have no UNIT record in draft-3.
            values.push_back(k);
            int killer=-1;
            for(auto e:c.m.columns[v.a.q]) if(e.q!=c.pk[i]&&(e.p!=v.a.l)!=bool(u.target[e.q])&&!(u.F[e.q]&~(U(1)<<i))) {killer=e.q;break;}
            if(killer==-1) for(auto [g,j]:u.needs) if(j==static_cast<int>(i)&&!c.flip(v,g)) {killer=g;break;}
            fk.push_back(killer); alive|=killer==-1;
        }
        if(!alive) u.wiped.push_back(static_cast<int>(i));
        u.first.push_back(std::move(fk));
        u.values.push_back(std::move(values));
    }
    return u;
}
Derivation unit_derivation(const System& c,const Unit& u) {
    require(u.refuted(),"unit derivation requested without refutation");
    if(!u.zeros.empty()) return {{Step::odd,u.zeros[0],-1}};
    int i=u.wiped[0]; std::vector<bool> alive(u.values[i].size(),true);
    Derivation steps;
    for(size_t k=0;k<alive.size();++k) if(alive[k]) {
        int g=u.first[i][k]; require(g>=0,"missing unit killer");
        steps.push_back({Step::elim,g,i});
        for(size_t j=0;j<alive.size();++j) if(c.flip(c.d[i][u.values[i][j]],g)!=bool(u.target[g])) alive[j]=false;
    }
    steps.push_back({Step::wiped,-1,i}); return steps;
}
// Test-only: the first PROP limit, in 7.2's order, that gave up (empty if none).
struct PropTrace { std::vector<const char*> limits; };
bool prop(const System& c,Derivation& proof,PropTrace* trace=nullptr) {
    if(c.choices()>1024) { if(trace) trace->limits.push_back("prop_choices_gt_1024"); return false; }
    std::set<int> local; size_t occurrences=0;
    for(const auto& d:c.d) for(const auto& v:d) for(auto e:c.m.columns[v.a.q]) if(e.p!=v.a.l) {local.insert(e.q);++occurrences;}
    if(trace&&local.size()>2048) trace->limits.push_back("prop_local_generators_gt_2048");
    else if(trace&&occurrences>6144) trace->limits.push_back("prop_occurrences_gt_6144");
    if(local.size()>2048||occurrences>6144) return false;
    for(int g:active_list(c.s)) if(!local.contains(g)) {proof={{Step::odd,g,-1}};return true;}
    auto alive=c.d;
    bool changed=true;
    while(changed) {
        changed=false;
        for(int g:local) {
            int mixed=0,mv=-1; bool parity=false;
            for(size_t i=0;i<alive.size();++i) {
                int count=0; for(const auto& v:alive[i]) count+=c.flip(v,g);
                if(count==static_cast<int>(alive[i].size())) parity^=1;
                else if(count) {++mixed;mv=static_cast<int>(i);}
            }
            if(mixed>1) continue;
            if(!mixed) {
                if(parity!=bool(c.s.syndrome[g])) {proof.push_back({Step::odd,g,-1});return true;}
            } else {
                bool want=bool(c.s.syndrome[g])^parity;
                auto& d=alive[mv]; size_t before=d.size();
                std::erase_if(d,[&](const Value& v){return c.flip(v,g)!=want;});
                if(d.size()!=before) {changed=true;proof.push_back({Step::elim,g,mv});}
                if(d.empty()) {proof.push_back({Step::wiped,-1,mv});return true;}
            }
        }
    }
    proof.clear(); return false;
}
bool s0(const System& c,Justification& j) {
    Unit u=unit(c);
    if(u.refuted()) {j.steps=unit_derivation(c,u);return true;}
    return prop(c,j.steps);
}
// Test-only record of how S1 ended (the vectors' stop, step-6 counts and first survivor).
struct S1Trace {
    const char* stop=""; int tests[6]={}; Action survivor{-1,-1};
    Unit unit; size_t values=0, qstar=0; long killer_total=-1;
    bool z_anchor=false; int anchor=-1; std::vector<int> killers, base;
};
bool s1(const System& c,Justification& j,S1Trace* trace=nullptr) {
    auto stop=[&](const char* why,bool pruned) { if(trace) trace->stop=why; return pruned; };
    Unit u=unit(c);
    if(trace) { trace->unit=u; trace->values=c.choices(); }
    if(!u.refuted()) return stop("unit_not_refuted",false);
    if(c.choices()>1024) return stop("values_gt_1024",false);
    j.steps=unit_derivation(c,u);
    // Q* (step 3): the undecided qubits in the killer support of every wiped variable and in
    // every recorded zero generator. hits[q] == k after k conditions iff q is in all of them.
    std::vector<int> hits(c.m.n); int conditions=0;
    auto restrict_to=[&](const std::vector<int>& gs) {
        for(int g:gs) for(auto e:c.m.rows[g]) if(hits[e.q]==conditions) hits[e.q]=conditions+1;
        ++conditions;
    };
    for(int i:u.wiped) restrict_to(u.first[i]);
    for(int g:u.zeros) restrict_to({g});
    bool filtered=true;
    for(int q=0;q<c.m.n;++q) if(c.s.lab[q]>=4&&hits[q]==conditions) {filtered=false;if(!trace)break;if(trace)++trace->qstar;}
    if(trace) {  // the step-2 anchor and base list, which the trace reports also at a filter prune
        trace->z_anchor=!u.zeros.empty();
        if(trace->z_anchor) {trace->anchor=u.zeros[0];trace->base={u.zeros[0]};}
        else {
            int bv=u.wiped[0]; for(int i:u.wiped) if(c.d[i].size()<c.d[bv].size()) bv=i;
            trace->anchor=bv; trace->killers=u.first[bv];
            for(int g:trace->killers) if(std::find(trace->base.begin(),trace->base.end(),g)==trace->base.end()) trace->base.push_back(g);
        }
    }
    // Filter prune: J_S1F (9.2.1) is I = the wiped variables, K_i = their first killers, ZL = Z.
    if(filtered) {
        j.filter=true; j.filter_vars=u.wiped; j.filter_zeros=u.zeros;
        for(int i:u.wiped) j.filter_killers.push_back(u.first[i]);
        return stop("filter_prune",true);
    }
    std::vector<int> base;
    if(!u.zeros.empty()) {j.z_anchor=true;j.anchor=u.zeros[0];base={j.anchor};}
    else {
        int bv=u.wiped[0];
        for(int i:u.wiped) if(c.d[i].size()<c.d[bv].size()) bv=i;
        j.anchor=bv; j.killers=u.first[bv];
        for(int g:j.killers) if(std::find(base.begin(),base.end(),g)==base.end()) base.push_back(g);
    }
    // The actions E of step 6, in its order: each g of B, each of its support positions with an
    // undecided qubit q, each l = 1, 2, 3 in dom(q) with l != g_q, skipping actions already considered
    // (considered[q] holds their labels).
    std::vector<uint8_t> considered(c.m.n);
    std::vector<Action> anchor_actions; anchor_actions.reserve(3*static_cast<size_t>(c.m.n));
    for(int g:base) for(auto e:c.m.rows[g]) if(c.s.lab[e.q]>=4)
        for(int l=1;l<=3;++l) if(l!=e.p&&(c.s.lab[e.q]==4||c.s.lab[e.q]==4+l)&&!((considered[e.q]>>l)&1)) {
            considered[e.q]|=1<<l; anchor_actions.push_back({e.q,l});
        }

    // K(c) of step 4, in order: c's column killers, then the recorded needs of variable i that c
    // does not flip. f(g) returns false to stop. The lists are enumerated, never stored.
    auto all_killers=[&](int i,const Value& v,auto&& f) {
        for(auto e:c.m.columns[v.a.q])
            if(e.q!=c.pk[i]&&(e.p!=v.a.l)!=bool(c.s.syndrome[e.q])&&!(c.F[e.q]&~(U(1)<<i))&&!f(e.q)) return;
        for(auto [g,vi]:u.needs) if(vi==i&&!c.flip(v,g)&&!f(g)) return;
    };
    size_t total=0;
    for(int i:u.wiped) for(const auto& v:c.d[i]) all_killers(i,v,[&](int){++total;return true;});
    if(trace) trace->killer_total=static_cast<long>(total);
    if(total>=6144) return stop("killers_ge_6144",false);
    if(c.pk.size()>31) return stop("h_gt_31",false);
    // Test 6.2 per generator: km[g] marks the wiped variables (by position in u.wiped) with g in KM(i).
    std::vector<U> km(c.m.m);
    for(size_t w=0;w<u.wiped.size();++w) for(int g:u.first[u.wiped[w]]) km[g]|=U(1)<<w;
    auto killed_by=[&](int g) {return km[g];};
    std::vector<int> shared(u.wiped.size(),-1);
    std::vector<std::pair<Action,Reason>> reasons; reasons.reserve(anchor_actions.size());
    std::array<int,128> ef; int nef=0;
    auto has=[&](int g){return std::binary_search(ef.begin(),ef.begin()+nef,g);};
    for(auto a:anchor_actions) {
        nef=0; for(auto e:c.m.columns[a.q]) if(e.p!=a.l) ef[nef++]=e.q;
        Reason reason{Reason::packed,-1,{},{}}; bool excluded=false; int step=0;
        for(int k=0;k<nef;++k) if(c.packed[ef[k]]) {reason={Reason::packed,ef[k],{},{}};excluded=true;break;}
        if(!excluded) {
            step=1;
            U hit=0; for(int k=0;k<nef;++k) hit|=killed_by(ef[k]);
            for(size_t w=0;w<u.wiped.size();++w) if(!((hit>>w)&1)) {
                // The first-killer list of a wiped variable is shared by all actions using it.
                if(shared[w]<0) {shared[w]=static_cast<int>(j.kill_lists.size());j.kill_lists.push_back(u.first[u.wiped[w]]);}
                reason={Reason::kill,u.wiped[w],{},{}};reason.shared=shared[w];excluded=true;break;
            }
        }
        if(!excluded) { step=2; for(int z:u.zeros) if(!has(z)) {reason={Reason::zgen,z,{},{}};excluded=true;break;} }
        if(!excluded) { step=3; for(int k=0;k<nef;++k) if(!c.s.syndrome[ef[k]]&&!c.F[ef[k]]) {reason={Reason::odd,ef[k],{},{}};excluded=true;break;} }
        if(!excluded) step=4;
        if(!excluded) for(int i:u.wiped) {
            std::vector<int> ks;
            for(const auto& v:c.d[i]) {
                int unflipped=-1; all_killers(i,v,[&](int g){if(has(g)) return true; unflipped=g; return false;});
                if(unflipped<0) break;
                ks.push_back(unflipped);
            }
            if(ks.size()==c.d[i].size()) {reason={Reason::kill,i,ks,{}};excluded=true;break;}
        }
        if(!excluded) step=5;
        if(trace) ++trace->tests[step];
        if(!excluded) {
            auto modified=unit(c,std::vector<int>(ef.begin(),ef.begin()+nef),a.q);
            if(!modified.refuted()) { if(trace) trace->survivor=a; return stop("E_survives",false); }
            reason={Reason::derivation,-1,{},unit_derivation(c,modified)};
        }
        reasons.emplace_back(a,std::move(reason));
    }
    j.reasons.assign(std::move(reasons));
    return stop("all_E_excluded",true);
}
// Reproducible S2 functionals (spec 10.2): nonpivot g gets mix(g + seed).
constexpr U functional_seed=0xd1b54a32d192ed03ULL;
using Bits=std::vector<U>;
bool bit(const U* a,int k) {return (a[k/64]>>(k%64))&1;}
// Echelon basis of W: distinct leading (highest set) bits, row_of[p] = the row led by p or -1.
struct Reduction {
    int m,words;
    std::vector<U> rows;
    std::vector<int> row_of;
    explicit Reduction(int size):m(size),words((size+63)/64),row_of(size,-1) {}
    const U* row(int k) const {return rows.data()+static_cast<size_t>(k)*words;}
    void add(U* a,int k) const {const U* b=row(k);for(int w=0;w<words;++w)a[w]^=b[w];}
    int lead(const U* a) const {for(int w=words-1;w>=0;--w) if(a[w]) return 64*w+63-std::countl_zero(a[w]); return -1;}
    // Cancels leading pivots; returns the first leading bit that is not a pivot, or -1 when a is in W.
    int eliminate(U* a) const {
        for(int p=lead(a);p>=0;p=lead(a)) { if(row_of[p]<0) return p; add(a,row_of[p]); }
        return -1;
    }
    // Clears every pivot bit (pivots in decreasing order); the remainder is zero iff a is in W.
    void reduce(U* a) const {for(int p=m-1;p>=0;--p) if(row_of[p]>=0&&bit(a,p)) add(a,row_of[p]);}
    bool zero(const U* a) const {return std::all_of(a,a+words,[](U v){return v==0;});}
    void insert(U* a) {
        int p=eliminate(a); if(p<0) return;
        row_of[p]=static_cast<int>(rows.size()/words); rows.insert(rows.end(),a,a+words);
    }
    // The unique functional vanishing on W with the given nonpivot values. A row's other bits lie
    // below its leading bit, so pivots are back-substituted in increasing order.
    std::vector<U> functionals(int coordinate=-1) const {
        std::vector<U> keys(m);
        for(int k=0;k<m;++k) if(row_of[k]<0) keys[k]=coordinate<0?mix(static_cast<U>(k)+functional_seed):U(k==coordinate);
        for(int p=0;p<m;++p) if(row_of[p]>=0) {
            const U* b=row(row_of[p]); U key=0;
            for(int w=0;w<words;++w) for(U bits=b[w];bits;bits&=bits-1) {
                int g=64*w+std::countr_zero(bits); if(g!=p) key^=keys[g];
            }
            keys[p]=key;
        }
        return keys;
    }
};
// Exact decision (10.1) with a fingerprint filter: kappa vanishes on W, so every E with
// sigma(E) XOR tau in W has kappa(sigma(E)) == kappa(tau). Only such E are reduced exactly;
// the ones that are not members are the exceptions of J_S2.
// Test-only record of how S2 ended (the vectors' rank_W, action count and stop).
struct S2Trace { const char* stop=""; long rank=-1, actions=-1; };
bool s2(const System& c,Justification& j,S2Trace* trace=nullptr) {
    auto stop=[&](const char* why,bool pruned) { if(trace) trace->stop=why; return pruned; };
    if(c.m.n>1365) return stop("n_gt_1365",false);
    for(const auto& d:c.d) if(d.empty()) return stop("empty_value_list",false);
    Reduction red(c.m.m);
    auto sigma=[&](U* v,Action a) {for(auto e:c.m.columns[a.q]) if(e.p!=a.l) v[e.q/64]^=U(1)<<(e.q%64);};
    Bits tau(red.words), v(red.words);
    std::copy(c.s.active.begin(),c.s.active.end(),tau.begin()); // sigma(P_v): the active generators
    for(const auto& d:c.d) {
        sigma(tau.data(),d[0].a);
        for(size_t k=1;k<d.size();++k) {
            std::fill(v.begin(),v.end(),0); sigma(v.data(),d[k].a); sigma(v.data(),d[0].a); red.insert(v.data());
        }
    }
    if(trace) trace->rank=static_cast<long>(red.rows.size()/red.words);
    v=tau; if(red.eliminate(v.data())<0) return stop("survives",false);
    j.kappa=red.functionals();
#ifdef HYQEC_V2_ZERO_KAPPA
    j.kappa.assign(c.m.m,0); // Test exact exception path; absent from release builds.
#endif
    // Exact test of sigma(E) XOR tau: a member means no prune, otherwise (E, y) is an exception.
    auto member=[&](const std::vector<Action>& es) {
        v=tau; for(auto a:es) sigma(v.data(),a);
        red.reduce(v.data());
        int coordinate=-1;
        for(int w=0;w<red.words;++w) if(v[w]) {coordinate=64*w+std::countr_zero(v[w]);break;}
        if(coordinate<0) return true;
        auto ys=red.functionals(coordinate); Bits y(red.words);
        for(int g=0;g<c.m.m;++g) if(ys[g]) y[g/64]|=U(1)<<(g%64);
        j.exceptions.push_back({es,std::move(y)});
        return false;
    };
    U gamma=0; for(int g=0;g<c.m.m;++g) if(bit(tau.data(),g)) gamma^=j.kappa[g];
    if(!gamma&&member({})) return stop("survives",false);
    // Fingerprints of the actions, in action order. kappa(sigma(q, l)) is the XOR of q's column
    // keys minus those with label l at q, so one pass over the column serves all three labels.
    std::vector<Action> actions(3*static_cast<size_t>(c.m.n)); std::vector<U> keys(actions.size()); size_t count=0;
    for(int q=0;q<c.m.n;++q) if(c.s.lab[q]>=4) {
        U total=0, same[4]={};
        for(auto e:c.m.columns[q]) {total^=j.kappa[e.q];same[e.p]^=j.kappa[e.q];}
        for(int l=1;l<=3;++l) if(c.s.lab[q]==4||c.s.lab[q]==4+l) {actions[count]={q,l};keys[count++]=total^same[l];}
    }
    actions.resize(count); keys.resize(count);
    if(trace) trace->actions=static_cast<long>(actions.size());
    for(size_t k=0;k<actions.size();++k) if(keys[k]==gamma&&member({actions[k]})) return stop("survives",false);
    // Actions grouped by fingerprint: an open-addressing table, at most half full, holds 1 + the
    // first action of each distinct key (and the key's high half, to skip most mismatches); next[]
    // chains the actions sharing it in increasing order. Keys are already pseudo-random.
    size_t size=2; while(size<2*keys.size()) size*=2;
    std::vector<uint32_t> head(size), tag(size), next(keys.size());
    auto slot=[&](U value) {
        uint32_t high=static_cast<uint32_t>(value>>32); size_t i=(value^high)&(size-1);
        while(head[i]&&!(tag[i]==high&&keys[head[i]-1]==value)) i=(i+1)&(size-1);
        return i;
    };
    for(size_t k=keys.size();k-->0;) {
        size_t i=slot(keys[k]); next[k]=head[i]; head[i]=static_cast<uint32_t>(k+1); tag[i]=static_cast<uint32_t>(keys[k]>>32);
    }
    for(size_t k=0;k<actions.size();++k)
        for(uint32_t b=head[slot(keys[k]^gamma)];b&&b-1<k;b=next[b-1])
            if(actions[b-1].q!=actions[k].q&&member({actions[b-1],actions[k]})) return stop("survives",false);
    return stop("prune",true);
}
// Greedy rule 2: J_P is the selected packing, verified before the decision is returned.
Decision checked_packing(const State& s,const Decision& d) {
    size_t size=static_cast<size_t>(d.size);
#ifdef HYQEC_V2_WRONG_J
    size=0; // Test-build-only injection, absent from release builds.
#endif
    v2verify::verify_packing_prune(s,d.packing.data(),size);
    return d;
}
Decision checked(const State& s,int rule,Justification& j) {
#ifdef HYQEC_V2_WRONG_J
    // Test-build-only injection, absent from release builds.
    j.packing.clear();
    if(rule==7) j.kappa.clear();
#endif
    v2verify::verify(s,rule,j);
    Decision out;out.rule=rule;out.size=static_cast<int>(j.packing.size());
    std::copy(j.packing.begin(),j.packing.end(),out.packing.begin());return out;
}
}
Decision decide(const State& s) {
    require(s.dead_count==0,"entered a dead state");
    if(!s.active_count) {
        require(s.in_stabilizers(),"nontrivial logical leaf");return {};
    }
    if(s.weight+1>s.model->cap) {Decision d;d.rule=1;return d;}
    const RuleSet rules=s.model->rule_set;
    if(rules==RuleSet::greedy) {
        Decision d=greedy_decide(s);
        return d.rule==2?v2decide::checked_packing(s,d):d;
    }
    int r=s.model->cap-s.weight;
    if(s.active_count>=r) {
        auto p=v2decide::packing(s);cov2::Justification j;j.packing=p.pk;
        if(p.np>r) return v2decide::checked(s,2,j);
        if(p.np==r || (p.np==r-1&&r>=2&&rules!=RuleSet::s0) || (p.np==r-2&&rules==RuleSet::s0s1s2)) {
            v2decide::System sys(s,p.pk);int rule=0;
            if(rules==RuleSet::s0s1s2&&p.np==r-2) {if(v2decide::s2(sys,j))rule=7;}
            else if(rules!=RuleSet::s0&&p.np==r-1&&r>=2) {if(v2decide::s1(sys,j))rule=6;}
            else if(p.np==r) {if(v2decide::s0(sys,j))rule=5;}
            if(rule) return v2decide::checked(s,rule,j);
        }
    }
    Decision d; d.rule=3; d.generator=branch_generator(s); return d;
}
