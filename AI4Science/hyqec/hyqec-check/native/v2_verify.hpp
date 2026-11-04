// Trusted local justification checks, independently rebuilding from lab and the instance rows/columns.
// This file must not call the decider or use its cached syndrome/domains/basis.
namespace v2verify {
using cov2::Action;
using cov2::Justification;
void check(bool b) { require(b,"v2 prune justification failed"); }
// VERIFY_PACKING (6.3): exactly `need` listed generators, pairwise distinct, active, each with a
// candidate, and with pairwise disjoint candidate sets. This is all of VERIFY for rule 2.
void verify_packing(const State& s,const int* pk,size_t size,int need) {
    const Model& m=*s.model;
    check(need>=0&&size==static_cast<size_t>(need));
    std::array<U,64> claimed; std::fill_n(claimed.begin(),(m.n+63)/64,U(0)); // n <= 4096
    for(size_t i=0;i<size;++i) {
        int g=pk[i];
        check(g>=0&&g<m.m&&std::find(pk,pk+i,g)==pk+i);
        bool active=false, any=false;
        for(auto e:m.rows[g]) {
            int a=s.lab[e.q];
            active^=a>=1&&a<=3&&a!=e.p;
            // q is a candidate iff dom(q) has a non-I label other than g_q (section 7 values).
            if(a==4||(a>4&&a!=4+e.p)) {
                U bit=U(1)<<(e.q%64); check(!(claimed[e.q/64]&bit)); claimed[e.q/64]|=bit; any=true;
            }
        }
        check(active&&any);
    }
}
// One Local per prune, built from the node state alone.
struct Local {
    const Model* model=nullptr;
    const std::vector<uint8_t>* labels_of=nullptr;
    std::vector<int> packed;
    std::vector<std::vector<Action>> domains;
    // F(g) as a mask over packed variables (|PK| <= cap + 1 <= 47).
    std::vector<U> flippers;
    std::vector<int> owner;
    int remaining=0;
    // Syndrome bits s(g), computed from lab on first use.
    mutable std::vector<uint8_t> synd_known, synd_value;
    mutable std::vector<uint8_t> alive;
    std::vector<size_t> offset;
    mutable std::vector<Action> scratch_actions;
    // Per variable, a killer list already checked at this node. Killer validity does not depend
    // on the action, and S1 repeats the same list for many actions.
    mutable std::vector<std::vector<int>> checked_kills;
    // kill_marks[g] bit i: g is in checked_kills[i].
    mutable std::vector<U> kill_marks;
    const Model& m() const { return *model; }
    uint8_t lab(int q) const { return (*labels_of)[q]; }
    bool synd(int g) const {
        if(!synd_known[g]) {
            uint8_t parity=0;
            for(auto e:m().rows[g]) parity^=lab(e.q)>=1&&lab(e.q)<=3&&lab(e.q)!=e.p;
            synd_value[g]=parity; synd_known[g]=1;
        }
        return synd_value[g];
    }
    bool flips(Action a,int g) const {
        check(g>=0&&g<m().m);
        const auto& row=m().rows[g];
        auto e=std::lower_bound(row.begin(),row.end(),a.q,[](Entry e,int q){return e.q<q;});
        return e!=row.end()&&e->q==a.q&&e->p!=a.l;
    }
    bool allowed(Action a) const {
        return a.q>=0&&a.q<m().n&&a.l>=1&&a.l<=3&&
            (lab(a.q)==4 || lab(a.q)==4+a.l);
    }
    Local(const State& s,const std::vector<int>& pk,int need,bool parity=true):model(s.model),labels_of(&s.lab) {
        const Model& mm=m();
        int weight=0, largest=0;
        for(int q=0;q<mm.n;++q) { int a=lab(q); largest=std::max(largest,a); weight+=(a>=1&&a<=3); }
        check(largest<=7);
        remaining=mm.cap-weight;
        verify_packing(s,pk.data(),pk.size(),need);
        if(!parity) return;
        packed=pk;
        synd_known.assign(mm.m,0); synd_value.assign(mm.m,0);
        owner.assign(mm.n,-1);
        // The value lists D_i (section 7); owner marks the verified, disjoint candidate qubits.
        domains.resize(pk.size());
        for(size_t i=0;i<pk.size();++i) {
            auto& values=domains[i]; values.clear(); values.reserve(2*mm.rows[pk[i]].size());
            for(auto e:mm.rows[pk[i]]) {
                bool cand=false;
                for(int l=1;l<=3;++l) if(l!=e.p&&allowed({e.q,l})) {
                    cand=true; values.push_back({e.q,l});
                }
                if(cand) owner[e.q]=static_cast<int>(i);
            }
        }
        offset.reserve(domains.size()+1); offset.assign(1,0);
        for(const auto& d:domains) offset.push_back(offset.back()+d.size());
        // F(g): the owners i of qubits q in supp(g) with some value of D_i at q flipping g.
        flippers.assign(mm.m,0);
        for(size_t i=0;i<domains.size();++i) for(auto a:domains[i])
            for(auto e:mm.columns[a.q]) if(e.p!=a.l) flippers[e.q]|=U(1)<<i;
    }
    bool is_packed(int g) const { return std::find(packed.begin(),packed.end(),g)!=packed.end(); }
    bool killer(int g,int i,Action a) const {
        if(g<0||g>=m().m||i<0||i>=static_cast<int>(domains.size())) return false;
        U f=flippers[g], self=U(1)<<i;
        bool subset=!(f&~self);
        bool contains=false;
        for(auto e:m().rows[g]) contains|=e.q==a.q;
        bool k1=contains&&g!=packed[i]&&flips(a,g)!=synd(g)&&subset;
        bool k2=synd(g)&&!is_packed(g)&&f==self&&!flips(a,g);
        return k1||k2;
    }
    bool zero(int g) const { return g>=0&&g<m().m&&!is_packed(g)&&synd(g)&&!flippers[g]; }
    void replay(const cov2::Derivation& steps, const Action* extra=nullptr) const {
        // alive[offset[i]+k]: value k of D_i; a modified system starts without the values at E's qubit.
        alive.assign(offset.back(),1);
        if(extra) for(size_t i=0;i<domains.size();++i)
            for(size_t k=0;k<domains[i].size();++k) if(domains[i][k].q==extra->q) alive[offset[i]+k]=0;
        check(!steps.empty());
        bool ended=false;
        for(size_t at=0;at<steps.size();++at) {
            auto st=steps[at]; check(!ended);
            if(st.kind==cov2::Step::wiped) {
                check(st.v>=0&&st.v<static_cast<int>(domains.size()));
                for(size_t k=0;k<domains[st.v].size();++k) check(!alive[offset[st.v]+k]);
                ended=true;
            } else {
                check(st.g>=0&&st.g<m().m);
                bool target=synd(st.g)^(extra&&flips(*extra,st.g)), parity=false;
                if(st.kind==cov2::Step::elim) check(st.v>=0&&st.v<static_cast<int>(domains.size()));
                else check(st.kind==cov2::Step::odd);
                // A value flips g only at a qubit of supp(g), so a variable owning none of them is
                // determined with contribution 0. Draft-3: an empty alive set is determined, contribution 0.
                U seen=0;
                for(auto e:m().rows[st.g]) {
                    int j=owner[e.q];
                    if(j<0||(seen>>j)&1||(st.kind==cov2::Step::elim&&j==st.v)) continue;
                    seen|=U(1)<<j;
                    int bit=-1;
                    for(size_t k=0;k<domains[j].size();++k) if(alive[offset[j]+k]) {
                        int f=flips(domains[j][k],st.g);
                        if(bit<0) bit=f; else check(f==bit);
                    }
                    parity^=bit==1;
                }
                if(st.kind==cov2::Step::odd) { check(parity!=target); ended=true; }
                else {
                    size_t removed=0;
                    for(size_t k=0;k<domains[st.v].size();++k) {
                        uint8_t& live=alive[offset[st.v]+k];
                        if(live&&flips(domains[st.v][k],st.g)!=(target^parity)) { live=0; ++removed; }
                    }
                    check(removed>0);
                }
            }
            if(ended) check(at+1==steps.size());
        }
        check(ended);
    }
    void kill_list(int i,const std::vector<int>& ks,const Action* extra) const {
        check(i>=0&&i<static_cast<int>(domains.size())&&ks.size()==domains[i].size());
        checked_kills.resize(domains.size());
        if(kill_marks.empty()) kill_marks.assign(m().m,0);
        if(checked_kills[i]!=ks) {
            for(size_t c=0;c<ks.size();++c) check(killer(ks[c],i,domains[i][c]));
            for(int g:checked_kills[i]) kill_marks[g]&=~(U(1)<<i);
            checked_kills[i]=ks;
            for(int g:ks) kill_marks[g]|=U(1)<<i;
        }
        // E flips k iff k is in the column of E's qubit with another label there.
        if(extra) for(auto e:m().columns[extra->q]) if(e.p!=extra->l) check(!((kill_marks[e.q]>>i)&1));
    }
    std::vector<Action> actions() const {
        std::vector<Action> out;
        for(int q=0;q<m().n;++q) for(int l=1;l<=3;++l) if(allowed({q,l})) out.push_back({q,l});
        return out;
    }
    // VERIFY_S1F (9.3.1), for filter prunes; step 1 (VERIFY_PACKING) ran in the constructor.
    void s1f(const Justification& j) const {
        replay(j.steps);
        const auto& vars=j.filter_vars;
        check(j.filter_killers.size()==vars.size());
        U seen=0;
        for(size_t t=0;t<vars.size();++t) {
            int i=vars[t]; check(i>=0&&i<static_cast<int>(domains.size())&&!((seen>>i)&1)); seen|=U(1)<<i;
            // Exactly one killer per value of D_i, aligned with D's canonical order (load-bearing).
            const auto& ks=j.filter_killers[t];
            check(ks.size()==domains[i].size());
            for(size_t c=0;c<ks.size();++c) check(killer(ks[c],i,domains[i][c]));
        }
        for(int z:j.filter_zeros) check(zero(z));
        // Q_J (step 4) as qubit masks: the undecided qubits, intersected with the union of supp(k) over each
        // K_i and with supp(z) for each z of ZL. Accept iff it is empty.
        size_t words=(m().n+63)/64;
        std::vector<U> qj(words), cond(words);
        for(int q=0;q<m().n;++q) if(lab(q)>=4) qj[q/64]|=U(1)<<(q%64);
        auto restrict_to=[&](const int* gens,size_t count) {
            std::fill(cond.begin(),cond.end(),U(0));
            for(size_t k=0;k<count;++k) for(auto e:m().rows[gens[k]]) cond[e.q/64]|=U(1)<<(e.q%64);
            for(size_t w=0;w<words;++w) qj[w]&=cond[w];
        };
        for(const auto& ks:j.filter_killers) restrict_to(ks.data(),ks.size());
        for(const int& z:j.filter_zeros) restrict_to(&z,1);
        for(U w:qj) check(!w);
    }
    // VERIFY_S1E (9.3.2), for enumeration prunes.
    void s1(const Justification& j) const {
        if(j.filter) { s1f(j); return; }
        replay(j.steps);
        std::vector<int> anchor;
        if(j.z_anchor) { check(zero(j.anchor)); anchor.push_back(j.anchor); }
        else { kill_list(j.anchor,j.killers,nullptr); anchor=j.killers; }
        // ANCH: the actions flipping some anchor generator, i.e. (q, l) with q in its support,
        // q undecided, l in dom(q) and l != its label at q. at[q] holds the anchor labels at q.
        std::vector<uint8_t> at(m().n);
        for(int g:anchor) for(auto e:m().rows[g]) at[e.q]|=1<<e.p;
        auto& anch=scratch_actions; anch.clear(); anch.reserve(3*static_cast<size_t>(m().n));
        for(int q=0;q<m().n;++q) if(at[q]) for(int l=1;l<=3;++l)
            if((at[q]&~(1<<l))&&allowed({q,l})) anch.push_back({q,l});
        // ANCH and the reasons are both in action order: one merged pass finds every reason.
        auto found=j.reasons.begin();
        for(auto a:anch) {
            while(found!=j.reasons.end()&&found->first<a) ++found;
            check(found!=j.reasons.end()&&found->first==a); const auto& r=found->second;
            int g=r.index;
            switch(r.kind) {
            case cov2::Reason::packed: check(is_packed(g)&&flips(a,g)); break;
            case cov2::Reason::kill:
                check(r.shared<static_cast<int>(j.kill_lists.size()));
                kill_list(g,r.shared>=0?j.kill_lists[r.shared]:r.killers,&a); break;
            case cov2::Reason::zgen: check(zero(g)&&!flips(a,g)); break;
            case cov2::Reason::odd: check(g>=0&&g<m().m&&!synd(g)&&!flippers[g]&&flips(a,g)); break;
            case cov2::Reason::derivation: replay(r.steps,&a); break;
            default: check(false);
            }
        }
    }
    U key(Action a,const std::vector<U>& k) const {
        U value=0; for(auto e:m().columns[a.q]) if(e.p!=a.l) value^=k[e.q]; return value;
    }
    void s2(const Justification& j) const {
        const Model& mm=m();
        check(j.kappa.size()==static_cast<size_t>(mm.m));
        // tau = sigma(P_v) XOR the b_i; sigma(a) is the set of g in a's column with g_q != l.
        // sigma(P_v): s(g) for every g, as the parity of the fixed non-I labels over the columns.
        std::vector<uint8_t> tau(mm.m);
        for(int q=0;q<mm.n;++q) if(lab(q)>=1&&lab(q)<=3) for(auto e:mm.columns[q]) tau[e.q]^=e.p!=lab(q);
        for(const auto& d:domains) {
            U ref=key(d[0],j.kappa);
            for(auto c:d) check(key(c,j.kappa)==ref);
            for(auto e:mm.columns[d[0].q]) if(e.p!=d[0].l) tau[e.q]^=1;
        }
        U gamma=0; for(int g=0;g<mm.m;++g) if(tau[g]) gamma^=j.kappa[g];
        std::set<std::vector<Action>> exceptions;
        for(const auto& e:j.exceptions) {
            check(e.actions.size()<=2&&e.y.size()==static_cast<size_t>((mm.m+63)/64));
            if(mm.m%64) check((e.y.back()>>(mm.m%64))==0);
            for(auto a:e.actions) check(allowed(a));
            if(e.actions.size()==2) check(e.actions[0].q!=e.actions[1].q);
            std::vector<U> bits(mm.m);
            for(int g=0;g<mm.m;++g) bits[g]=(e.y[g/64]>>(g%64))&1;
            for(const auto& d:domains) for(auto c:d) check(key(c,bits)==key(d[0],bits));
            U parity=0; for(int g=0;g<mm.m;++g) if(tau[g]) parity^=bits[g];
            for(auto a:e.actions) parity^=key(a,bits);
            check(parity==1);
            auto es=e.actions; std::sort(es.begin(),es.end()); exceptions.insert(es);
        }
        if(!gamma) check(exceptions.contains({}));
        // Fingerprint table: slot -> 1 + first action with that key, chained in action order, so every
        // pair with kappa(a) XOR kappa(b) = gamma is found by one lookup per action.
        // kappa(sigma(q, l)): XOR of the column keys at q whose label differs from l, all three
        // labels from one column pass.
        auto& all=scratch_actions; all.assign(3*static_cast<size_t>(mm.n),{});
        std::vector<U> ka(all.size()); size_t count=0;
        for(int q=0;q<mm.n;++q) if(lab(q)>=4) {
            U total=0, same[4]={};
            for(auto e:mm.columns[q]) { total^=j.kappa[e.q]; same[e.p]^=j.kappa[e.q]; }
            for(int l=1;l<=3;++l) if(allowed({q,l})) { all[count]={q,l}; ka[count++]=total^same[l]; }
        }
        all.resize(count); ka.resize(count);
        std::vector<uint32_t> chain(all.size());
        size_t size=2; while(size<2*all.size()) size*=2;
        std::vector<uint32_t> table(size);
        auto bucket=[&](U value)->uint32_t& {
            size_t i=(value^(value>>32))&(size-1); // J's kappa comes from this checker: keys are pseudo-random
            while(table[i]&&ka[table[i]-1]!=value) i=(i+1)&(size-1);
            return table[i];
        };
        for(size_t k=all.size();k-->0;) { uint32_t& h=bucket(ka[k]); chain[k]=h; h=static_cast<uint32_t>(k+1); }
        for(size_t k=0;k<all.size();++k) {
            auto a=all[k];
            if(ka[k]==gamma) check(exceptions.contains({a}));
            for(uint32_t b=bucket(ka[k]^gamma);b&&b-1<k;b=chain[b-1]) {
                if(a.q==all[b-1].q) continue;
                std::vector<Action> pair{a,all[b-1]}; std::sort(pair.begin(),pair.end());
                check(exceptions.contains(pair));
            }
        }
    }
};
// r = cap - w, from lab.
int remaining(const State& s) {
    int weight=0, largest=0;
    for(uint8_t a:s.lab) { largest=std::max<int>(largest,a); weight+=a>=1&&a<=3; }
    check(largest<=7);
    return s.model->cap-weight;
}
// A rule-2 prune: J_P is the packing itself, of size r + 1.
void verify_packing_prune(const State& s,const int* pk,size_t size) {
    int r=remaining(s); check(r>=1);
    verify_packing(s,pk,size,r+1);
}
void verify(const State& s,int rule,const Justification& j) {
    int r=remaining(s);
    int size=rule==2?r+1:rule==5?r:rule==6?r-1:r-2;
    check(r>=1&&(rule==2||rule==5||rule==6||rule==7));
    if(rule==6) check(r>=2);
    if(rule==2) { verify_packing(s,j.packing.data(),j.packing.size(),size); return; }
    Local local(s,j.packing,size);
    if(rule==5) local.replay(j.steps);
    if(rule==6) local.s1(j);
    if(rule==7) local.s2(j);
}
}
