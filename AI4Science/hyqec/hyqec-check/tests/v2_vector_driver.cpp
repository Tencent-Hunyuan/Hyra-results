// Private conformance adapter. Includes the unchanged production checker.
// Diagnostic intermediates below are reconstructed from checker-owned objects;
// decisions and every prune justification run through production decide/VERIFY.
#define HYQEC_V2_NO_MAIN
#include "../native/checker_v2.cpp"
#include <sstream>

namespace audit {
using namespace v2decide;
std::string enc(const std::string& s) {return json_string(s);}
std::string enc(const char* s) {return json_string(s);}
std::string enc(bool b) {return b?"true":"false";}
template<class T> std::string enc(T v) {return std::to_string(v);}
template<class T> std::string enc(const std::vector<T>& values) {
    std::string out="[";bool first=true;
    for(const auto& v:values) {if(!first)out+=',';first=false;out+=enc(v);}return out+"]";
}
struct Object {
    std::vector<std::string> fields;
    void raw(const std::string& k,const std::string& v) {fields.push_back(enc(k)+":"+v);}
    template<class T> void set(const std::string& k,const T& v) {raw(k,enc(v));}
    std::string str() const {std::string s="{";for(const auto& f:fields) {if(s.size()>1)s+=',';s+=f;}return s+"}";}
};
Object unit_info(const System& c,const Unit& u,std::vector<std::string>& limits) {
    int zero=0,need=0;
    for(int g:active_list(c.s)) if(!c.packed[g]) {zero+=!u.F[g];need+=std::popcount(u.F[g])==1;}
    if(zero>96)limits.push_back("zero_list_record_truncated_to_96");
    if(need>96)limits.push_back("need_list_truncated_to_96");
    Object o;o.set("unit_refuted",u.refuted());o.set("Z",u.zeros);o.set("wiped",u.wiped);
    o.set("zero_generators",zero);o.set("need_entries",need);return o;
}
Object s0_info(const System& c,std::vector<std::string>& limits) {
    auto u=unit(c);auto o=unit_info(c,u,limits);
    if(u.refuted()) {o.raw("prop","null");return o;}
    std::set<int> local;size_t occurrences=0;
    for(const auto& d:c.d)for(const auto& v:d) {local.insert(v.flips.begin(),v.flips.end());occurrences+=v.flips.size();}
    std::string why;
    if(c.choices()>1024)why="prop_choices_gt_1024";
    else if(local.size()>2048)why="prop_local_generators_gt_2048";
    else if(occurrences>6144)why="prop_occurrences_gt_6144";
    if(!why.empty()) {o.set("prop","gave_up");o.set("prop_limit",why);limits.push_back(why);}
    else {Derivation proof;o.set("prop",prop(c,proof)?"refuted":"not_refuted");}
    return o;
}
Object s1_info(const System& c,std::vector<std::string>& limits) {
    auto u=unit(c);auto o=unit_info(c,u,limits);o.set("values",c.choices());
    auto stop=[&](const char* name) {o.set("stop",name);return o;};
    if(!u.refuted())return stop("unit_not_refuted");
    if(c.choices()>1024) {limits.push_back("S1_values_gt_1024");return stop("values_gt_1024");}
    std::vector<int> base;
    if(!u.zeros.empty()) {base={u.zeros[0]};o.raw("anchor","[\"Z\","+enc(base[0])+"]");}
    else {
        int bv=u.wiped[0];for(int i:u.wiped)if(c.d[i].size()<c.d[bv].size())bv=i;
        o.raw("anchor","[\"WIPED\","+enc(bv)+","+enc(u.first[bv])+"]");
        for(int g:u.first[bv])if(std::find(base.begin(),base.end(),g)==base.end())base.push_back(g);
    }
    o.set("B",base);
    std::vector<bool> qs(c.m.n);for(int q=0;q<c.m.n;++q)qs[q]=c.s.lab[q]>=4;
    auto intersect=[&](const std::vector<int>& gs) {
        std::vector<bool> support(c.m.n);
        for(int g:gs)for(auto e:c.m.rows[g])support[e.q]=true;
        for(int q=0;q<c.m.n;++q)qs[q]=qs[q]&&support[q];
    };
    for(int i:u.wiped)intersect(u.first[i]);
    for(int z:u.zeros)intersect({z});
    int qsize=std::count(qs.begin(),qs.end(),true);o.set("qstar_size",qsize);
    if(!qsize)return stop("filter_prune");
    std::vector<std::vector<std::vector<int>>> all(c.d.size());size_t total=0;
    // Count the full K lists for the vector's informative total, even if the
    // production decider has already reached the give-up threshold.
    for(int i:u.wiped)for(const auto& v:c.d[i]) {
        std::vector<int> ks;
        for(auto e:c.m.columns[v.a.q])if(e.q!=c.pk[i]&&c.flip(v,e.q)!=bool(c.s.syndrome[e.q])&&!(c.F[e.q]&~(U(1)<<i)))ks.push_back(e.q);
        for(auto [g,vi]:u.needs)if(vi==i&&!c.flip(v,g))ks.push_back(g);
        total+=ks.size();all[i].push_back(std::move(ks));
    }
    o.set("killer_total",total);
    if(total>=6144) {limits.push_back("S1_killers_ge_6144");return stop("killers_ge_6144");}
    if(c.pk.size()>31) {limits.push_back("S1_h_gt_31");return stop("h_gt_31");}
    std::vector<Action> actions;std::set<Action> seen;
    for(int g:base)for(auto e:c.m.rows[g])for(int l=1;l<=3;++l)
        if(l!=e.p&&(c.s.lab[e.q]==4||c.s.lab[e.q]==4+l)&&seen.insert({e.q,l}).second)actions.push_back({e.q,l});
    std::array<int,6> tests{};
    auto counts=[&]() {Object t;for(int i=0;i<6;++i)t.set("6."+std::to_string(i+1),tests[i]);o.raw("tests",t.str());};
    for(auto a:actions) {
        auto ef=c.flips(a);auto has=[&](int g){return std::binary_search(ef.begin(),ef.end(),g);};
        if(std::any_of(ef.begin(),ef.end(),[&](int g){return c.packed[g];})) {++tests[0];continue;}
        bool excluded=false;
        for(int i:u.wiped)if(std::none_of(u.first[i].begin(),u.first[i].end(),has)) {excluded=true;break;}
        if(excluded){++tests[1];continue;}
        for(int z:u.zeros)if(!has(z)){excluded=true;break;}
        if(excluded){++tests[2];continue;}
        for(int g:ef)if(!c.s.syndrome[g]&&!c.F[g]){excluded=true;break;}
        if(excluded){++tests[3];continue;}
        for(int i:u.wiped) {
            bool killed=true;
            for(const auto& ks:all[i])if(std::none_of(ks.begin(),ks.end(),[&](int g){return !has(g);})) {killed=false;break;}
            if(killed){excluded=true;break;}
        }
        if(excluded){++tests[4];continue;}
        ++tests[5];
        if(!unit(c,ef,a.q).refuted()) {counts();o.set("survivor",std::vector<int>{a.q,a.l});return stop("E_survives");}
    }
    counts();return stop("all_E_excluded");
}
Object s2_info(const System& c,std::vector<std::string>& limits) {
    Object o;
    if(c.m.n>1365) {o.set("stop","n_gt_1365");limits.push_back("S2_n_gt_1365");return o;}
    for(const auto& d:c.d)if(d.empty()){o.set("stop","empty_value_list");return o;}
    Reduction red(c.m.m);
    auto sigma=[&](Action a){Bits b(red.words);for(int g:c.flips(a))b[g/64]^=U(1)<<(g%64);return b;};
    Bits tau(red.words);for(int g:active_list(c.s))tau[g/64]|=U(1)<<(g%64);
    for(const auto& d:c.d) {auto b=sigma(d[0].a);xor_into(tau,b);
        for(const auto& v:d) {auto x=sigma(v.a);xor_into(x,b);red.insert(std::move(x));}}
    o.set("rank_W",red.basis.size());
    auto target=red.reduce(tau);
    if(zero(target)){o.set("stop","survives");return o;}
    auto actions=c.actions();o.set("actions",actions.size());
    Justification j;o.set("stop",s2(c,j)?"prune":"survives");return o;
}
void node(const State& s) {
    auto d=decide(s);Object o;o.set("w",s.weight);int r=s.model->cap-s.weight;
    o.set("r",r);o.set("R",s.active_count);
    const char* names[]={"LEAF","CAP","PACKING","BRANCH","unused","S0","S1","S2"};o.set("decision",names[d.rule]);
    if(d.rule==3)o.set("branch_generator",d.generator);
    std::vector<std::string> limits;
    if(s.active_count<r){o.raw("np","null");o.raw("PK","null");}
    else {
        auto p=packing(s);o.set("np",p.np);o.set("PK",p.pk);
        if(s.active_count>96)limits.push_back("greedy_list_truncated_to_96");
        if(p.np<r&&s.active_count>64)limits.push_back("threshold_search_not_run_R_gt_64");
        if(p.np==r) {System c(s,p.pk);o.raw("s0",s0_info(c,limits).str());}
        else if(p.np==r-1&&r>=2&&s.model->rules!="s0") {System c(s,p.pk);o.raw("s1",s1_info(c,limits).str());}
        else if(p.np==r-2&&s.model->rules=="s0s1s2") {System c(s,p.pk);o.raw("s2",s2_info(c,limits).str());}
    }
    o.set("limits_crossed",limits);std::cout<<o.str()<<'\n'<<std::flush;
}
Derivation steps(std::istream& in) {
    size_t n;in>>n;Derivation out;
    for(size_t i=0;i<n;++i){int k,g,v;in>>k>>g>>v;out.push_back({static_cast<Step::Kind>(k),g,v});}return out;
}
std::vector<int> integers(std::istream& in) {size_t n;in>>n;std::vector<int> v(n);for(int& x:v)in>>x;return v;}
void verify_j(const State& s,std::istream& in) {
    int rule;in>>rule;Justification j;j.packing=integers(in);j.steps=steps(in);
    in>>j.z_anchor>>j.anchor;j.killers=integers(in);size_t nr;in>>nr;
    for(size_t i=0;i<nr;++i) {Action a;int kind,index;in>>a.q>>a.l>>kind>>index;
        auto ks=integers(in);auto ds=steps(in);j.reasons[a]={static_cast<Reason::Kind>(kind),index,ks,ds};}
    require(bool(in),"truncated vector J bridge");
    try {v2verify::verify(s,rule,j);std::cout<<"{\"verdict\":\"accept\"}\n";}
    catch(const std::exception& e){std::cout<<"{\"verdict\":\"reject\",\"reason\":"<<json_string(e.what())<<"}\n";}
    std::cout<<std::flush;
}
void search(const Model& model,int threads) {
    int roots=model.root_count();std::vector<Summary> sums(roots);std::vector<std::string> errors(roots);
    #pragma omp parallel for schedule(dynamic) num_threads(threads)
    for(int i=0;i<roots;++i)try {State s(model);s.root(i);Walker w;w.walk<false>(s);sums[i]=w.summary;}
        catch(const std::exception& e){errors[i]=e.what();}
    Summary total;for(int i=0;i<roots;++i) {require(errors[i].empty(),errors[i].c_str());total.append(sums[i]);}
    Object o;o.set("total_nodes",total.count);o.set("trace",total.hash);o.set("logical_found",false);o.set("threads",threads);
    std::cout<<o.str()<<'\n';
}
}
int main(int argc,char** argv) {
    try {
        require(argc>=3,"usage: vector-driver MODEL MODE [THREADS]");Model model(argv[1]);std::string mode=argv[2];
        if(mode=="search"){audit::search(model,argc==4?std::stoi(argv[3]):4);return 0;}
        State state(model);state.root(0);std::string lab,rules;int cap;
        while(std::cin>>cap>>rules>>lab) {
            require(lab.size()==static_cast<size_t>(model.n),"bad label count");model.cap=cap;model.rules=rules;
            for(int q=0;q<model.n;++q){require(lab[q]>='0'&&lab[q]<='7',"bad label");state.change(q,lab[q]-'0');}
            if(mode=="node")audit::node(state);else if(mode=="verify")audit::verify_j(state,std::cin);
            else require(false,"unknown vector mode");
        }
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
