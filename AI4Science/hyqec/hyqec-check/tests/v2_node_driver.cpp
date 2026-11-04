// Private test harness: never shipped as a public certificate verifier.
#define HYQEC_V2_NO_MAIN
#include "../native/checker_v2.cpp"
#include <sstream>
int mutations(const State& state,const Decision& d) {
    if(d.rule!=2&&d.rule<5) return 0;
    cov2::Justification j; j.packing.assign(d.packing.begin(),d.packing.begin()+d.size);
    v2decide::System sys(state,j.packing);
    if(d.rule==5) require(v2decide::s0(sys,j),"missing S0 refutation");
    if(d.rule==6) require(v2decide::s1(sys,j),"missing S1 refutation");
    if(d.rule==7) require(v2decide::s2(sys,j),"missing S2 refutation");
    v2verify::verify(state,d.rule,j);
    int rejected=0;
    auto reject=[&](const cov2::Justification& bad) {
        try {v2verify::verify(state,d.rule,bad);}
        catch(const std::exception&) {++rejected;return;}
        throw std::runtime_error("mutated justification accepted");
    };
    auto bad=j;bad.packing.push_back(-1);reject(bad);
    if(!j.packing.empty()) {bad=j;bad.packing[0]=-1;reject(bad);}
    if(d.rule==5||d.rule==6) {bad=j;bad.steps.clear();reject(bad);bad=j;bad.steps={{cov2::Step::wiped,-1,0}};reject(bad);}
    if(d.rule==6&&!j.filter) {
        bad=j;bad.anchor=-1;reject(bad);
        for(const auto& [action,reason]:j.reasons) {
            (void)reason;bad=j;bad.reasons.erase(action);reject(bad);
            bad=j;bad.reasons[action]={cov2::Reason::packed,-1,{},{}};reject(bad);
        }
    }
    if(d.rule==6&&j.filter) {
        // J_S1F: every list must keep exactly |D_i| verified killers, and I must stay distinct.
        for(size_t t=0;t<j.filter_vars.size();++t) {
            bad=j;bad.filter_killers[t].pop_back();reject(bad);
            bad=j;bad.filter_killers[t].push_back(bad.filter_killers[t].back());reject(bad);
            bad=j;bad.filter_killers[t][0]=-1;reject(bad);
        }
        if(!j.filter_vars.empty()) {bad=j;bad.filter_vars.push_back(j.filter_vars[0]);bad.filter_killers.push_back(j.filter_killers[0]);reject(bad);}
        bad=j;bad.filter_zeros.push_back(-1);reject(bad);
    }
    if(d.rule==7) {
        bad=j;bad.kappa.assign(sys.m.m,0);bad.exceptions.clear();reject(bad);
        for(size_t i=0;i<j.exceptions.size();++i) {bad=j;bad.exceptions[i].y.assign((sys.m.m+63)/64,0);reject(bad);}
    }
    return rejected;
}
void unit_records_only(const State& state) {
    v2decide::System sys(state,{0}); auto u=v2decide::unit(sys);
    auto list=[](const auto& values) {
        std::cout<<'[';for(size_t i=0;i<values.size();++i)std::cout<<(i?",":"")<<values[i];std::cout<<']';
    };
    std::cout<<"{\"zeros\":";list(u.zeros);
    std::cout<<",\"wiped\":";list(u.wiped);
    std::cout<<",\"first\":[";
    for(size_t i=0;i<u.first.size();++i) {if(i)std::cout<<',';list(u.first[i]);}
    std::cout<<"],\"needs\":[";
    for(size_t i=0;i<u.needs.size();++i) {if(i)std::cout<<',';std::cout<<'['<<u.needs[i].first<<','<<u.needs[i].second<<']';}
    std::cout<<"],\"refuted\":"<<(u.refuted()?"true":"false")<<"}\n"<<std::flush;
}
#include "v2_verify_regressions.hpp"
// --verify-j: one external J per line, as whitespace tokens (scripts/v2-vectors.py writes them):
// lab[n] rule |PK| PK... steps form ...; form 0 = J_S1E (anchor, reasons), 1 = J_S1F (I, K_i, ZL).
cov2::Derivation read_steps(std::istream& in) {
    size_t n; in>>n; cov2::Derivation d(n);
    for(auto& st:d) { int k; in>>k>>st.g>>st.v; st.kind=k==0?cov2::Step::elim:k==1?cov2::Step::wiped:cov2::Step::odd; }
    return d;
}
std::vector<int> read_list(std::istream& in) { size_t n; in>>n; std::vector<int> v(n); for(int& x:v) in>>x; return v; }
void verify_external(const Model& model,State& state) {
    std::string line;
    while(std::getline(std::cin,line)) {
        if(line.empty()) continue;
        std::istringstream in(line);
        try {
            for(int q=0;q<model.n;++q) { int a; require(bool(in>>a)&&a>=0&&a<=7,"invalid test label"); state.change(q,a); }
            int rule; in>>rule; cov2::Justification j; j.packing=read_list(in); j.steps=read_steps(in);
            int form; in>>form;
            if(form==0) {
                int z; in>>z>>j.anchor; j.z_anchor=z; j.killers=read_list(in);
                size_t nr; in>>nr;
                for(size_t r=0;r<nr;++r) { cov2::Action a; int kind; cov2::Reason rs; in>>a.q>>a.l>>kind>>rs.index;
                    rs.kind=static_cast<cov2::Reason::Kind>(kind); rs.killers=read_list(in); rs.steps=read_steps(in); j.reasons[a]=rs; }
            } else {
                j.filter=true; j.filter_vars=read_list(in);
                for(size_t t=0;t<j.filter_vars.size();++t) j.filter_killers.push_back(read_list(in));
                j.filter_zeros=read_list(in);
            }
            require(bool(in),"J parse");
            v2verify::verify(state,rule,j); std::cout<<"{\"verify\":\"accept\"}\n"<<std::flush;
        } catch(const std::exception& e) { std::cout<<"{\"verify\":\"reject\",\"error\":"<<json_string(e.what())<<"}\n"<<std::flush; }
    }
}
int main(int argc,char** argv) {
    if(argc!=2&&argc!=3) return 2;
    try {
        Model model(argv[1]);
        bool unit_only=argc==3&&std::string(argv[2])=="--unit-records";
        if(argc==3&&std::string(argv[2])=="--verify-j") { State state(model);state.root(0);verify_external(model,state);return 0; }
        if(argc==3&&!unit_only) { require(std::string(argv[2])=="--draft3-verify","unknown test mode");draft3_verify_regressions(model);return 0; }
        State state(model);state.root(0);
        int first;
        while(std::cin>>first) {
            require(first>=0&&first<=7,"invalid test label");state.change(0,first);
            for(int q=1;q<model.n;++q) {int a;require(bool(std::cin>>a)&&a>=0&&a<=7,"invalid test label");state.change(q,a);}
            try {
                if(unit_only) {unit_records_only(state);continue;}
                if(state.dead_count) {std::cout<<"{\"dead\":"<<state.dead_hint()<<"}\n"<<std::flush;continue;}
                Decision d=decide(state);
                std::cout<<"{\"rule\":"<<d.rule<<",\"generator\":"<<d.generator<<",\"packing\":[";
                for(int i=0;i<d.size;++i)std::cout<<(i?",":"")<<d.packing[i];
                std::cout<<"]";
                if(state.active_count&&state.weight<model.cap) {
                    auto p=v2decide::packing(state);v2decide::System c(state,p.pk);auto u=v2decide::unit(c);
                    std::cout<<",\"np\":"<<p.np<<",\"PK\":[";
                    for(size_t i=0;i<p.pk.size();++i)std::cout<<(i?",":"")<<p.pk[i];
                    std::cout<<"],\"zeros\":[";
                    for(size_t i=0;i<u.zeros.size();++i)std::cout<<(i?",":"")<<u.zeros[i];
                    std::cout<<"],\"wiped\":[";
                    for(size_t i=0;i<u.wiped.size();++i)std::cout<<(i?",":"")<<u.wiped[i];
                    std::cout<<"],\"first\":[";
                    for(size_t i=0;i<u.first.size();++i) {std::cout<<(i?",[":"[");for(size_t k=0;k<u.first[i].size();++k)std::cout<<(k?",":"")<<u.first[i][k];std::cout<<"]";}
                    std::cout<<"]";
                    // The rule section 5.2 dispatches, with the intermediates the vectors record.
                    int r=model.cap-state.weight, R=state.active_count;
                    std::vector<std::string> limits;
                    auto list=[](const auto& xs){std::string o="[";for(size_t k=0;k<xs.size();++k)o+=(k?",":"")+std::to_string(xs[k]);return o+"]";};
                    auto unit_fields=[&](const v2decide::Unit& w){
                        if(w.zero_count>96) limits.push_back("zero_list_record_truncated_to_96");
                        if(w.need_count>96) limits.push_back("need_list_truncated_to_96");
                        return "\"unit_refuted\":"+std::string(w.refuted()?"true":"false")+",\"Z\":"+list(w.zeros)+",\"wiped\":"+list(w.wiped)
                            +",\"zero_generators\":"+std::to_string(w.zero_count)+",\"need_entries\":"+std::to_string(w.need_count);
                    };
                    if(R>=r) {
                        if(R>96) limits.push_back("greedy_list_truncated_to_96");
                        if(p.np<r&&R>64) limits.push_back("threshold_search_not_run_R_gt_64");
                        cov2::Justification j1;
                        if(p.np>r) {}
                        else if(model.rule_set==RuleSet::s0s1s2&&p.np==r-2) {
                            v2decide::S2Trace t; v2decide::s2(c,j1,&t);
                            if(std::string(t.stop)=="n_gt_1365") limits.push_back("S2_n_gt_1365");
                            std::cout<<",\"s2\":{\"stop\":\""<<t.stop<<"\"";
                            if(t.rank>=0) std::cout<<",\"rank_W\":"<<t.rank;
                            if(t.actions>=0) std::cout<<",\"actions\":"<<t.actions;
                            std::cout<<"}";
                        } else if(model.rule_set>=RuleSet::s0s1&&p.np==r-1&&r>=2) {
                            v2decide::S1Trace t; v2decide::s1(c,j1,&t);
                            std::string stop=t.stop;
                            if(stop=="values_gt_1024") limits.push_back("S1_values_gt_1024");
                            if(stop=="killers_ge_6144") limits.push_back("S1_killers_ge_6144");
                            if(stop=="h_gt_31") limits.push_back("S1_h_gt_31");
                            std::cout<<",\"s1\":{"<<unit_fields(t.unit)<<",\"values\":"<<t.values<<",\"stop\":\""<<stop<<"\"";
                            if(t.anchor>=0) {
                                if(t.z_anchor) std::cout<<",\"anchor\":[\"Z\","<<t.anchor<<"]";
                                else std::cout<<",\"anchor\":[\"WIPED\","<<t.anchor<<","<<list(t.killers)<<"]";
                                std::cout<<",\"B\":"<<list(t.base)<<",\"qstar_size\":"<<t.qstar;
                            }
                            if(t.killer_total>=0) std::cout<<",\"killer_total\":"<<t.killer_total;
                            if(stop=="E_survives"||stop=="all_E_excluded") {
                                std::cout<<",\"tests\":[";
                                for(int k=0;k<6;++k) std::cout<<(k?",":"")<<t.tests[k];
                                std::cout<<"]";
                            }
                            if(stop=="E_survives") std::cout<<",\"survivor\":["<<t.survivor.q<<","<<t.survivor.l<<"]";
                            std::cout<<"}";
                        } else if(p.np==r) {
                            auto w=v2decide::unit(c); std::string pr="null";
                            if(!w.refuted()) { cov2::Derivation d; v2decide::PropTrace pt; bool ref=v2decide::prop(c,d,&pt);
                                pr=ref?"\"refuted\"":pt.limits.empty()?"\"not_refuted\"":"\"gave_up\"";
                                for(auto x:pt.limits) limits.push_back(x);
                                if(!pt.limits.empty()) pr+=",\"prop_limit\":\""+std::string(pt.limits[0])+"\"";
                            }
                            std::cout<<",\"s0\":{"<<unit_fields(w)<<",\"prop\":"<<pr<<"}";
                        }
                    }
                    std::sort(limits.begin(),limits.end());
                    std::cout<<",\"limits_crossed\":[";
                    for(size_t k=0;k<limits.size();++k) std::cout<<(k?",":"")<<"\""<<limits[k]<<"\"";
                    std::cout<<"]";
                }
                std::cout<<",\"mutations_rejected\":"<<mutations(state,d)<<"}\n"<<std::flush;
            } catch(const std::exception& e) {std::cout<<"{\"error\":"<<json_string(e.what())<<"}\n"<<std::flush;}
        }
    } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
