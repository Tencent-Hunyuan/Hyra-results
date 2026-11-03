// Independent implementation of docs/BNB_CERT_SPEC.md. No prover code imported.
#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#include <omp.h>

namespace {
using U = uint64_t;
constexpr U limit = (U(1) << 63) - 1, base = 0x100000001b3;
void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
U add(U a, U b) { require(b <= limit && a <= limit - b, "counter overflow"); return a + b; }
U mix(U x) { x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL; return x ^ (x >> 31); }
U power(U exponent) { U result=1, value=base; while(exponent) {
    if(exponent & 1) result *= value;
    value *= value; exponent >>= 1; } return result; }
struct Summary {
    U count=0, hash=0;
    void word(U x) { hash=hash*base+x; }
    void entry(U fingerprint, U packed) { count=add(count,1); word(fingerprint); word(packed); }
    void append(Summary b) { count=add(count,b.count); hash=hash*power(2*b.count)+b.hash; }
};
class Reader {
    std::ifstream input;
    std::array<unsigned char, 65536> buffer{};
    size_t at=0, end=0;
    bool refill() { input.read(reinterpret_cast<char*>(buffer.data()),buffer.size());
        end=static_cast<size_t>(input.gcount()); at=0;
        require(!input.bad(), "proof stream read failed"); return end != 0; }
public:
    Reader(const std::string& path,U offset):input(path,std::ios::binary) {
        require(bool(input),"cannot open proof stream"); if(offset) input.seekg(offset);
        require(bool(input),"cannot seek proof stream"); }
    unsigned byte() { if(at==end) require(refill(),"truncated proof stream"); return buffer[at++]; }
    U fixed() { U value=0; for(int i=0;i<8;++i) value |= U(byte()) << (8*i); return value; }
    U integer() { U value=0; for(int i=0;i<10;++i) { unsigned b=byte();
        require(i!=9 || b<=1,"LEB128 exceeds 64 bits"); value |= U(b&127) << (7*i);
        if(!(b&128)) { require(i==0 || b!=0,"nonminimal LEB128");
            require(value<=limit,"integer exceeds signed counter limit"); return value; } }
        throw std::runtime_error("overlong LEB128"); }
    void expect_byte(unsigned value,const char* name) { if(byte()!=value) throw std::runtime_error(std::string("wrong ")+name); }
    void expect(U value,const char* name) { if(integer()!=value) throw std::runtime_error(std::string("mismatching ")+name); }
    void expect_fixed(U value,const char* name) { if(fixed()!=value) throw std::runtime_error(std::string("mismatching ")+name); }
    bool eof() { return at==end && !refill(); }
};
void put_fixed(std::ostream& s,U value) { for(int i=0;i<8;++i) s.put(char(value>>(8*i))); }
U get_fixed(std::istream& s) { U result=0; for(int i=0;i<8;++i) { int c=s.get();
    require(c!=EOF,"truncated private frontier spool"); result |= U(static_cast<unsigned char>(c))<<(8*i); } return result; }
struct Entry { int q, p; };
struct BasisRow { int pivot; std::vector<U> words; };
struct Model {
    int n,m,rank,cap,split;
    char domain;
    bool full;
    std::vector<std::vector<Entry>> rows, columns;
    std::vector<BasisRow> basis;
    std::vector<int> orbit,reps;
    std::vector<std::array<U,8>> salts;
    int root_count() const { return cap ? static_cast<int>(reps.size())*(domain=='P'?3:1) : 0; }
    explicit Model(const std::string& path) {
        std::ifstream in(path); std::string detail;
        require(bool(in>>n>>m>>rank>>cap>>split>>domain>>detail),"invalid private model header");
        require(n>=1&&n<=4096&&m>=0&&m<=4096&&rank>=0&&rank<n&&cap>=0&&cap<=46&&split>=1&&split<=47,
                "private model dimensions out of range");
        require(domain=='P'||domain=='X'||domain=='Z',"invalid private model domain");
        require(detail=="full"||detail=="frontier","invalid private model detail"); full=detail=="full";
        rows.resize(m); columns.resize(n); salts.resize(n);
        for(int g=0;g<m;++g) { int count; require(bool(in>>count)&&count>=0&&count<=32,"invalid generator length");
            int previous=-1;
            for(int i=0;i<count;++i) { int q,p; require(bool(in>>q>>p)&&q>previous&&q<n&&p>=1&&p<=3,"invalid generator row");
                previous=q; rows[g].push_back({q,p}); columns[q].push_back({g,p});
                require(columns[q].size()<=64,"column degree exceeds limit"); } }
        int words=(2*n+63)/64, previous=-1;
        for(int i=0;i<rank;++i) { BasisRow row; require(bool(in>>row.pivot)&&row.pivot>previous&&row.pivot<2*n,"invalid basis pivot");
            previous=row.pivot; row.words.resize(words); for(U& word:row.words) require(bool(in>>word),"invalid basis word");
            basis.push_back(std::move(row)); }
        orbit.resize(n); for(int& oi:orbit) require(bool(in>>oi)&&oi>=0&&oi<n,"invalid private orbit");
        int no; require(bool(in>>no)&&no>0&&no<=n,"invalid private orbit count"); reps.resize(no);
        for(int i=0;i<no;++i) require(bool(in>>reps[i])&&reps[i]>=0&&reps[i]<n&&orbit[reps[i]]==i,"invalid private representative");
        for(int oi:orbit) require(oi<no,"orbit exceeds count");
        std::string trailing; require(!(in>>trailing),"trailing private model data");
        for(int q=0;q<n;++q) for(int a=0;a<8;++a) salts[q][a]=mix(8*U(q)+a+1);
    }
};
bool candidate(int lab,int pauli) { return lab==4 || (lab>4 && lab-4!=pauli); }
int options(int lab,int pauli) { return candidate(lab,pauli) ? (lab==4?2:1) : 0; }
bool anti(int lab,int pauli) { return lab>0 && lab<4 && lab!=pauli; }
struct State {
    const Model* model;
    std::vector<uint8_t> lab, syndrome, cnt, child_count;
    std::vector<U> active, dead;
    int weight=0, active_count=0, dead_count=0;
    U fp=0;
    explicit State(const Model& m): model(&m),lab(m.n),syndrome(m.m),cnt(m.m),child_count(m.m),
        active((m.m+63)/64),dead((m.m+63)/64) {}
    void root(int ordinal) {
        const Model& m=*model; require(ordinal>=0&&ordinal<m.root_count(),"root ordinal out of range");
        int oi=ordinal/(m.domain=='P'?3:1), label=m.domain=='P'?ordinal%3+1:(m.domain=='X'?1:2);
        int initial=m.domain=='P'?4:(m.domain=='X'?5:6);
        std::fill(active.begin(),active.end(),0); std::fill(dead.begin(),dead.end(),0);
        weight=1; fp=0; active_count=dead_count=0;
        for(int q=0;q<m.n;++q) { lab[q]=m.orbit[q]<oi?0:initial; if(q==m.reps[oi]) lab[q]=label; fp^=m.salts[q][lab[q]]; }
        for(int g=0;g<m.m;++g) { cnt[g]=child_count[g]=syndrome[g]=0;
            for(auto e:m.rows[g]) { cnt[g]+=candidate(lab[e.q],e.p); child_count[g]+=options(lab[e.q],e.p); syndrome[g]^=anti(lab[e.q],e.p); }
            if(syndrome[g]) { active[g/64]|=U(1)<<(g%64); ++active_count;
                if(!cnt[g]) { dead[g/64]|=U(1)<<(g%64); ++dead_count; } } }
    }
    void change(int q,int value) {
        int old=lab[q]; if(old==value) return;
        const Model& m=*model;
        fp ^= m.salts[q][old]^m.salts[q][value];
        weight += (value>0&&value<4)-(old>0&&old<4);
        for(auto e:m.columns[q]) { int g=e.q; bool was_dead=syndrome[g]&&!cnt[g];
            cnt[g]+=candidate(value,e.p)-candidate(old,e.p);
            child_count[g]+=options(value,e.p)-options(old,e.p);
            if(anti(old,e.p)!=anti(value,e.p)) { syndrome[g]^=1; active[g/64]^=U(1)<<(g%64); active_count+=syndrome[g]?1:-1; }
            bool now_dead=syndrome[g]&&!cnt[g];
            if(was_dead!=now_dead) { dead[g/64]^=U(1)<<(g%64); dead_count+=now_dead?1:-1; }
        }
        lab[q]=value;
    }
    int dead_hint() const { require(dead_count>0,"dead hint requested for nondead state");
        for(size_t b=0;b<dead.size();++b) if(dead[b]) return static_cast<int>(64*b+std::countr_zero(dead[b]));
        throw std::runtime_error("dead-state invariant failed"); }
    bool in_stabilizers() const {
        const Model& m=*model; size_t words=(2*m.n+63)/64;
        std::array<U,128> value; std::fill_n(value.begin(),words,U(0));
        for(int q=0;q<m.n;++q) if(lab[q]<4) { if(lab[q]&1) value[q/64]|=U(1)<<(q%64);
            if(lab[q]&2) value[(m.n+q)/64]|=U(1)<<((m.n+q)%64); }
        for(const auto& row:m.basis) if((value[row.pivot/64]>>(row.pivot%64))&1)
            for(size_t i=0;i<words;++i) value[i]^=row.words[i];
        for(size_t i=0;i<words;++i) if(value[i]) return false;
        return true;
    }
};
struct Decision { int rule=0, generator=65535, size=0; std::array<int,96> packing;
    U packed(const State& s) const { return U(s.weight)|(U(s.active_count)<<8)|(U(rule)<<24)|(U(generator)<<32)|(U(size)<<48); } };
Decision decide(const State& s) {
    require(s.dead_count==0,"entered a dead state"); const Model& m=*s.model; Decision result;
    if(!s.active_count) {
        if(!s.in_stabilizers()) throw std::runtime_error("nontrivial logical leaf of weight "+std::to_string(s.weight));
        return result;
    }
    if(s.weight+1>m.cap) { result.rule=1; return result; }
    if(s.weight+s.active_count>m.cap) {
        std::array<int,96> order; int length=0;
        for(size_t b=0;b<s.active.size()&&length<96;++b) { U bits=s.active[b];
            while(bits&&length<96) { int g=static_cast<int>(64*b+std::countr_zero(bits)); bits&=bits-1; order[length++]=g; } }
        std::sort(order.begin(),order.begin()+length,[&](int a,int b){ return s.cnt[a]!=s.cnt[b]?s.cnt[a]<s.cnt[b]:a<b; });
        std::array<U,64> used; std::fill_n(used.begin(),(m.n+63)/64,U(0)); int selected=0;
        for(int i=0;i<length;++i) {
            if(s.weight+selected+length-i<=m.cap) break;
            int g=order[i]; bool overlaps=false;
            for(auto e:m.rows[g]) if(candidate(s.lab[e.q],e.p)&&(used[e.q/64]&(U(1)<<(e.q%64)))) { overlaps=true; break; }
            if(overlaps) continue;
            result.packing[selected++]=g;
            for(auto e:m.rows[g]) if(candidate(s.lab[e.q],e.p)) used[e.q/64]|=U(1)<<(e.q%64);
            if(s.weight+selected>m.cap) { result.rule=2; result.size=selected; return result; }
        }
    }
    int best=std::numeric_limits<int>::max();
    for(size_t b=0;b<s.active.size();++b) { U bits=s.active[b]; while(bits) {
        int g=static_cast<int>(64*b+std::countr_zero(bits)); bits&=bits-1;
        if(s.child_count[g]<best) { best=s.child_count[g]; result.generator=g; } } }
    require(result.generator!=65535,"missing branch generator"); result.rule=3; return result;
}
struct Task { U id=0, root=0; std::vector<uint8_t> path; Summary before,expected,actual; bool selected=false; std::string error; };
struct Sink {
    Reader& reader;
    bool top;
    U events=0,tasks=0,root=0;
    Summary segment;
    std::ostream* spool=nullptr;
    Sink(Reader& r,bool t):reader(r),top(t) {}
    void advance() {
        events=add(events,1);
        if(top && (events&((U(1)<<20)-1))==0)
            std::cerr<<"top traversal: "<<events<<" events, "<<tasks<<" frontier tasks\n";
    }
    U enter(const State& s,U parent,U edge,const Decision& d) {
        U id=events; advance();
        reader.expect_byte(0x10,"ENTER tag"); reader.expect(parent,"entry parent"); reader.expect(edge,"entry edge");
        reader.expect_byte(d.rule,"entry rule"); reader.expect_fixed(s.fp,"entry fingerprint");
        reader.expect(s.weight,"entry weight"); reader.expect(s.active_count,"active count");
        if(d.rule==2) { reader.expect(d.size,"packing size"); for(int i=0;i<d.size;++i) reader.expect(d.packing[i],"packing generator"); }
        if(d.rule==3) reader.expect(d.generator,"branch generator");
        if(top) segment.entry(s.fp,d.packed(s));
        return id;
    }
    void dead_event(const State& s,U parent,U edge,bool suffix=false) {
        advance(); reader.expect_byte(suffix?0x12:0x11,suffix?"DEAD_SUFFIX tag":"DEAD tag");
        reader.expect(parent,"dead parent"); reader.expect(edge,suffix?"dead suffix position":"dead edge");
        reader.expect(s.dead_hint(),"dead generator"); reader.expect_fixed(s.fp,"dead fingerprint");
    }
    void frontier(const State& s,U parent,U edge,const std::vector<uint8_t>& path) {
        require(top&&spool,"unexpected frontier"); advance();
        reader.expect_byte(0x13,"FRONTIER tag"); reader.expect(parent,"frontier parent"); reader.expect(edge,"frontier edge");
        reader.expect(tasks,"frontier task id"); reader.expect_fixed(s.fp,"frontier fingerprint");
        put_fixed(*spool,segment.count); put_fixed(*spool,segment.hash); put_fixed(*spool,root);
        spool->put(char(path.size())); for(auto p:path) spool->put(char(p)); require(bool(*spool),"frontier spool write failed");
        segment={}; tasks=add(tasks,1);
    }
};
struct Walker {
    Summary summary;
    Sink* sink=nullptr;
    std::vector<uint8_t> path;
    template<bool events> void walk(State& s,U parent=0,U edge=0) {
        const Model& m=*s.model;
        if constexpr(events) {
            if(sink->top&&m.split==1&&path.empty()) { sink->frontier(s,parent,edge,path); return; }
        }
        if(s.dead_count) { if constexpr(events) sink->dead_event(s,parent,edge); return; }
        if constexpr(events) {
            if(sink->top&&s.weight==m.split) { sink->frontier(s,parent,edge,path); return; }
        }
        Decision d=decide(s); U id=0;
        if constexpr(events) id=sink->enter(s,parent,edge,d);
        summary.entry(s.fp,d.packed(s));
        if(d.rule!=3) return;
        const auto& row=m.rows[d.generator]; std::array<uint8_t,32> old;
        for(size_t p=0;p<row.size();++p) old[p]=s.lab[row[p].q];
        for(size_t p=0;p<row.size();++p) {
            int q=row[p].q, gp=row[p].p, a=old[p]; if(!candidate(a,gp)) continue;
            int li=0;
            for(int label=1;label<=3;++label) {
                if(a==4 ? label==gp : label!=a-4) continue;
                int decision=static_cast<int>(2*p+li++); s.change(q,label);
                if constexpr(events) path.push_back(decision);
                walk<events>(s,id+1,decision);
                if constexpr(events) path.pop_back();
                s.change(q,a);
            }
            s.change(q,a==4?4+gp:0);
            if(s.dead_count) { if constexpr(events) sink->dead_event(s,id+1,p,true); break; }
        }
        for(size_t p=0;p<row.size();++p) s.change(row[p].q,old[p]);
    }
};
void replay(State& s,const std::vector<uint8_t>& path) {
    const Model& m=*s.model;
    for(unsigned byte:path) {
        Decision d=decide(s); require(d.rule==3,"task path crosses a terminal node");
        const auto& row=m.rows[d.generator]; size_t target=byte>>1; int option=byte&1;
        require(target<row.size(),"task support position out of range");
        for(size_t p=0;p<target;++p) if(candidate(s.lab[row[p].q],row[p].p)) {
            int a=s.lab[row[p].q]; s.change(row[p].q,a==4?4+row[p].p:0);
            require(!s.dead_count,"task path crosses a dead suffix"); }
        int q=row[target].q, a=s.lab[q]; require(candidate(a,row[target].p),"task decision is not a candidate");
        int label=-1, li=0;
        for(int p=1;p<=3;++p) if(a==4 ? p!=row[target].p : p==a-4) { if(li++==option) label=p; }
        require(label!=-1,"task label option out of range"); s.change(q,label);
        require(!s.dead_count,"task path ends in a dead child");
    }
    require(s.weight==m.split,"task path does not reach split weight");
}
U parse_number(const char* text) { std::string value=text; require(!value.empty(),"empty numeric argument"); U result=0;
    for(char c:value) { require(c>='0'&&c<='9',"invalid numeric argument");
        require(result<=(limit-U(c-'0'))/10,"numeric argument overflow"); result=result*10+U(c-'0'); } return result; }
std::string json_string(const std::string& value) { std::string out="\"";
    for(unsigned char c:value) { if(c=='\"'||c=='\\') { out+='\\'; out+=c; } else if(c>=32) out+=c; else out+=' '; } return out+'\"'; }
struct Selection {
    std::string mode; U start,stop; std::vector<U> sample;
    Selection(std::string kind,U total,U a,U b,U size,U seed):mode(std::move(kind)),start(a),stop(b) {
        require(mode=="all"||mode=="range"||mode=="sample","invalid selection mode");
        if(mode=="range") require(start<=stop&&stop<=total,"task range exceeds canonical task set");
        if(mode=="sample") {
            require(size<=total,"sample size exceeds task count"); require(size<=16000000,"sample exceeds configured 128MB selection limit");
            sample.resize(size); std::iota(sample.begin(),sample.end(),U(0)); std::mt19937_64 rng(seed);
            for(U i=size;i<total;++i) { U range=i+1, threshold=-range%range, r;
                do { r=rng(); } while(r<threshold); U j=r%range; if(j<size) sample[j]=i; }
            std::sort(sample.begin(),sample.end());
        }
    }
    bool selected(U id) const { if(mode=="all") return true; if(mode=="range") return id>=start&&id<stop;
        return std::binary_search(sample.begin(),sample.end(),id); }
};
Task read_spooled(std::istream& spool,U id) {
    Task task; task.id=id; task.before={get_fixed(spool),get_fixed(spool)}; task.root=get_fixed(spool);
    int length=spool.get(); require(length>=0&&length<=46,"invalid private task path length");
    for(int j=0;j<length;++j) { int c=spool.get(); require(c>=0&&c<=63,"invalid private task path byte"); task.path.push_back(c); }
    return task;
}
void read_result(Reader& reader,Task& task) {
    reader.expect_byte(0x20,"TASK_RESULT tag"); reader.expect(task.id,"task result id"); reader.expect(task.root,"task result root");
    reader.expect(task.path.size(),"task path length"); for(auto p:task.path) reader.expect_byte(p,"task path byte");
    task.expected.count=reader.integer(); task.expected.hash=reader.fixed();
}
void skip_transcript(Reader& reader) {
    U events=0;
    for(;;) { unsigned tag=reader.byte(); if(tag==0x22) { reader.expect(events,"skipped task event count"); return; }
        events=add(events,1);
        require(tag==0x10||tag==0x11||tag==0x12,"invalid full transcript event tag");
        U parent=reader.integer(); require(parent<=events-1,"forward parent in skipped transcript");
        reader.integer();
        if(tag!=0x10) { reader.integer(); reader.fixed(); continue; }
        unsigned rule=reader.byte(); require(rule<=3,"invalid full transcript rule"); reader.fixed(); reader.integer(); reader.integer();
        if(rule==3) reader.integer();
        if(rule==2) { U n=reader.integer(); require(n<=96,"packing size exceeds AMAX"); for(U i=0;i<n;++i) reader.integer(); }
    }
}
struct Cached { U root=limit; State base_state,work;
    explicit Cached(const Model& m):base_state(m),work(m) {}
    State& task(const Task& t) { if(root!=t.root) { require(t.root<=12288,"invalid root ordinal"); base_state.root(static_cast<int>(t.root)); root=t.root; }
        work=base_state; replay(work,t.path); return work; }
};
void execute(int argc,char** argv) {
    require(argc==12,"usage: checker MODEL PROOF OFFSET SPOOL THREADS MODE START STOP SAMPLE SEED");
    auto started=std::chrono::steady_clock::now();
    Model model(argv[1]); Reader reader(argv[2],parse_number(argv[3]));
    int threads=static_cast<int>(parse_number(argv[5])); require(threads>=1&&threads<=384,"invalid thread count");
    std::ofstream spool_out(argv[4],std::ios::binary|std::ios::trunc); require(bool(spool_out),"cannot open frontier spool");
    Sink top(reader,true); top.spool=&spool_out; Walker generator; generator.sink=&top; State root(model);
    for(int i=0;i<model.root_count();++i) { top.root=i; root.root(i); generator.walk<true>(root,0,i); }
    reader.expect_byte(0x1f,"TOP_END tag"); reader.expect(top.events,"top event count"); reader.expect(top.tasks,"top task count");
    reader.expect(generator.summary.count,"generation node count"); reader.expect_fixed(generator.summary.hash,"top trace");
    spool_out.close(); require(bool(spool_out),"frontier spool close failed");
    double top_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();
    std::cerr<<"frontier verified: "<<top.events<<" events, "<<top.tasks<<" tasks, "<<generator.summary.count<<" generation nodes\n";
    Selection selection(argv[6],top.tasks,parse_number(argv[7]),parse_number(argv[8]),parse_number(argv[9]),parse_number(argv[10]));
    // argv[11] is a reserved protocol version, pinned by the Python adapter.
    require(std::string(argv[11])=="1","unsupported private checker protocol");
    std::ifstream spool(argv[4],std::ios::binary); require(bool(spool),"cannot read frontier spool");
    std::vector<std::unique_ptr<Cached>> cache; for(int i=0;i<threads;++i) cache.push_back(std::make_unique<Cached>(model));
    U checked=0, checked_nodes=0, task_nodes=0; Summary global; double replay_seconds=0;
    auto progress=std::chrono::steady_clock::now();
    for(U begin=0;begin<top.tasks;) {
        U count=model.full?1:std::min<U>(4096,top.tasks-begin);
        std::vector<Task> batch; batch.reserve(count);
        for(U j=0;j<count;++j) {
            Task task=read_spooled(spool,begin+j); task.selected=selection.selected(task.id);
            if(model.full) {
                reader.expect_byte(0x21,"TASK_BEGIN tag"); reader.expect(task.id,"task begin id");
                if(task.selected) {
                    auto t=std::chrono::steady_clock::now(); State& state=cache[0]->task(task);
                    Sink local(reader,false); Walker walk; walk.sink=&local; walk.walk<true>(state);
                    reader.expect_byte(0x22,"TASK_END tag"); reader.expect(local.events,"task event count");
                    task.actual=walk.summary;
                    replay_seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-t).count();
                } else skip_transcript(reader);
            }
            read_result(reader,task); batch.push_back(std::move(task));
        }
        if(!model.full) {
            auto t=std::chrono::steady_clock::now();
            #pragma omp parallel for schedule(dynamic) num_threads(threads)
            for(size_t j=0;j<batch.size();++j) if(batch[j].selected) {
                try { State& state=cache[omp_get_thread_num()]->task(batch[j]); Walker walk;
                    walk.walk<false>(state); batch[j].actual=walk.summary;
                } catch(const std::exception& error) { batch[j].error=error.what(); }
            }
            replay_seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-t).count();
        }
        for(auto& task:batch) {
            if(!task.error.empty()) throw std::runtime_error("task "+std::to_string(task.id)+": "+task.error);
            if(task.selected) {
                if(task.actual.count!=task.expected.count) throw std::runtime_error("task "+std::to_string(task.id)+" node count mismatch");
                if(task.actual.hash!=task.expected.hash) throw std::runtime_error("task "+std::to_string(task.id)+" trace mismatch");
                checked=add(checked,1); checked_nodes=add(checked_nodes,task.actual.count);
            }
            global.append(task.before); global.append(task.expected); task_nodes=add(task_nodes,task.expected.count);
        }
        begin+=count;
        if(std::chrono::duration<double>(std::chrono::steady_clock::now()-progress).count()>5) {
            std::cerr<<"checked "<<checked<<" tasks; parsed "<<begin<<'/'<<top.tasks<<"; replayed "<<checked_nodes<<" nodes\n";
            progress=std::chrono::steady_clock::now();
        }
    }
    require(spool.peek()==EOF,"trailing private frontier spool bytes"); global.append(top.segment);
    require(global.count==add(generator.summary.count,task_nodes),"global composition count mismatch");
    reader.expect_byte(0x7f,"COMPLETE tag"); reader.expect(top.tasks,"footer task count"); reader.expect(generator.summary.count,"footer generation nodes");
    reader.expect(task_nodes,"footer task nodes"); reader.expect(global.count,"footer total nodes"); reader.expect_fixed(global.hash,"global trace");
    require(reader.eof(),"data follows logical footer");
    double elapsed=std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();
    bool accepted=selection.mode=="all" && checked==top.tasks;
    std::cout<<"{\"status\":\""<<(accepted?"accepted":"partial")<<"\",\"certificate_verified\":"<<(accepted?"true":"false")
      <<",\"frontier_verified\":true,\"tasks_checked\":"<<checked<<",\"tasks_total\":"<<top.tasks
      <<",\"gen_nodes\":"<<generator.summary.count<<",\"task_nodes\":"<<task_nodes<<",\"total_nodes\":"<<global.count
      <<",\"replayed_task_nodes\":"<<checked_nodes<<",\"global_trace\":"<<json_string(std::to_string(global.hash))
      <<",\"wall_seconds\":"<<elapsed<<",\"top_wall_seconds\":"<<top_seconds<<",\"replay_wall_seconds\":"<<replay_seconds
      <<",\"nodes_per_replay_second\":"<<(replay_seconds?checked_nodes/replay_seconds:0)
      <<",\"threads\":"<<(model.full?1:threads)<<",\"mismatches\":[]}\n";
}
}
int main(int argc,char** argv) {
    try { execute(argc,argv); return 0; }
    catch(const std::exception& error) { std::cout<<"{\"status\":\"rejected\",\"certificate_verified\":false,\"mismatches\":["<<json_string(error.what())<<"]}\n"; return 1; }
}
