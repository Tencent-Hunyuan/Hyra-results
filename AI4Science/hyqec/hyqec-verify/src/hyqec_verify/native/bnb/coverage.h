// Coverage exporter. Search semantics and bytes: docs/BNB_CERT_SPEC.md (version 1, rules greedy) and
// docs/BNB_CERT_SPEC_v2.md (version 2: rules greedy, s0, s0s1, s0s1s2). Included by bnb.cu after the executors.
#pragma once
#include <functional>
#include <memory>
#include <cctype>
#include <unistd.h>

struct CovSummary {
  uint64_t nodes = 0, hash = 0;
  void append(const CovSummary& b) {
    if (b.nodes > uint64_t(INT64_MAX) - nodes) throw runtime_error("coverage node count overflow");
    uint64_t x = UINT64_C(0x100000001b3), e = 2*b.nodes, power = 1;
    while (e) { if (e & 1) power *= x; x *= x; e >>= 1; }
    hash = hash*power + b.hash; nodes += b.nodes;
  }
  void entry(const Dfs& s) {
    if (nodes == uint64_t(INT64_MAX)) throw runtime_error("coverage node count overflow");
    nodes++; hash = trace_append(hash, s.state_fp, trace_word(s.w,s.nA,s.last_rule,s.last_g,s.last_h));
  }
};

// All spools are disk-backed and unlinked automatically, including on exceptions.
struct CovFile {
  FILE* f;
  CovFile(): f(tmpfile()) { if (!f) throw runtime_error("cannot create coverage spool"); }
  ~CovFile() { fclose(f); }
  CovFile(const CovFile&) = delete;
  void write(const void* p, size_t n) { if (n && fwrite(p,1,n,f)!=n) throw runtime_error("coverage write failed"); }
  void read(void* p, size_t n) { if (n && fread(p,1,n,f)!=n) throw runtime_error("coverage spool truncated"); }
  void rewind_read() { if (fflush(f) || fseek(f,0,SEEK_SET)) throw runtime_error("coverage spool seek failed"); }
  void byte(uint8_t b) { write(&b,1); }
  void u(uint64_t v) { do { uint8_t b=v&127; v>>=7; byte(b|(v?128:0)); } while(v); }
  void fixed(uint64_t v) { for(int i=0;i<8;i++) { byte(v&255); v>>=8; } }
  void summary(const CovSummary& s) { fixed(s.nodes); fixed(s.hash); }
  uint64_t read_fixed() { uint8_t b[8]; read(b,8); uint64_t v=0; for(int i=7;i>=0;i--) v=(v<<8)|b[i]; return v; }
  CovSummary read_summary() { CovSummary s; s.nodes=read_fixed(); s.hash=read_fixed(); return s; }
  void copy_to(FILE* out) {
    rewind_read(); char b[65536]; size_t n;
    while ((n=fread(b,1,sizeof b,f))) if(fwrite(b,1,n,out)!=n) throw runtime_error("coverage publish write failed");
    if(ferror(f)) throw runtime_error("coverage spool read failed");
  }
};

static string cov_quote(const string& s) {
  string out="\"";
  for (unsigned char c:s) {
    if (!c || c>127) throw invalid_argument("coverage instance must be non-NUL ASCII");
    if(c=='"'||c=='\\') { out+='\\'; out+=c; }
    else if(c<32) { char buf[7]; snprintf(buf,sizeof buf,"\\u%04x",c); out+=buf; }
    else out+=c;
  }
  return out+'"';
}
static string cov_read_text(const string& path) {
  ifstream in(path,ios::binary); if(!in) throw invalid_argument("cannot read coverage metadata/instance");
  return string(istreambuf_iterator<char>(in),{});
}

// Deliberately narrow metadata schema; no general JSON library or hash code.
struct CovMetadata {
  string text, group; size_t pos=0;
  vector<vector<int>> permutations;
  void ws() { while(pos<text.size() && (text[pos]==' '||text[pos]=='\n'||text[pos]=='\r'||text[pos]=='\t')) pos++; }
  bool eat(char c) { ws(); if(pos<text.size()&&text[pos]==c){pos++;return true;}return false; }
  void need(char c) { if(!eat(c)) throw invalid_argument("invalid certificate metadata JSON"); }
  string str() {
    need('"'); string s;
    while(pos<text.size()&&text[pos]!='"') {
      unsigned char c=text[pos++]; if(c<32||c>126||c=='\\') throw invalid_argument("metadata strings must be literal ASCII");
      s+=c;
    }
    need('"'); return s;
  }
  int integer() {
    ws(); size_t begin=pos; uint64_t v=0;
    while(pos<text.size()&&text[pos]>='0'&&text[pos]<='9') {
      v=v*10+(text[pos++]-'0'); if(v>4096) throw invalid_argument("permutation index out of range");
    }
    if(begin==pos||(pos>begin+1&&text[begin]=='0')) throw invalid_argument("invalid permutation integer");
    return (int)v;
  }
  CovMetadata(const string& path,const HostCode& h):text(cov_read_text(path)) {
    bool sg=false,sp=false; need('{');
    do {
      string key=str(); need(':');
      if(key=="group_sha256"&&!sg) { group=str(); sg=true; }
      else if(key=="permutations"&&!sp) {
        sp=true; need('[');
        if(!eat(']')) { do {
          need('['); vector<int> p;
          if(!eat(']')) { do { p.push_back(integer()); if(p.size()>(size_t)h.n) throw invalid_argument("long permutation"); } while(eat(',')); need(']'); }
          if(p.size()!=(size_t)h.n) throw invalid_argument("permutation length differs from n");
          vector<int> seen(h.n,0);
          for(int q:p) if(q>=h.n||seen[q]++) throw invalid_argument("permutation is not bijective");
          for(int q=0;q<h.n;q++) if(h.orbit[q]!=h.orbit[p[q]]) throw invalid_argument("permutation crosses declared orbits");
          permutations.push_back(std::move(p));
        } while(eat(',')); need(']'); }
      } else throw invalid_argument("duplicate or unknown certificate metadata key");
    } while(eat(','));
    need('}'); ws();
    if(pos!=text.size()||!sg||!sp||!digest_ok(group)) throw invalid_argument("metadata requires group_sha256 and permutations");
    vector<int> component(h.n); for(int q=0;q<h.n;q++) component[q]=q;
    function<int(int)> root=[&](int q){while(component[q]!=q){component[q]=component[component[q]];q=component[q];}return q;};
    for(const auto& p:permutations) for(int q=0;q<h.n;q++) component[root(q)]=root(p[q]);
    for(int q=0;q<h.n;q++) if(root(q)!=root(h.reps[h.orbit[q]])) throw invalid_argument("permutations do not justify declared orbits");
  }
  string perm_json() const {
    string s="["; bool first=true;
    for(const auto& p:permutations){if(!first)s+=',';first=false;s+='[';
      for(size_t i=0;i<p.size();i++){if(i)s+=',';s+=to_string(p[i]);}s+=']';}
    return s+']';
  }
};

struct CovFound { Found found; };

// The executor's node evaluator: enter_any selects the opt-in rule pass when c.prune (rules build).
static int cov_enter(const Code& c, Dfs& st, int* found) {
#ifdef STABCERT_RULES
  return st.enter_any(c, found);
#else
  return st.enter(c, found);
#endif
}

// Host top/full transcript traversal, using the same node evaluator and changes
// as the executor. Explicit dead cuts expose the branches skipped by next_child.
struct CovWalk {
  const Code c; Dfs& st; CovFile* out;
  uint64_t events=0; CovSummary entries, pending;
  int split; uint32_t root_code; vector<uint8_t> path;
  function<void(uint32_t,const vector<uint8_t>&,uint64_t,CovSummary)> frontier;
  CovWalk(const Code& code,Dfs& state,CovFile* output,int cut):c(code),st(state),out(output),split(cut) {}
  int dead_hint() const { for(int g=0;g<c.m;g++)if(st.sbit(g)&&!st.gcnt[g])return g;throw runtime_error("missing dead hint"); }
  void dead(uint8_t tag,uint64_t parent,uint64_t edge) {
    events++; if(out){out->byte(tag);out->u(parent);out->u(edge);out->u(dead_hint());out->fixed(st.state_fp);}
  }
  void visit(uint64_t parent,uint64_t edge,bool split_dead_root=false) {
    if(st.dead&&!split_dead_root){dead(0x11,parent,edge);return;}
    if(st.w==split) {
      // The callback writes the frontier record with its canonical task ID.
      frontier(root_code,path,st.state_fp,pending); pending={}; events++;
      return;
    }
    int f=-1; int expanded=cov_enter(c,st,&f);
    if(f>=0){CovFound e;take_found(st,e.found);throw e;}
    uint64_t id=events++;
    entries.entry(st);pending.entry(st);
    if(out){
      out->byte(0x10);out->u(parent);out->u(edge);out->byte(st.last_rule);out->fixed(st.state_fp);out->u(st.w);out->u(st.nA);
#ifdef STABCERT_RULES
      // Version 2, rules s0*: packing_size = h_field = |PK|, then PK in PK order (packed[], left by csp_prune).
      if(c.prune&&(st.last_rule==2||st.last_rule>=5)){out->u(st.last_h);for(int i=0;i<st.last_h;i++)out->u(st.packed[i]);}
      else
#endif
      if(st.last_rule==2){uint16_t selected[AMAX];int h=st.bound(c,selected);out->u(h);for(int i=0;i<h;i++)out->u(selected[i]);}
      else if(st.last_rule==3)out->u(st.last_g);
    }
    if(!expanded)return;
    Frame& frame=st.fr[st.depth-1];
    for(int p=st.first_cand(c,frame.g,0);p>=0;p=st.first_cand(c,frame.g,p+1)){
      int q=c.gq[frame.g*c.W+p],a=st.lab[q],opt[2];
      int no=st.options(a,c.gl[frame.g*c.W+p],opt);
      for(int li=0;li<no;li++){
        st.change(c,q,a,opt[li]);path.push_back((p<<1)|li);
        // Give the callback the top parent/edge without storing full node states.
        next_parent=id+1;next_edge=(p<<1)|li;
        visit(id+1,(p<<1)|li);path.pop_back();st.change(c,q,opt[li],a);
      }
      st.modify(c,frame,p);
      if(st.dead){dead(0x12,id+1,p);break;}
    }
    st.unmodify(c,frame);st.depth--;
  }
  uint64_t next_parent=0,next_edge=0;
};

struct CovArena {
  CovSummary value,tail;
  bool refined=false;
  vector<size_t> children;
  vector<CovSummary> before;
};

struct CovOptions {
  string output,metadata,detail="frontier",instance,instance_sha,binary_sha,gpus,rules="greedy";
  int version=1,cap,split,threads,tpb,bps,delta,slots;
  long long chunk,budget,cpu_budget;
  bool quiet;
};

static int emit_coverage(HostCode h,const CovOptions& o) {
  double start=now(); h.trace_enabled=true; Code c=h.view();
  CovMetadata meta(o.metadata,h);
  string instance=cov_read_text(o.instance);
  string domain=h.domain==1?"X":h.domain==2?"Z":"P";
  // Version 2 identifiers and rule set: BNB_CERT_SPEC_v2.md section 0. Version 1 bytes are unchanged.
  string v=to_string(o.version);
  string header="{\"format\":\"stabcert-bnb-coverage/"+v+"\",\"search\":\"irreducible-pauli-dfs/"+v+"\",\"trace\":\"poly64-state-rules/"+v+"\""
    +(o.version==2?",\"rules\":"+cov_quote(o.rules)+",\"rules_version\":1":"")+",\"detail\":"+cov_quote(o.detail)
    +",\"instance_text\":"+cov_quote(instance)+",\"instance_sha256\":"+cov_quote(o.instance_sha)
    +",\"group_sha256\":"+cov_quote(meta.group)+",\"n\":"+to_string(h.n)+",\"m\":"+to_string(h.m)+",\"nl\":"+to_string(h.nl)+",\"k\":"+to_string(h.nl/2)
    +",\"cap\":"+to_string(o.cap)+",\"domain\":"+cov_quote(domain)+",\"split\":"+to_string(o.split)+",\"amax\":96,\"permutations\":"+meta.perm_json()
    +",\"provenance\":{\"source_sha256\":"+cov_quote(STABCERT_SOURCE_SHA256)+",\"binary_sha256\":"+cov_quote(o.binary_sha)+"}}";
  CovFile proof,paths,segments;
  proof.write(o.version==2?"SBCOV2\r\n":"SBCOV1\r\n",8);proof.fixed(header.size());proof.write(header.data(),header.size());
  uint64_t task_count=0,event_count=0,aborted=0,physical=0;
  CovSummary top,global,task_sum;
  Found found;long long replay_fail=0;
  auto ordinal=[&](uint32_t r)->uint64_t{return h.domain?(r>>2):3*(r>>2)+(r&3)-1;};
  try {
    auto st=make_unique<Dfs>(); CovSummary pending;
    for(int oi=0;o.cap>0&&oi<(int)h.reps.size();oi++)for(int l=1;l<=3;l++)if(!h.domain||h.domain==l){
      st->init(c,o.cap);st->root(c,h.orbit.data(),h.reps[oi],oi,l);
      CovWalk w(c,*st,&proof,o.split);w.root_code=(oi<<2)|l;w.events=event_count;w.pending=pending;
      w.next_edge=ordinal(w.root_code);
      w.frontier=[&](uint32_t r,const vector<uint8_t>& p,uint64_t fp,CovSummary prefix){
        proof.byte(0x13);proof.u(w.next_parent);proof.u(w.next_edge);proof.u(task_count);proof.fixed(fp);
        paths.fixed(r);paths.byte(p.size());paths.write(p.data(),p.size());segments.summary(prefix);task_count++;
      };
      w.visit(0,ordinal(w.root_code),o.split==1);
      top.append(w.entries);pending=w.pending;event_count=w.events;
    }
    segments.summary(pending);
    proof.byte(0x1f);proof.u(event_count);proof.u(task_count);proof.u(top.nodes);proof.fixed(top.hash);
    paths.rewind_read();segments.rewind_read();
    printf("{\"event\":\"tasks\",\"cap\":%d,\"split\":%d,\"ntasks\":%llu,\"gen_nodes\":%llu,\"gen_sec\":%.3f}\n",o.cap,o.split,(unsigned long long)task_count,(unsigned long long)top.nodes,now()-start);fflush(stdout);
#ifdef __CUDACC__
    vector<Gpu> devices;
    if(!o.gpus.empty()){
      size_t p=0;while(p<o.gpus.size()){size_t q=o.gpus.find(',',p);if(q==string::npos)q=o.gpus.size();
        for(int slot=0;slot<o.slots;slot++)devices.push_back(gpu_setup(stoi(o.gpus.substr(p,q-p)),h,o.tpb,o.bps));p=q+1;}
    }
#endif
    for(uint64_t first=0;first<task_count;){
      uint64_t count=min<uint64_t>(o.chunk,task_count-first);
      Batch original;original.stride=max(1,o.split-1);
      for(uint64_t i=0;i<count;i++){
        uint32_t r=paths.read_fixed();uint8_t len;paths.read(&len,1);vector<uint8_t> p(len);paths.read(p.data(),len);original.add(r,p.data(),len);
      }
      vector<CovArena> arena(count);
      Batch batch=original;vector<size_t> ids(count);for(size_t i=0;i<count;i++)ids[i]=i;
      while(batch.count()){
        size_t nt=batch.count();physical+=nt;
        vector<long long> nodes(nt);vector<uint64_t> hashes(nt);
        atomic<int> gcap(o.cap);Found round_found;long long rf=0;
#ifdef __CUDACC__
        if(!devices.empty()){
          atomic<long long> next(0);mutex mu;vector<thread> workers;
          long long piece=max<long long>(1,min<long long>(100000,(nt+devices.size()-1)/devices.size()));
          // Adjacent canonical tasks can have very different costs from later
          // roots. Stripe them across pieces without shrinking GPU work queues.
          // Scatter results back before refinement and canonical emission.
          size_t npieces=(nt+piece-1)/piece;
          Batch work;work.stride=batch.stride;
          work.root.reserve(nt);work.len.reserve(nt);work.path.reserve(nt*work.stride);
          vector<size_t> order;order.reserve(nt);
          vector<pair<long long,long long>> ranges;
          for(size_t p=0;p<npieces;p++){
            long long a=work.count();
            for(size_t t=p;t<nt;t+=npieces){
              work.add(batch.root[t],batch.path.data()+t*batch.stride,batch.len[t]);
              order.push_back(t);
            }
            ranges.emplace_back(a,work.count());
          }
          for(size_t di=0;di<devices.size();di++)workers.emplace_back([&,di]{
            while(true){long long p=next.fetch_add(1);if(p>=(long long)ranges.size())break;
              long long a=ranges[p].first,b=ranges[p].second;
              vector<long long> local(b-a);vector<uint64_t> trace(b-a);long long fail=0;Found f;
              gpu_start(devices[di],work,a,b,o.cap,gcap.load(),o.budget);
              gpu_finish(devices[di],h,work,o.cap,a,local,gcap,fail,f,&trace);
              for(long long t=a;t<b;t++){nodes[order[t]]=local[t-a];hashes[order[t]]=trace[t-a];}
              lock_guard<mutex> lock(mu);rf+=fail;if(f.w<round_found.w)round_found=f;
            }
          });
          for(auto& worker:workers)worker.join();
        }else
#endif
        run_cpu(h,batch,o.cap,0,nt,o.threads,gcap,nodes,rf,round_found,o.cpu_budget,&hashes);
        replay_fail+=rf;if(rf)throw runtime_error("coverage task replay failed");
        if(round_found.w<(1<<30)){CovFound e;e.found=round_found;throw e;}
        Batch next;next.stride=min(o.cap,max(1,batch.stride+o.delta));vector<size_t> next_ids;
        for(size_t i=0;i<nt;i++){
          size_t id=ids[i];
          if(nodes[i]>=0){arena[id].value={(uint64_t)nodes[i],hashes[i]};continue;}
          aborted++;arena[id].refined=true;
          st->init(c,o.cap);uint32_t r=batch.root[i];int oi=r>>2;
          if(!st->root(c,h.orbit.data(),h.reps[oi],oi,r&3)||!st->replay(c,batch.path.data()+i*batch.stride,batch.len[i]))throw runtime_error("coverage refinement replay failed");
          int cut=min(o.cap,(int)batch.len[i]+1+o.delta);
          if(cut<=st->w)throw runtime_error("coverage refinement made no progress");
          CovWalk w(c,*st,nullptr,cut);w.root_code=r;w.path.assign(batch.path.begin()+i*batch.stride,batch.path.begin()+i*batch.stride+batch.len[i]);
          w.frontier=[&](uint32_t root,const vector<uint8_t>& path,uint64_t,CovSummary prefix){
            size_t child=arena.size();arena.emplace_back();arena[id].children.push_back(child);arena[id].before.push_back(prefix);
            next.add(root,path.data(),path.size());next_ids.push_back(child);
          };
          w.visit(0,0);arena[id].tail=w.pending;
        }
        batch=std::move(next);ids=std::move(next_ids);
      }
      for(size_t i=arena.size();i-->0;)if(arena[i].refined){
        CovSummary s;
        for(size_t j=0;j<arena[i].children.size();j++){s.append(arena[i].before[j]);s.append(arena[arena[i].children[j]].value);}
        s.append(arena[i].tail);arena[i].value=s;
      }
      for(size_t i=0;i<count;i++){
        uint32_t r=original.root[i];uint8_t len=original.len[i];const uint8_t* path=original.path.data()+i*original.stride;
        CovSummary s=arena[i].value;
        if(o.detail=="full"){
          proof.byte(0x21);proof.u(first+i);
          st->init(c,o.cap);bool alive=st->root(c,h.orbit.data(),h.reps[r>>2],r>>2,r&3);
          if(alive&&!st->replay(c,path,len))throw runtime_error("full transcript replay failed");
          if(!alive&&len)throw runtime_error("dead full transcript ancestor");
          CovWalk full(c,*st,&proof,DMAXW+1);full.visit(0,0);
          if(full.entries.nodes!=s.nodes||full.entries.hash!=s.hash)throw runtime_error("full transcript disagrees with executor");
          proof.byte(0x22);proof.u(full.events);
        }
        proof.byte(0x20);proof.u(first+i);proof.u(ordinal(r));proof.u(len);proof.write(path,len);proof.u(s.nodes);proof.fixed(s.hash);
        global.append(segments.read_summary());global.append(s);task_sum.append(s);
      }
      first+=count;
      if(!o.quiet){
        double elapsed=now()-start;
        uint64_t completed=top.nodes+task_sum.nodes;
        printf("{\"event\":\"progress\",\"completed_tasks\":%llu,\"ntasks\":%llu,\"completed_nodes\":%llu,\"nodes_per_second\":%.3f,\"aborted_total\":%llu,\"elapsed\":%.3f}\n",
          (unsigned long long)first,(unsigned long long)task_count,(unsigned long long)completed,
          completed/elapsed,(unsigned long long)aborted,elapsed);
        fflush(stdout);
      }
    }
    global.append(segments.read_summary());
    if(global.nodes!=top.nodes+task_sum.nodes)throw runtime_error("coverage composition count mismatch");
    proof.byte(0x7f);proof.u(task_count);proof.u(top.nodes);proof.u(task_sum.nodes);proof.u(global.nodes);proof.fixed(global.hash);
    // Same-directory staging makes publication atomic; an existing artifact is
    // replaced only after the complete new stream has been flushed successfully.
    string temp=o.output+".tmp.XXXXXX";vector<char> name(temp.begin(),temp.end());name.push_back(0);
    int fd=mkstemp(name.data());if(fd<0)throw runtime_error("cannot create certificate destination");
    FILE* dest=fdopen(fd,"wb");if(!dest){close(fd);unlink(name.data());throw runtime_error("cannot open certificate destination");}
    try {proof.copy_to(dest);if(fflush(dest)||fsync(fd))throw runtime_error("certificate flush failed");
      if(fclose(dest)){dest=nullptr;throw runtime_error("certificate close failed");}dest=nullptr;
      if(rename(name.data(),o.output.c_str()))throw runtime_error("certificate atomic publication failed");
    }catch(...){if(dest)fclose(dest);unlink(name.data());throw;}
  }catch(const CovFound& e){found=e.found;}
  bool hit=found.w<(1<<30);
  printf("{\"event\":\"done\",\"interface\":1,\"instance_sha256\":%s,\"domain\":%s,\"delta\":%d,\"executor\":%s,\"resumed\":false,\"stopped_on_find\":%s,\"cap0\":%d,\"split\":%d,\"tasks_total\":%llu,\"aborted_total\":%llu,\"gen_nodes\":%llu,\"task_nodes\":%llu,\"nodes\":%llu,\"replay_fail\":%lld,\"final_cap\":%d,\"found\":%d,\"sec\":%.3f,\"certificate_complete\":%s,\"global_trace\":\"%016llx\",\"witness\":[",
    cov_quote(o.instance_sha).c_str(),cov_quote(domain).c_str(),o.delta,cov_quote(o.gpus.empty()?"cpu":"gpu").c_str(),stabcert::json_bool(hit),o.cap,o.split,
    (unsigned long long)physical,(unsigned long long)aborted,(unsigned long long)top.nodes,(unsigned long long)task_sum.nodes,(unsigned long long)(top.nodes+task_sum.nodes),replay_fail,o.cap,hit?found.w:-1,now()-start,stabcert::json_bool(!hit),(unsigned long long)global.hash);
  for(size_t i=0;i<found.wit.size();i++)printf("%s[%d,%d]",i?",":"",found.wit[i].first,found.wit[i].second);
  printf("]}\n");return 0;
}
