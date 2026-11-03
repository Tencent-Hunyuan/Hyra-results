// S2 (slack two) for the opt-in --bound policy greedy+s0s1s2. Included after struct Dfs (core.h); host-only.
// Ported verbatim from the reviewed S2 core, frozen/v1-c54147fb/fast.inc (sha256 126594f8...).
// [port] changes: this header comment; the S2AUDIT statistics counters removed; S2_HT; exact quotient membership
// (fingerprint matches confirmed by reduction modulo W). Soundness: docs/SOUNDNESS.md 5.1.
// CPU-only S2 affine-quotient relaxation. Every domain used here is ORIGINAL
// at this search node: no propagated domains, killer chains, or S1 caches.
// [port] The linear fingerprint is only a filter: every match is confirmed by exact membership in W,
// so the decision is independent of the keys. Failure to find an E of size <=2 proves infeasibility.
#include <cstring>
#ifndef S2_HT
#define S2_HT 8192   // [port] hash-table size (reviewed value)
#endif
struct S2Scratch {
  static const int NW=(MMAX+63)/64, HT=S2_HT;
  uint64_t basis[MMAX][NW], key[MMAX], hk[HT];
  uint32_t epoch=0, be[MMAX]={}, he[HT]={};
  int hhead[HT], aq[HT], al[HT], anext[HT]; // [port] chains of actions per fingerprint (at most 3n <= HT/2)
  static uint64_t mix(uint64_t x){x+=0x9e3779b97f4a7c15ull;x=(x^(x>>30))*0xbf58476d1ce4e5b9ull;x=(x^(x>>27))*0x94d049bb133111ebull;return x^(x>>31);}
  int lookup(uint64_t k)const{int p=mix(k)&(HT-1);while(he[p]==epoch&&hk[p]!=k)p=(p+1)&(HT-1);return p;}
  // [port] sigma of action (q,l): the generators it flips
  static void action(const Code&c,int q,int l,uint64_t*v,int nw){for(int k=0;k<nw;++k)v[k]=0;for(int j=0;j<c.qc[q];++j)if(c.ql[q*c.C+j]!=l){int g=c.qg[q*c.C+j];v[g>>6]^=1ull<<(g&63);}}
  // [port] exact membership in W: reduce by the lowest-pivot basis (as in the insertion loop) and test for zero
  bool inW(uint64_t*v,int nw)const{int word=0;while(true){while(word<nw&&!v[word])++word;if(word==nw)return true;int pivot=word*64+__builtin_ctzll(v[word]);
      if(be[pivot]!=epoch)return false;for(int k=word;k<nw;++k)v[k]^=basis[pivot][k];}}
  int refute(const Code&c,const Dfs&st,int h){
#ifdef STABCERT_RULE_LIMIT
    if(h<0||h>DMAX||c.n*3>HT/2){STABCERT_RULE_LIMIT(c.n*3>HT/2?"s2_table":"s2_h_range");return 0;}
#else
    if(h<0||h>DMAX||c.n*3>HT/2)return 0;
#endif
    if(++epoch==0){memset(be,0,sizeof(be));memset(he,0,sizeof(he));epoch=1;}
    const int nw=(c.m+63)/64;
    uint64_t target[NW]={};
    for(int g=0;g<c.m;++g)if(st.sbit(g))target[g>>6]|=1ull<<(g&63);
    for(int i=0;i<h;++i){int g=st.packed[i];uint64_t base[NW]={};bool first=true;
      for(int p=0;p<c.gw[g];++p){int q=c.gq[g*c.W+p],gl=c.gl[g*c.W+p],a=st.lab[q];if(!cand(a,gl))continue;
        for(int l=1;l<=3;++l){if(l==gl||(a>LFREE&&a-LFREE!=l))continue;
          uint64_t v[NW]={};for(int j=0;j<c.qc[q];++j)if(c.ql[q*c.C+j]!=l){int k=c.qg[q*c.C+j];v[k>>6]^=1ull<<(k&63);}
          if(first){first=false;for(int k=0;k<nw;++k){base[k]=v[k];target[k]^=v[k];}continue;}
          for(int k=0;k<nw;++k)v[k]^=base[k];
          int word=0;while(true){while(word<nw&&!v[word])++word;if(word==nw)break;int pivot=word*64+__builtin_ctzll(v[word]);
            if(be[pivot]!=epoch){be[pivot]=epoch;for(int k=word;k<nw;++k)basis[pivot][k]=v[k];break;}
            for(int k=word;k<nw;++k)v[k]^=basis[pivot][k];
          }
        }
      }
#ifdef STABCERT_RULE_LIMIT
      if(first){STABCERT_RULE_LIMIT("s2_empty_domain");return 0;} // Conservative guard; a valid live packing is nonempty.
#else
      if(first)return 0; // Conservative guard; a valid live packing is nonempty.
#endif
    }
    // Lowest-pivot basis: each row's other entries have strictly larger index.
    // This constructs a linear map that annihilates every difference vector.
    for(int g=c.m-1;g>=0;--g){
      if(be[g]!=epoch){key[g]=mix(g+17);continue;}
      uint64_t z=0;int word=g>>6;
      for(int k=word;k<nw;++k){uint64_t b=basis[g][k];if(k==word)b&=~(1ull<<(g&63));while(b){int j=k*64+__builtin_ctzll(b);b&=b-1;z^=key[j];}}
      key[g]=z;
    }
    uint64_t goal=0;for(int k=0;k<nw;++k)for(uint64_t b=target[k];b;b&=b-1)goal^=key[k*64+__builtin_ctzll(b)];
    uint64_t v[NW],u[NW];
    // [port] E empty survives iff tau is in W (goal == 0 is necessary; confirm exactly)
    if(!goal){for(int k=0;k<nw;++k)v[k]=target[k];if(inW(v,nw))return 0;}
    int na=0;
    for(int q=0;q<c.n;++q){int a=st.lab[q];if(a<LFREE)continue;
      for(int l=1;l<=3;++l){if(a>LFREE&&a-LFREE!=l)continue;uint64_t z=0;
        for(int j=0;j<c.qc[q];++j)if(c.ql[q*c.C+j]!=l)z^=key[c.qg[q*c.C+j]];
        // [port] one extra (q,l) survives iff sigma(q,l) XOR tau is in W
        if(z==goal){action(c,q,l,v,nw);for(int k=0;k<nw;++k)v[k]^=target[k];if(inW(v,nw))return 0;}
        // [port] a pair with an earlier action on a different qubit survives iff sigma(a) XOR sigma(b) XOR tau is in W
        int p=lookup(z^goal);
        if(he[p]==epoch)for(int b=hhead[p];b>=0;b=anext[b]){if(aq[b]==q)continue;
          action(c,q,l,v,nw);action(c,aq[b],al[b],u,nw);for(int k=0;k<nw;++k)v[k]^=u[k]^target[k];if(inW(v,nw))return 0;}
        p=lookup(z);if(he[p]!=epoch){he[p]=epoch;hk[p]=z;hhead[p]=-1;}
        aq[na]=q;al[na]=l;anext[na]=hhead[p];hhead[p]=na;na++;
      }
    }
    return 1;
  }
};
inline int s2_extra(const Code&c,Dfs&st,int h){
  static thread_local S2Scratch scratch;
  int result=scratch.refute(c,st,h);
  return result;
}
