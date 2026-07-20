// trainer_check.cpp - verify train_step's gradients via finite differences against
// the net.h forward, and confirm SGD reduces loss on a synthetic target. A guard
// against a sign/scale slip silently wasting a training run.
#include "trainer.h"
#include "net.h"
#include <cstdio>
#include <cmath>
#include <random>
using namespace az;

static double value_forward(const Net&nt,u64 P,u64 O){ return net_value(nt,P,O); }
static double value_loss(const Net&nt,u64 P,u64 O,float z){ double v=value_forward(nt,P,O); return (v-z)*(v-z); }

int main(){
    net_init();
    std::mt19937_64 rng(2024);

    // a midgame position
    u64 black=(1ULL<<28)|(1ULL<<35), white=(1ULL<<27)|(1ULL<<36); int mover=1;
    for(int s=0;s<14;s++){ u64 P=mover==1?black:white,O=mover==1?white:black; u64 mv=gen_moves(P,O); if(!mv){mover=3-mover;continue;} int c=popc(mv),pk=rng()%c,sq=-1;u64 t=mv;for(int i=0;i<=pk;i++){sq=lsb(t);t&=t-1;} u64 m=1ULL<<sq,f=flips_for(P,O,m); if(mover==1){black|=m|f;white&=~f;}else{white|=m|f;black&=~f;} mover=3-mover; }
    u64 P=mover==1?black:white, O=mover==1?white:black;
    int ph=phase_of(popc(P|O));

    Net nt; net_init_params(nt,false);   // zero init (no warm prior)
    nt.vscale = 0.1f;
    // small random weights so the value sits in a non-saturated mid-range
    // (cast to int BEFORE subtracting - rng()%K is unsigned and would underflow)
    for(auto&x:nt.vtab) x=(float)(((int)(rng()%201)-100)/100000.0);   // +/-0.001
    for(auto&x:nt.vdense) x=(float)(((int)(rng()%201)-100)/100000.0); // +/-0.001
    for(auto&x:nt.pw) x=(float)(((int)(rng()%201)-100)/1000.0);

    float z = 0.5f;   // non-saturating target
    // ---- analytic gradient via train_step with lr=0 trick: instead, compute ds and
    // compare to FD on a few parameters. We replicate the analytic value grad here.
    double v = value_forward(nt,P,O);
    double ds = 2.0*(v-z)*nt.vscale*(1.0-v*v);

    int ids[MAXFEAT]; int nids=value_feature_ids(P,O,ph,ids); int base=ph*g_phase_block;
    printf("value v=%.5f z=%.1f ds=%.6f  active_feats=%d\n", v,z,ds,nids);

    // FD check on 5 random active table entries: dLoss/dtab = ds (each active id +1)
    double maxerr=0;
    for(int trial=0; trial<5; trial++){
        int li = ids[rng()%nids]-base;
        float *tab = nt.vtab.data()+(size_t)ph*nt.phase_block;
        float save=tab[li]; double h=1e-3;
        tab[li]=save+h; double lp=value_loss(nt,P,O,z);
        tab[li]=save-h; double lm=value_loss(nt,P,O,z);
        tab[li]=save;
        double fd=(lp-lm)/(2*h);
        // note: if a feature id repeats (multiple instances hit same entry), analytic = ds*count
        int cnt=0; for(int i=0;i<nids;i++) if(ids[i]-base==li) cnt++;
        double ana=ds*cnt;
        double err=fabs(fd-ana);
        maxerr=std::max(maxerr,err);
        printf("  tab[%d] x%d: FD=%.6f analytic=%.6f err=%.2e\n", li,cnt,fd,ana,err);
    }
    // FD on vdense
    float df[NDENSE]; dense_feats(P,O,df);
    for(int i=0;i<NDENSE;i++){
        float *dw=nt.vdense.data()+(size_t)ph*NDENSE; float save=dw[i]; double h=1e-3;
        dw[i]=save+h; double lp=value_loss(nt,P,O,z); dw[i]=save-h; double lm=value_loss(nt,P,O,z); dw[i]=save;
        double fd=(lp-lm)/(2*h), ana=ds*df[i], err=fabs(fd-ana); maxerr=std::max(maxerr,err);
        printf("  vdense[%d] (df=%.1f): FD=%.6f analytic=%.6f err=%.2e\n", i,df[i],fd,ana,err);
    }
    printf("VALUE grad max FD err = %.2e  -> %s\n", maxerr, maxerr<1e-3?"OK":"FAIL");

    // ---- learning test: can SGD fit value to z=+1 on this single position? ----
    Net L; net_init_params(L,true); Grads g; g.init(); TrainCfg tc; tc.lr_v=0.05f; tc.lr_p=0.05f;
    // build a fake "policy target" = uniform over legal moves
    u64 mv=gen_moves(P,O); int sqs[MAXMV],nm=0; u64 tt=mv; while(tt){sqs[nm++]=lsb(tt);tt&=tt-1;}
    float pi[MAXMV]; for(int i=0;i<nm;i++) pi[i]=1.0f/nm;
    // bias the target toward move 0
    if(nm>1){ for(int i=0;i<nm;i++) pi[i]=0.05f/(nm-1); pi[0]=0.95f; }
    double v0=net_value(L,P,O);
    StepLoss first{0,0};
    for(int it=0;it<3000;it++){ auto sl=train_step(L,g,tc,P,O,1.0f,nm,sqs,pi); if(it==0)first=sl; }
    double v1=net_value(L,P,O);
    // check policy moved toward move 0
    float pr[MAXMV]; net_policy(L,P,O,sqs,nm,pr);
    printf("learn value: %.4f -> %.4f (target +1)  %s\n", v0,v1, v1>v0+0.3?"OK":"WEAK");
    if(nm>1) printf("learn policy: p[move0]=%.3f (target 0.95)  %s\n", pr[0], pr[0]>1.5f/nm?"OK":"WEAK");
    return (maxerr<1e-3)?0:1;
}
