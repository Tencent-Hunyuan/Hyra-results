// ============================================================================
// trainer.h - AlphaZero gradient + AdaGrad optimizer for the az_mcts pattern net.
//
// The net is a SPARSE LINEAR model, so the gradients are exact and cheap:
//
//   VALUE  v = tanh(vscale * s),  s = vbias[ph] + Σ tab[id] + Σ vdense[ph][i]*df[i]
//     loss_v = (v - z)^2
//     ds = 2(v - z) * vscale * (1 - v^2)            [dloss/ds]
//     grad: vbias[ph] += ds ; tab[id] += ds (each active id) ; vdense += ds*df
//
//   POLICY logits l_a = pbias[ph][sq_a] + Σ pw[i]*mf_a[i];  p = softmax(l/ptemp)
//     loss_p = -Σ_a pi_a log p_a                     [cross-entropy to MCTS visits]
//     dl_a = (p_a - pi_a) / ptemp
//     grad: pbias[ph][sq_a] += dl_a ; pw[i] += Σ_a dl_a * mf_a[i]
//
// AdaGrad gives every parameter its own learning rate (essential for pattern
// tables, whose entries are hit at wildly different frequencies); updates are
// SPARSE (only parameters touched by the minibatch move), so a step costs O(active
// features), not O(1.67M). A tiny lazy L2 decay on touched value-table entries
// keeps rarely-seen weights from drifting. ptemp is FIXED during training (a
// hyperparameter), not learned, for stability.
//
// A finite-difference gradient check (trainer_check) verifies this math against
// the net.h forward, so a sign/scale slip can't silently waste a training run.
// ============================================================================
#ifndef AZ_MCTS_TRAINER_H
#define AZ_MCTS_TRAINER_H

#include "net.h"
#include <vector>
#include <cmath>
#include <cstring>

namespace az {

struct Grads {  // AdaGrad sum-of-squared-grad accumulators, mirroring Net shapes
    std::vector<float> vtab, vdense, vbias, pbias, pw;
    void init() {
        vtab.assign((size_t)NPHASE * g_phase_block, 0.f);
        vdense.assign((size_t)NPHASE * NDENSE, 0.f);
        vbias.assign(NPHASE, 0.f);
        pbias.assign((size_t)NPHASE * 64, 0.f);
        pw.assign(NMOVEF, 0.f);
    }
};

struct TrainCfg {
    float lr_v = 0.02f;     // AdaGrad base step (value)
    float lr_p = 0.02f;     // AdaGrad base step (policy)
    float l2 = 1e-7f;       // lazy L2 on touched value-table entries
    float eps = 1e-8f;      // AdaGrad epsilon
    float v_weight = 1.0f;
    float p_weight = 1.0f;
};

// Initialise a net's parameters. If warm=true, set a hand-eval-like prior so MCTS
// has useful guidance from step 0 (mobility>0, frontier<0 for value; corners /
// fewer-opp-replies / positional for policy). Pattern tables start at 0.
static inline void net_init_params(Net &nt, bool warm) {
    nt.ok = true;
    nt.nphase = NPHASE; nt.npat = g_npat; nt.phase_block = g_phase_block;
    nt.ndense = NDENSE; nt.nmovef = NMOVEF;
    nt.vtab.assign((size_t)NPHASE * g_phase_block, 0.f);
    nt.vdense.assign((size_t)NPHASE * NDENSE, 0.f);
    nt.vbias.assign(NPHASE, 0.f);
    nt.pbias.assign((size_t)NPHASE * 64, 0.f);
    nt.pw.assign(NMOVEF, 0.f);
    nt.vscale = 0.02f;
    nt.ptemp = 1.0f;
    if (warm) {
        for (int ph = 0; ph < NPHASE; ph++) {
            nt.vdense[ph * NDENSE + 0] = 0.30f;   // actual mobility (+)
            nt.vdense[ph * NDENSE + 1] = 0.10f;   // potential mobility (+)
            nt.vdense[ph * NDENSE + 3] = -0.10f;  // frontier mine-opp (-)
        }
        nt.pw[0] = 0.20f;   // flips
        nt.pw[1] = 2.00f;   // corner capture
        nt.pw[2] = -1.50f;  // X-square (bad)
        nt.pw[3] = -0.40f;  // opponent mobility after (lower better)
        nt.pw[5] = 0.50f;   // positional SQW
    }
}

// One SGD step on a single sample. P,O are side-to-move; z in {-1,0,1} is the game
// result from that side's perspective; (mv[0..nm), pi[0..nm)) is the MCTS visit
// target (probabilities summing to ~1). Returns {value_loss, policy_loss}.
struct StepLoss { double vl, pl; };
static inline StepLoss train_step(Net &nt, Grads &g, const TrainCfg &cfg,
                                  u64 P, u64 O, float z,
                                  int nm, const int *mv, const float *pi) {
    int ph = phase_of(popc(P | O));

    // ---- value forward ----
    int ids[MAXFEAT];
    int nids = value_feature_ids(P, O, ph, ids);
    int base = ph * g_phase_block;
    float *tab = nt.vtab.data() + (size_t)ph * nt.phase_block;
    double s = nt.vbias[ph];
    for (int i = 0; i < nids; i++) s += tab[ids[i] - base];
    float df[NDENSE]; dense_feats(P, O, df);
    float *dw = nt.vdense.data() + (size_t)ph * NDENSE;
    for (int i = 0; i < NDENSE; i++) s += dw[i] * df[i];
    double v = std::tanh(s * nt.vscale);
    double vl = (v - z) * (v - z);

    double ds = 2.0 * (v - z) * nt.vscale * (1.0 - v * v) * cfg.v_weight;

    // ---- value AdaGrad update (sparse) ----
    {
        float *avb = &g.vbias[ph];
        *avb += (float)(ds * ds);
        nt.vbias[ph] -= (float)(cfg.lr_v * ds / (std::sqrt(*avb) + cfg.eps));
        float *atab = g.vtab.data() + (size_t)ph * g_phase_block;
        for (int i = 0; i < nids; i++) {
            int li = ids[i] - base;
            atab[li] += (float)(ds * ds);
            double step = cfg.lr_v * ds / (std::sqrt(atab[li]) + cfg.eps);
            tab[li] -= (float)step;
            tab[li] -= cfg.l2 * tab[li];        // lazy L2 on touched entries
        }
        float *adw = g.vdense.data() + (size_t)ph * NDENSE;
        for (int i = 0; i < NDENSE; i++) {
            double gi = ds * df[i];
            adw[i] += (float)(gi * gi);
            dw[i] -= (float)(cfg.lr_v * gi / (std::sqrt(adw[i]) + cfg.eps));
        }
    }

    // ---- policy forward (softmax over the sample's legal moves) ----
    double pl = 0;
    if (nm > 1) {
        float logit[MAXMV], mf[MAXMV][NMOVEF];
        float maxl = -1e30f;
        for (int a = 0; a < nm; a++) {
            move_feats(P, O, mv[a], mf[a]);
            double l = nt.pbias[(size_t)ph * 64 + mv[a]];
            for (int i = 0; i < NMOVEF; i++) l += nt.pw[i] * mf[a][i];
            logit[a] = (float)(l / nt.ptemp);
            if (logit[a] > maxl) maxl = logit[a];
        }
        float pp[MAXMV]; double zsum = 0;
        for (int a = 0; a < nm; a++) { pp[a] = std::exp(logit[a] - maxl); zsum += pp[a]; }
        for (int a = 0; a < nm; a++) pp[a] /= (float)zsum;
        for (int a = 0; a < nm; a++) if (pi[a] > 0) pl -= pi[a] * std::log(pp[a] + 1e-12f);

        // gradient dl_a = (pp_a - pi_a)/ptemp ; accumulate pw grad over moves
        double gpw[NMOVEF]; for (int i = 0; i < NMOVEF; i++) gpw[i] = 0;
        for (int a = 0; a < nm; a++) {
            double dl = ((double)pp[a] - pi[a]) / nt.ptemp * cfg.p_weight;
            // pbias update (sparse, per (phase,sq))
            size_t bi = (size_t)ph * 64 + mv[a];
            g.pbias[bi] += (float)(dl * dl);
            nt.pbias[bi] -= (float)(cfg.lr_p * dl / (std::sqrt(g.pbias[bi]) + cfg.eps));
            for (int i = 0; i < NMOVEF; i++) gpw[i] += dl * mf[a][i];
        }
        for (int i = 0; i < NMOVEF; i++) {
            g.pw[i] += (float)(gpw[i] * gpw[i]);
            nt.pw[i] -= (float)(cfg.lr_p * gpw[i] / (std::sqrt(g.pw[i]) + cfg.eps));
        }
    }
    return {vl, pl};
}

} // namespace az

#endif // AZ_MCTS_TRAINER_H
