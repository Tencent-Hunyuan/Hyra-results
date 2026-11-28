# Research notebook example

This sanitized snapshot captures a research run on the Grünbaum–Loewner centroid-section theorem. The walkthrough connects the original problem, proof routes, proofs, and verification records. Its links point to the problem statement and notebook saved during the run. [Source theorem](https://arxiv.org/abs/2606.19865v2)

## Reading the proof and research records

1. **Read the problem.** In each dimension $n\ge 3$, construct a smooth, strictly convex body with exactly one hyperplane section whose centroid is the body's centroid. Let $\mu(n)$ be the largest integer that bounds the number of such sections from below for every $n$-dimensional convex body. Show that $\mu(2)=3$ and $\mu(n)=1$ for $n\ge 3$. The lower bound includes nonsmooth convex bodies. [Full statement and definitions](research-notebook/PROBLEM.md)
2. **Trace the premises of the final proof.** Follow F013's `Depends on` links to its direct predecessors. F003 proves the planar upper and lower bounds, F008 gives the lower bound for all convex bodies, and F009 constructs the smooth, strictly convex examples in higher dimensions. The `Proof` link on each entry opens the full proof. [F013 final argument](research-notebook/KNOWLEDGE.md#fact-846e296f22ef7853)
3. **Compare proof routes.** The main route starts with a spherical twist construction. It uses finite spherical harmonic approximation and inversion, exact centroid control, and critical-point stability to complete the higher-dimensional construction. The notebook also records geometric shears and conditional dimension lifting as alternatives. [Route R002](research-notebook/KNOWLEDGE.md#entry-f5a5afb3f0a8d294) · [Other routes](research-notebook/KNOWLEDGE.md#route)
4. **Compare publication and completion records.** V004 records a candidate the verifier judged `correct` but the service could not publish because of a terminology conflict. V016 records publication of F013. V018 records the independent review of all dependencies and the argument answering the original problem. [V004 submission feedback](research-notebook/KNOWLEDGE.md#entry-7d0fc2492a0547d7) · [V016 publication feedback](research-notebook/KNOWLEDGE.md#entry-c72215b98f9c410b) · [V018 completion feedback](research-notebook/KNOWLEDGE.md#entry-1f7592f4059e7fcc)

The labels `F/O/R/C/E/L/V/P` stand for facts, overviews, routes, conjectures, experience, leads, verification feedback, and reports. Active published facts supply proof premises. The other records document the research process. [Notebook structure](../docs/ARCHITECTURE.md#notebook-layout)

## Two entries to follow

**C001, from an intermediate conjecture to F010.** C001 isolates an analytic step in the construction of a convex body with exactly one centroid-matching section. For each $n\ge 3$, find a finite real linear combination of spherical harmonics of odd degree at least three whose hemispherical sign transform has exactly two critical points, both nondegenerate and antipodal. F010 starts with a smooth potential from F001, uses F007 to approximate it while preserving a zero first moment and one antipodal pair of nondegenerate critical points, and applies F002 to invert the transform. When the service published the independently verified F010, it marked C001 as `proved`. P016 records the scope of verification and manual review for the analytic route. [C001 conjecture](research-notebook/KNOWLEDGE.md#entry-97292e71c3b56db4) · [F010 published fact](research-notebook/KNOWLEDGE.md#fact-c7fafba4b4d5a72f) · [P016 audit report](research-notebook/KNOWLEDGE.md#entry-728de1323489ba0c)

**E001, reviewing a published proof.** This `review` entry records a concern about Part 6 of F011 at revision 1. The scaling estimate there needs an argument that the section and its relative interior have equal volume. The independent review recorded in V017 found that the earlier proof already showed that strict contractions lie in the relative interior, which gives the required volume equality. The receipt returned `accepted=false` and `applied=false` for the requested revocation, so F011 remained `verified`. E001 records the revision, review scope, and clarification. [E001 review entry](research-notebook/KNOWLEDGE.md#entry-683f18cbb33b8f19) · [F011 published fact](research-notebook/KNOWLEDGE.md#fact-81a5afe69f10c3d2) · [V017 review feedback](research-notebook/KNOWLEDGE.md#entry-ffad014f49254883)

The snapshot uses `review` entries for experience. HyProver also uses `method` for reusable techniques and `boundary` for conditions and limits of applicability. [Experience and audits](../docs/concepts.md)

## Run provenance

The run used `gpt-6-astra` with eight `max` workers, a main agent, and independent verifiers. The dependency closure of F013 contains nine proofs. A tenth review checked how the established facts answer the entire original problem. The snapshot preserves the original mathematical statements, 13 proofs, 23 notes, content IDs, dependencies, and historical verdicts. Generic role names identify the authors. [Project metadata](research-notebook/project.json)

The research run began in a fresh context with external search disabled. An earlier extraction of the problem had returned material beyond the theorem statement. A subsequent comparison found the main route closely matched the source paper. The snapshot shows how a proof was organized, published, and reviewed. Its verification records reflect the LLM reviews from the original run.

To start a new research project, create it with `hyprover new`, enter the problem statement, and either start the background main agent or assign work manually. [Operations guide](../docs/operating-guide.md)

## Checking the snapshot

After installing the development dependencies, run the following from the repository root to check formatting, content IDs, proof dependencies, links, and sanitization. The workflow tests use a simulated backend.

```bash
python -m pytest hyprover/tests/test_examples.py hyprover/tests/test_knowledge_workflow.py -q
```
