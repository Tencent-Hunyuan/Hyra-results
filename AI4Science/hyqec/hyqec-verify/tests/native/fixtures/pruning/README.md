# Independent pruning counterexamples

These three files are preserved byte-for-byte from the companion exact-distance study's
independent S1 review. `provenance.json` records their origins and SHA-256
digests. They are regression data, not a dependency on that implementation.

The lifted counterexample is an [[18,1,3]] code with independent commuting
generators. Its logical `Z15 X16 Z17` has weight 3, but a stale ownership cache
in the reviewed S1 implementation discarded it and falsely reported no logical
at cap 3. That implementation had passed 210 reference and 10,000 random
whole-distance cases. The release does not include that pruning rule.

`cache-repro.inst` and `chain.inst` preserve smaller generator presentations,
including dependent rows. Their old files have zero supplied logical rows.
The distance tests add a complete, validated logical basis while retaining
the generator order, support order and trivial orbit/root order. Treating the
old zero-row logical lists as distance inputs would make the test vacuous.

`test_pruning.py` exercises every shipped policy, CPU and manual GPU, at cap 3
and below the exact distance, with both splits and forced refinements. The
cut auditor also includes all three presentations in P/X/Z domains.

Source: the independent S1 review's finding report.
