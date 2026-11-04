# hyqec-check 2.0.2 release

Version 2.0.2 corrects documentation and updates version metadata. Internal
conformance and process reports are retained outside distributed contents.
The public validation summary states the results and their scope. Remaining
development branch names and references to excluded local evidence paths are
removed from the public documentation. The current node/VERIFY command uses
an existing build target and works from plain source exports without Git.
External conformance data and optional baseline drivers are identified. The
unchanged v1 specification and companion arithmetic/example vectors are now
included at the paths cited by the frozen source and specification.

The distribution and Python v2 profile report 2.0.2. Verifier logic and native
source are unchanged; all four native executables remain byte-identical to
2.0.0. The native v2 executable retains its 2.0.0 identity, and v1 profiles
remain frozen at 1.3.0. The only Python profile change is its version constant;
the existing CLI-version test expectation is updated accordingly.

The short release gates build source and wheel packages, install each in a
fresh environment, run all 64 installed tests, check licence/version metadata,
compare every native payload byte with 2.0.0, verify the 30 frozen-file hashes,
and scan source, wheel and canonical export contents with the frozen publication
scanner. The documentation audit checks shipped branch references and command
prerequisites and references into excluded or missing paths; the current
vector command is exercised from a plain export.

Copyright (C) 2026 Tencent, Apache-2.0. Licence text and third-party notices are
unchanged. The 2.0.0 certificate, vector, 888-search node-total and sanitizer
evidence carries forward under unchanged native code and executable bytes.
See [PUBLIC_VALIDATION.md](docs/PUBLIC_VALIDATION.md) for the validation scope.
