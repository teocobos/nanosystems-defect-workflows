# AiiDA provenance validation — LUMI-C SiO2

## Validation scope

The existing alpha-quartz SiO2 CP2K cutoff campaign was inspected
against the local AiiDA profile `nsdw-dev`.

This was a read-only inspection. No calculations were submitted,
modified, or linked retrospectively.

## Verified calculation identities

| Cutoff (Ry) | AiiDA PK | UUID | Finished OK | CALL_CALC parent |
|---|---:|---|---|---|
| 300 | 2027 | e559dbd6-3249-4309-a951-07d53b2630c5 | Yes | None |
| 400 | 2029 | 927207c2-4beb-4e80-a8a7-de8271510296 | Yes | None |
| 500 | 2031 | 2f97f607-7d90-4afc-85f0-83a95d7ae93f | Yes | None |
| 600 | 2022 | 04f07448-68ea-47eb-bd4b-060b53f22aa5 | Yes | None |
| 800 | 2033 | 88b8a799-639a-4c0b-b009-7e60bfaf9b7b | Yes | None |

Every node was a `CalcJobNode` with process type
`aiida.calculations:cp2k`.

## Provenance classification

**Legacy independent campaign.** All five calculations have no
incoming AiiDA `CALL_CALC` links. They must not be represented as
children of an NSDW convergence WorkChain.

Their existing calculation-level AiiDA provenance and reconciled
campaign evidence remain valid for legacy reporting.

## WorkChain-bound campaigns

For registrations carrying both a WorkChain UUID and scientific
specification SHA-256, NSDW requires:

1. A real AiiDA WorkChain with matching scientific specification.
2. Matching candidate identities and CP2K cutoff parameters.
3. Direct `CALL_CALC` links to every campaign calculation.
4. Exact PK, UUID and cutoff correspondence.
5. Existing completion and reporting checks before publication.

Failure of either provenance gate blocks report generation.

This strict verifier currently supports direct `CALL_CALC` links,
not nested WorkChain traversal.

## Validation status

- Focused reporting and provenance tests: 40 passed.
- Full NSDW regression: 1,622 passed, 31 deselected,
  65 nonfatal warnings.
- Actual LUMI-C campaign: calculation-level identities inspected.
- End-to-end real WorkChain-bound HPC campaign: not yet validated.

The existing LUMI-C calculations must not be relabelled or
retrospectively linked to a WorkChain to satisfy the new gate.
