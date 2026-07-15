# Dependency Security

## Known Unfixed Vulnerabilities

As of 2026-07-15, the following packages have known vulnerabilities with no available fixes:

### ragas 0.4.3
- **CVE-2026-6587**: SSRF via Multi-Modal Faithfulness Collections Module (GHSA-95ww-475f-pr4f, PYSEC-2026-3046)
- **Status**: No fix versions available
- **Usage**: Evaluation only (`[dependency-groups] eval`)
- **Impact**: Local development only, not in production or CI

### diskcache 5.6.3
- **CVE-2025-69872**: Unsafe pickle deserialization (GHSA-w8v5-vhqr-4h9v, PYSEC-2026-2447)
- **Status**: No fix versions available; 5.6.3 is the latest version
- **Usage**: Transitive dependency (pulled in via evaluation stack)
- **Impact**: Local development only, not in production or CI

### langchain-community
- **Status**: Marked as archived in audit tooling
- **Usage**: Transitive dependency via ragas
- **Impact**: Local development only, not in production or CI

## Mitigation

**CI Exclusion**: The audit job in `.github/workflows/lint.yml` runs `uv audit --exclude-groups eval`, so these vulnerabilities do not block CI. Only production (`dependencies`) and dev (`[dependency-groups] dev`) are audited.

**Local Development**: Developers can still install and use ragas locally for evaluation work:
```bash
uv sync --all-groups  # Includes eval group
```

## Review Schedule

Monitor the following resources quarterly:
- [CVE-2026-6587](https://nvd.nist.gov/vuln/detail/CVE-2026-6587)
- [CVE-2025-69872](https://nvd.nist.gov/vuln/detail/CVE-2025-69872)
- [ragas releases](https://pypi.org/project/ragas/) (currently at 0.4.3)
- [diskcache releases](https://pypi.org/project/diskcache/) (currently at 5.6.3)

Once patches are available, upgrade and re-run `uv audit` to confirm.

## Alternative: Remove ragas

If evaluation work is not actively used, `ragas` can be removed entirely from `pyproject.toml` to eliminate the vulnerability surface. This would also remove the `diskcache` transitive dependency.
