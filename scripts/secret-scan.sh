#!/bin/bash
# Secret gate. Run from the repo root before every commit.
# Exits non-zero if anything that looks like a live credential is about to be committed.
#
# Enumerates via `git ls-files`, not a worktree walk: that is the question a pre-commit gate
# should ask (what would git actually commit), it skips gitignored .env files that hold real
# keys by design, and it skips node_modules — a recursive grep over those does not finish.
cd "$(dirname "$0")/.." || exit 2
fail=0
scan(){ git ls-files -z | xargs -0 grep -IEn "$@" 2>/dev/null; }

# Pass 1 - key=value shaped credentials.
# Allowlisted (each provably value-free):
#   REDACTED / example / CHANGEME / change-me / replace-me / paste / your-key / <angle> / **yours
#                                  -> placeholder, not a value
#   ${VAR} / $(cmd) / {env:VAR} / $BARE_VAR / process.env / os.environ
#                                  -> indirection, no literal value
#   value starting / or ./         -> a filesystem path
#   value starting ident.          -> dotted property access (j.access_token)
#   value starting ( { [ or `      -> an expression, object literal, index or code span
#   value that is a call ident(   -> a function call, never a literal credential
#   string/number/boolean/true/false -> a type annotation or a bool
#   "none"                         -> LiteLLM's placeholder for keyless local providers
# -i is REQUIRED: matching only the UPPERCASE spellings is what let a live lowercase
# YAML `secret_key:` sit in fleet-infra for eight days before gitleaks found it.
hits=$(scan -i '(PASSWORD|TOKEN|SECRET|_KEY|apikey|api_key)[[:space:]]*[:=][[:space:]]*\S{6,}' \
  | grep -viE 'REDACTED|example|CHANGEME|change[-_]me|replace[-_]?me|paste|your[-_]?(key|token|secret|password)|\*\*yours|\$\{|\$\(|\{env:|\$[A-Z_]{2,}|process\.env|environ|[:=][[:space:]]*\.?/|[:=][[:space:]]*"?<|[:=][[:space:]]*[A-Za-z_][A-Za-z0-9_]*[.(]|[:=][[:space:]]*[({[`]|[:=][[:space:]]*"?(string|number|boolean|true|false)|"none"|/secret-scan\.sh')
[ -n "$hits" ] && { echo "FAIL pass1:"; echo "$hits"; fail=1; }

# Pass 2 - provider key prefixes and bcrypt hashes.
# The ONLY exception: a prefix followed by 6+ literal x is a format placeholder, which is what
# .env.example exists to show. A real key still fails - the x-run is what is allowed.
hits=$(scan 'sk-or-|sk-ant-|ghp_|ork_proxy_|\$2[aby]\$' | grep -vE 'x{6,}|/secret-scan\.sh')
[ -n "$hits" ] && { echo "FAIL pass2:"; echo "$hits"; fail=1; }

# Pass 3 - curl basic-auth carrying a literal password.
# -o matters: each -u occurrence is judged on its own, because one line can hold both
# an indirect form and a literal one - fleet-ops/scripts/wazuh-rotate.sh did.
# Not a finding: a $VAR value, or an <angle-bracket> placeholder. A single-quoted
# '$VAR' never matches the pattern at all (the value class excludes the quote); the
# UNQUOTED $VAR form is what ':[$]' filters. Write it as the bracket [$] and not a
# backslash-escaped $ - the shell eats the backslash and ERE then reads a bare $ as
# the end-of-line anchor, matching nothing.
hits=$(scan -o -- "-u[[:space:]]+'?[A-Za-z0-9_-]+:[^'\"[:space:]]{4,}" \
  | grep -vE ':[$]|:<|/secret-scan\.sh')
[ -n "$hits" ] && { echo "FAIL pass3:"; echo "$hits"; fail=1; }

[ $fail -eq 0 ] && echo "secret-scan: clean"
exit $fail
