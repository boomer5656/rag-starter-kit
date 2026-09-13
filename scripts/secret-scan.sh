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

# Pass 4 - street-address shapes. Credentials are not the only thing that must not reach
# git. Passes 1-3 detect credentials only and report clean on a tree that is full of real
# property addresses, which is exactly what a portfolio repo carries.
# Shape-matching ONLY. The real addresses are deliberately not listed here: writing them
# down to match them exactly would put them into git in every repo that carries this gate.
# The core accepts 1-3 words so a two-word street name is seen, and an ordinal word so a
# numbered street is seen. Both were blind spots in the first cut of this pass, and each
# hid a real property outright rather than merely mis-scoring it.
# Write the groups as ( ), never (?: ). This is POSIX ERE, where (?: is not a
# non-capturing group - it is a syntax error that makes the whole pattern match NOTHING,
# so the pass would report clean forever. That exact suggestion was proposed and rejected.
# Keep each street type and its spelled-out form in step. Blvd/Ct/Pl first shipped without
# Boulevard/Court/Place, so "12 Melrose Boulevard" matched nothing and passed silently.
# -o matters for the same reason it does in pass 3 - each occurrence is judged on its own,
# so a placeholder elsewhere on the line cannot excuse a real address next to it.
# Allowlisted: a core containing Main / Oak / Elm, the textbook fake street names, which is
# the placeholder vocabulary these repos already use. It matches the word anywhere in the
# core, not just next to the street type, so a two-word placeholder still passes.
#   TRADE-OFF, accepted knowingly: a REAL address on a street whose name contains Main, Oak
#   or Elm is missed. Narrow street names still beat allowlisting whole test-file paths - a
#   path allowlist goes stale silently when a file is renamed or a real address lands in an
#   existing test file, and it would hide far more than three street names do.
#   Do NOT grow this list by scraping the current tree. Deriving the allowlist from what is
#   already committed makes the gate self-certifying: it would allowlist a real address that
#   has already leaked. Any addition must come from a declared placeholder vocabulary.
hits=$(scan -o '[0-9]{2,5}(-[0-9]{1,5})? (([A-Z][A-Za-z]+|[0-9]{1,2}(st|nd|rd|th)) ){1,3}(St|Ave|Rd|Dr|Ln|Blvd|Ct|Pl|Street|Avenue|Road|Drive|Lane|Boulevard|Court|Place)\b' \
  | grep -vE ' (Main|Oak|Elm) |/secret-scan\.sh')
[ -n "$hits" ] && { echo "FAIL pass4:"; echo "$hits"; fail=1; }

# Pass 5 - employer identification numbers. Pass 4 covers addresses; this covers the other
# PII class that turned up in a real tree.
# LABEL PROXIMITY, not bare shape. [0-9]{2}-[0-9]{7} on its own matched 50 lines in the one
# repo that handles EINs, nearly all of them prose ABOUT EIN handling. A red signal that is
# all noise teaches everyone to ignore the gate, so the number must sit near a label.
# The label is matched CASE-INSENSITIVELY and the 40-char window allows digits. The strict
# spelling was tried first and REJECTED on evidence: an uppercase-only label with a
# digit-free gap cut 50 hits to 15 but caught only ONE of the two real EINs that repo
# actually carried. Both live values sat in ein(.<entity> LLC., .<number>.) fixtures, where
# the label is lowercase and the entity name in the gap contains a house number, which broke
# both halves of the strict rule at once. The loose form catches 2 of 2. A pass that misses
# half the live values is worse than no pass, so the wider window is the correct trade.
# NO WORD BOUNDARY, and the spelled-out label is required. Both were real misses:
#   - a trailing \b cannot match between a digit and a letter, so an EIN pressed straight
#     against text - EIN<number>SoloHouseLLCFormation - was invisible. The trailing guard is
#     ([^0-9]|$) instead, which still refuses an 8th digit so a longer digit run and a UUID
#     fragment are both excluded, but accepts a letter jammed against the number.
#   - .Employer ID Number<number>. carries no e-i-n substring at all, so the label never fired
#     on the spelled-out form. Employer is now its own label term.
#   In a repo that runs OCR over tax documents, the jammed form is exactly where a real EIN
#   is most likely to sit, so neither miss was academic - both hid live OCR fixtures.
# Allowlisted: 00-0000000 and 12-3456789. All-zeros and the sequential documentation example
# are self-evidently not values - the same class of textbook placeholder as Main/Oak/Elm in
# pass 4. Masked forms need no term at all: ##-####### and xx-xxxxxxx contain no digits, so
# the pattern never sees them.
#   Any FURTHER allowlisting must come from a declaration committed in the repo, never from
#   scraping the tree. See the pass 4 note for why that rule exists - it is the same failure,
#   and here it was nearly realised: the two known-real EINs sat in the same fixture file as
#   the invented ones, so a scraped list would have blessed them permanently.
hits=$(scan -o '([Ee][Ii][Nn]|FEIN|TIN|[Ee]mployer|[Tt]ax[ _-]?[Ii][Dd]).{0,40}[0-9]{2}-[0-9]{7}([^0-9]|$)' | grep -vE '00-0000000|12-3456789|/secret-scan\.sh')
[ -n "$hits" ] && { echo "FAIL pass5:"; echo "$hits"; fail=1; }

# Pass 6 - phone numbers inside payment-processor merchant descriptors. Pass 4 covers
# addresses, pass 5 covers EINs; this covers the third PII class found in a real tree - a
# Plaid transaction descriptor that carries a vendor or person NAME next to their PHONE.
# The name is what makes it personal data, but the name is not machine-detectable. The
# descriptor structure is: a processor prefix, then a name, then the number. So the pass
# anchors on the descriptor, not on the phone.
# DESCRIPTOR-ANCHORED, NOT a bare phone shape, and the difference is not cosmetic. A bare
# phone-shaped scan finds 23 hits in the repo that holds transaction data, and 20 of them
# are ordinary vendor phone fields in a declared-fictional demo fixture that were never a
# leak. The anchored form finds exactly the 3 that are real. A pass whose signal is 87%
# noise gets ignored, and then the 3 get ignored with it.
# The name segment is [^0-9]{2,40} rather than a spelled-out character class. Two reasons:
# it accepts an apostrophe in a vendor name (O.BRIEN & SONS) which an explicit class would
# have to list, and listing it is exactly the mistake this file warns about - a literal
# apostrophe inside a single-quoted bash filter closes the string and the quotes vanish from
# the regex. Negating digits sidesteps the whole problem and is shorter.
# Four number forms, since processors are not consistent: 3-7, 3-3-4, (3) 3-4, and a bare
# 10-digit run. The bare run was measured to cost nothing - zero new hits across every
# gated repo - so there is no reason to leave that variant undetected.
# NO ALLOWLIST, deliberately. Every current hit is real unremediated PII and must fail. If a
# fixture ever needs to carry a descriptor-shaped string, declare it in the repo first and
# the term gets read off that declaration, as passes 4 and 5 do.
# SCOPE: this is the merchant-descriptor pass, NOT a general phone-number pass. A pass
# keying on phone next to a contact label would light up 20 declared-fictional demo values
# and needs a declaration before it is worth anything. Deliberately not built.
# ONE allowlist term, and it is provable rather than merely likely: the NANP range reserved for
# fiction, 555-0100 to 555-0199. No real line can be issued in it, so exempting it cannot ever
# launder a live number - which is a stronger guarantee than a fake-sounding street name, where
# a real address could in principle sit on a Main Street. The term matches the reserved NUMBER,
# never a vendor name, so a real phone beside a synthetic name still fails.
hits=$(scan -o '[A-Z]{2,4} ?\*[^0-9]{2,40}([0-9]{3}-[0-9]{7}|[0-9]{3}-[0-9]{3}-[0-9]{4}|\([0-9]{3}\) ?[0-9]{3}-[0-9]{4}|[0-9]{10})' | grep -vE '555-?01[0-9]{2}|/secret-scan\.sh')
[ -n "$hits" ] && { echo "FAIL pass6:"; echo "$hits"; fail=1; }

[ $fail -eq 0 ] && echo "secret-scan: clean"
exit $fail
