#!/bin/sh
# Writes an .env with new database and Qdrant secrets, from .env.example or an existing .env.
#
#   scripts/new-env.sh .env.example .env    # on the server, then fill in the API key and shop
#   scripts/new-env.sh .env vps.env         # from your laptop's .env, keys and settings copied
#
# Lines only meant for local dev are dropped.
set -eu

src=${1:?usage: new-env.sh <source .env> <new .env>}
out=${2:?usage: new-env.sh <source .env> <new .env>}
[ -e "$out" ] && { echo "$out exists, not overwriting" >&2; exit 1; }

pg_password=$(openssl rand -hex 24)
qdrant_key=$(openssl rand -hex 24)

umask 077
awk -v pg="$pg_password" -v qd="$qdrant_key" '
    /^(VISITOR_COOKIE_SECURE|LANGFUSE_[A-Z_]+)=/ { next }
    /^POSTGRES_PASSWORD=/ { print "POSTGRES_PASSWORD=" pg; next }
    /^QDRANT_API_KEY=/ { print "QDRANT_API_KEY=" qd; next }
    { print }
' "$src" > "$out"

echo "wrote $out with new POSTGRES_PASSWORD and QDRANT_API_KEY"
