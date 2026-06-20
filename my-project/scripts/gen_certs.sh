#!/usr/bin/env bash
# Generate self-signed TLS certificates for a Flower SuperLink <-> SuperNode
# deployment. For development / internal use; in production use a real CA.
#
# Produces, under certs/ (gitignored):
#   ca.crt / ca.key        - certificate authority
#   server.pem / server.key- SuperLink server cert (SAN must match the dialed host)
#
# Usage:
#   bash scripts/gen_certs.sh [SUPERLINK_HOST]
# SUPERLINK_HOST defaults to "localhost"; pass the DNS name or IP clients dial.
set -euo pipefail

HOST="${1:-localhost}"
CERT_DIR="${CERT_DIR:-certs}"
DAYS="${DAYS:-365}"

mkdir -p "$CERT_DIR"
cd "$CERT_DIR"

echo "[certs] Generating CA..."
openssl genrsa -out ca.key 4096
openssl req -x509 -new -nodes -key ca.key -sha256 -days "$DAYS" \
  -subj "/O=FL_AV/CN=FL_AV-CA" -out ca.crt

echo "[certs] Generating SuperLink server cert for host: $HOST (SAN must match dialed address)"
openssl genrsa -out server.key 4096
openssl req -new -key server.key -subj "/O=FL_AV/CN=$HOST" -out server.csr

cat > server.ext <<EOF
subjectAltName = DNS:$HOST, DNS:localhost, IP:127.0.0.1
EOF

openssl x509 -req -in server.csr -CA ca.crt -CAkey ca.key -CAcreateserial \
  -days "$DAYS" -sha256 -extfile server.ext -out server.pem

rm -f server.csr server.ext ca.srl
echo "[certs] Done. Files in $CERT_DIR/: ca.crt ca.key server.pem server.key"
echo "[certs] SuperLink:  --ssl-ca-certfile $CERT_DIR/ca.crt --ssl-certfile $CERT_DIR/server.pem --ssl-keyfile $CERT_DIR/server.key"
echo "[certs] SuperNodes: --root-certificates $CERT_DIR/ca.crt   (and set the federation root-certificates in pyproject.toml)"
