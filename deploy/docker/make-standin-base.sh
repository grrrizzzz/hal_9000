#!/usr/bin/env bash
# Build a stand-in for python:3.12-slim from this host's own files, for testing
# Dockerfile.kb where no container registry is reachable (the build agents'
# workspace). It is NOT used on the Pi: there the real python:3.12-slim and
# node:22-slim images are pulled from Docker Hub.
#
#   deploy/docker/make-standin-base.sh [TAG]      # default tag: hal-kb-standin:py312
#
# Needs: an x86_64 Linux host with /usr/bin/python3.12 (Ubuntu 24.04 has it),
# ldd, tar, docker. The result is a minimal root filesystem (a few tens of MB):
# python3.12 with its standard library, pip (from the host's bundled wheel),
# a POSIX shell and the coreutils the Dockerfile uses, useradd/groupadd, and the
# host's CA bundle. Like python:3.12-slim it has python, python3 and pip on PATH
# under /usr/local/bin, no curl, no system tz database (the tzdata wheel covers
# zoneinfo) and no EXTERNALLY-MANAGED marker.
set -euo pipefail

TAG="${1:-hal-kb-standin:py312}"
PY=/usr/bin/python3.12
[ -x "$PY" ] || { echo "needs $PY on the host" >&2; exit 1; }

ROOT="$(mktemp -d)"
trap 'rm -rf "$ROOT"' EXIT

# usrmerge layout, as on Debian 12+/Ubuntu 24.04
mkdir -p "$ROOT"/usr/{bin,sbin,lib,lib64,local/bin,local/lib/python3.12/dist-packages} \
         "$ROOT"/{etc,tmp,root,home,var/tmp,proc,sys,dev,run}
ln -s usr/bin "$ROOT/bin"; ln -s usr/sbin "$ROOT/sbin"
ln -s usr/lib "$ROOT/lib"; ln -s usr/lib64 "$ROOT/lib64"
chmod 1777 "$ROOT/tmp" "$ROOT/var/tmp"

copy() {  # copy a file (following symlinks) to the same path under $ROOT
  local src="$1"
  mkdir -p "$ROOT$(dirname "$src")"
  cp -L "$src" "$ROOT$src"
}

BINS=(
  "$PY" /usr/bin/dash /usr/bin/bash /usr/bin/env /usr/bin/id /usr/bin/sleep
  /usr/bin/true /usr/bin/false /usr/bin/test /usr/bin/\[ /usr/bin/mkdir /usr/bin/rm
  /usr/bin/rmdir /usr/bin/cp /usr/bin/mv /usr/bin/ln /usr/bin/cat /usr/bin/ls /usr/bin/chown
  /usr/bin/chmod /usr/bin/touch /usr/bin/du /usr/bin/find /usr/bin/grep /usr/bin/sed
  /usr/bin/head /usr/bin/tail /usr/bin/uname /usr/bin/dirname /usr/bin/basename
  /usr/bin/readlink /usr/bin/stat /usr/bin/wc /usr/bin/sort /usr/bin/tr /usr/bin/xargs
  /usr/bin/tar /usr/bin/gzip /usr/sbin/useradd /usr/sbin/groupadd
)
for b in "${BINS[@]}"; do copy "$b"; done
ln -sf dash "$ROOT/usr/bin/sh"

# Every shared library the binaries and Python's extension modules need, plus
# the two manylinux wheels may rely on (libstdc++, libgcc_s).
{
  for b in "${BINS[@]}"; do ldd "$b" || true; done
  find /usr/lib/python3.12/lib-dynload -name '*.so' -exec ldd {} \; || true
  echo "x => /lib/x86_64-linux-gnu/libstdc++.so.6 (0)"
  echo "x => /lib/x86_64-linux-gnu/libgcc_s.so.1 (0)"
} 2>/dev/null | awk '/=>/ && $3 ~ /^\// {print $3} /^\t\/lib64/ {print $1}' | sort -u |
  while read -r lib; do copy "$lib"; done
copy /lib64/ld-linux-x86-64.so.2
# the rest of glibc, as a Debian base has it (uv needs librt, for example)
for lib in /lib/x86_64-linux-gnu/{librt,libpthread,libdl,libutil,libresolv,libanl,libmvec,libnss_files,libnss_dns}.so.[0-9]; do
  copy "$lib"
done

# The standard library, without its tests or the PEP 668 marker
mkdir -p "$ROOT/usr/lib/python3.12"
tar -C /usr/lib/python3.12 --exclude=./test --exclude='./*/tests' --exclude=./idlelib \
    --exclude=./tkinter --exclude=./EXTERNALLY-MANAGED -cf - . | tar -C "$ROOT/usr/lib/python3.12" -xf -

# pip, from the wheel Ubuntu bundles for ensurepip
WHEEL="$(ls /usr/share/python-wheels/pip-*.whl | head -1)"
"$PY" -c "import sys, zipfile; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" \
  "$WHEEL" "$ROOT/usr/local/lib/python3.12/dist-packages"
for n in python python3 python3.12; do ln -s /usr/bin/python3.12 "$ROOT/usr/local/bin/$n"; done
for n in pip pip3; do
  printf '#!/bin/sh\nexec /usr/bin/python3.12 -m pip "$@"\n' > "$ROOT/usr/local/bin/$n"
  chmod 755 "$ROOT/usr/local/bin/$n"
done

# /etc: users, the loader cache, NSS and the CA bundle
printf 'root:x:0:0:root:/root:/bin/sh\nnobody:x:65534:65534:nobody:/nonexistent:/usr/sbin/nologin\n' > "$ROOT/etc/passwd"
printf 'root:x:0:\nnogroup:x:65534:\n' > "$ROOT/etc/group"
printf 'root:*:19000:0:99999:7:::\nnobody:*:19000:0:99999:7:::\n' > "$ROOT/etc/shadow"
printf 'root:*::\nnogroup:*::\n' > "$ROOT/etc/gshadow"
chmod 640 "$ROOT/etc/shadow" "$ROOT/etc/gshadow"
cp /etc/login.defs /etc/nsswitch.conf /etc/ld.so.cache /etc/ld.so.conf "$ROOT/etc/"
cp -r /etc/ld.so.conf.d "$ROOT/etc/"
mkdir -p "$ROOT/etc/ssl/certs" "$ROOT/etc/default"
# The host's trust store (here: SSL_CERT_FILE, which includes the workspace's
# TLS-inspecting proxy CA), so pip and uv can reach PyPI during a test build.
cp -L "${SSL_CERT_FILE:-/etc/ssl/certs/ca-certificates.crt}" "$ROOT/etc/ssl/certs/ca-certificates.crt"
mkdir -p "$ROOT/usr/lib/ssl"
ln -s /etc/ssl/certs "$ROOT/usr/lib/ssl/certs"
ln -s /etc/ssl/certs/ca-certificates.crt "$ROOT/usr/lib/ssl/cert.pem"
[ -f /etc/default/useradd ] && cp /etc/default/useradd "$ROOT/etc/default/"
printf 'PRETTY_NAME="hal-kb stand-in for python:3.12-slim (host files)"\nID=standin\n' > "$ROOT/etc/os-release"

tar -C "$ROOT" --numeric-owner -cf - . | docker import \
  -c 'ENV PATH=/usr/local/bin:/usr/local/sbin:/usr/sbin:/usr/bin:/sbin:/bin' \
  -c 'ENV LANG=C.UTF-8' \
  -c 'ENV SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt PIP_CERT=/etc/ssl/certs/ca-certificates.crt' \
  -c 'CMD ["python3"]' - "$TAG" >/dev/null
docker run --rm --network none "$TAG" python -c \
  "import sqlite3, ssl, sys; sqlite3.connect(':memory:').enable_load_extension(True); print(sys.version.split()[0], 'ok')"
docker image ls "$TAG" --format '{{.Repository}}:{{.Tag}} {{.Size}}'

# The Node stand-in for the web stage (node:22-slim), when the host has Node 22:
# the Python stand-in plus the host's Node install.
NODE_BIN="$(command -v node || true)"
if [ -n "$NODE_BIN" ] && "$NODE_BIN" --version | grep -q '^v22\.'; then
  NODE_HOME="$(dirname "$(dirname "$(readlink -f "$NODE_BIN")")")"
  printf 'FROM %s\nCOPY . /usr/local/\nENV NODE_EXTRA_CA_CERTS=/etc/ssl/certs/ca-certificates.crt\nCMD ["node"]\n' "$TAG" |
    docker build -q -t "${TAG%%:*}:node22" -f - "$NODE_HOME" >/dev/null
  docker run --rm --network none "${TAG%%:*}:node22" sh -c 'echo "node $(node --version), npm $(npm --version)"'
fi
