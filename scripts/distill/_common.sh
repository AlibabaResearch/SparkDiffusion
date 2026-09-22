#!/usr/bin/env bash
# Shared preflight helpers for public training launchers.

require_path() {
    local name="$1"
    local path="${2:-}"
    if [[ -z "${path}" || ! -e "${path}" ]]; then
        echo "ERROR: required path does not exist: ${name}=${path:-<empty>}" >&2
        exit 1
    fi
}

require_file() {
    local name="$1"
    local path="${2:-}"
    if [[ -z "${path}" || ! -f "${path}" ]]; then
        echo "ERROR: required file does not exist: ${name}=${path:-<empty>}" >&2
        exit 1
    fi
}

require_dir() {
    local name="$1"
    local path="${2:-}"
    if [[ -z "${path}" || ! -d "${path}" ]]; then
        echo "ERROR: required directory does not exist: ${name}=${path:-<empty>}" >&2
        exit 1
    fi
}
