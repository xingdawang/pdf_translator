#!/bin/zsh
set -e

SCRIPT_DIR=${0:A:h}
cd "$SCRIPT_DIR"

if [[ ! -x ".venv/bin/python" ]]; then
  echo "没有找到 .venv。请先按照 README 安装项目。"
  exit 1
fi

exec ".venv/bin/python" -m pdf_translator.hybrid_app --host 0.0.0.0 --port 5050
