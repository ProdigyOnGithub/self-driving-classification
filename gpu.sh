#!/usr/bin/env bash
# Laptop <-> GPU server loop. The server has no internet, so everything (data + models) goes from here.
#
#   ./gpu.sh check                      # can we reach the server? which python/torch/GPU does it have?
#   ./gpu.sh push                       # code + photos + BDD data + hf_cache (rsync: only sends what changed)
#   ./gpu.sh train --name C_lora384     # starts training in the background on the server (survives ssh drop)
#   ./gpu.sh log                        # last lines of the newest training log
#   ./gpu.sh pull                       # bring runs/ and logs back here
#   ./gpu.sh cleanup                    # END OF PROJECT: delete everything we put on the server + our ssh key
set -euo pipefail
cd "$(dirname "$0")"

HOST="${GPU_HOST:-user@your-gpu-server}"   # export GPU_HOST=user@server
DIR="${GPU_DIR:-vlm}"          # folder in your home on the server
PY="${GPU_PY:-\$HOME/$DIR/venv/bin/python}"   # the venv you created on the server

case "${1:-}" in
  check)
    ssh "$HOST" "$PY -c 'import torch, transformers; print(\"torch\", torch.__version__, \"| transformers\", transformers.__version__, \"| cuda\", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"\")'; nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader; df -h ~ | tail -1"
    ;;
  push)
    ssh "$HOST" "mkdir -p $DIR"
    rsync -avz --progress --exclude '*.HEIC' --exclude '*.heic' --exclude '.DS_Store' \
      vlm.py train.py infer.py data data_bddoia hf_cache "$HOST:$DIR/"
    ;;
  train)
    shift
    name=""; prev=""
    for a in "$@"; do [ "$prev" = "--name" ] && name="$a"; prev="$a"; done
    if ssh "$HOST" "pgrep -u \$(id -u) -f '[v]lm/venv/bin/python'" >/dev/null; then
      echo "A training run is already going on the server - wait for it to finish (./gpu.sh log). Not starting another."; exit 1
    fi
    if [ -n "$name" ] && ssh "$HOST" "[ -e $DIR/runs/$name ]"; then
      echo "runs/$name already exists on the server - use a different --name. Not starting."; exit 1
    fi
    # only the python process goes to the background (with all its I/O redirected), so ssh returns at once
    ssh "$HOST" "cd $DIR && { HF_HOME=\$PWD/hf_cache HF_HUB_OFFLINE=1 \
      nohup $PY -u train.py $* > train_\$(date +%m%d_%H%M%S).log 2>&1 < /dev/null & }"
    echo "started on $HOST - check with: ./gpu.sh log"
    ;;
  log)
    ssh "$HOST" "cd $DIR && tail -n 40 \$(ls -t train_*.log | head -1)"
    ;;
  pull)
    mkdir -p runs logs
    rsync -avz --progress "$HOST:$DIR/runs/" runs/
    rsync -avz "$HOST:$DIR/train_*.log" logs/ || true
    ;;
  cleanup)
    # Safety: only ever deletes ~/$DIR and the pip download cache - refuse odd folder names.
    case "$DIR" in ""|/*|*..*|.) echo "refusing: DIR='$DIR' looks unsafe"; exit 1;; esac

    echo "== Is training still running on the server?"
    if ssh "$HOST" "pgrep -u \$(id -u) -f '[v]lm/venv/bin/python'" >/dev/null; then
      echo "   YES - a run is still going. Wait for it to finish (./gpu.sh log), then try again."; exit 1
    fi
    echo "   no"

    echo "== Results that exist on the server but NOT on this laptop:"
    missing=$(comm -23 <(ssh "$HOST" "cd $DIR/runs 2>/dev/null && ls -1" | sort) <(ls -1 runs 2>/dev/null | sort) || true)
    if [ -n "$missing" ]; then
      echo "$missing" | sed 's/^/   /'
      echo "   Run ./gpu.sh pull first (or answer 'delete' anyway to lose them)."
    else
      echo "   none - everything is pulled"
    fi

    echo "== Will be permanently deleted on $HOST:"
    ssh "$HOST" "du -sh ~/$DIR ~/.cache/pip 2>/dev/null | sed 's/^/   /'"
    echo "   + this laptop's ssh key from the server's ~/.ssh/authorized_keys (you'll need the password again after)"
    echo "   (shell history and system logs are not touched - they belong to the shared account)"
    read -r -p "Type 'delete' to continue: " ans
    [ "$ans" = "delete" ] || { echo "cancelled, nothing deleted"; exit 1; }

    ssh "$HOST" "rm -rf ~/$DIR ~/.cache/pip"

    # remove our key(s) last - after this, ssh asks for the password again
    keys=$( { ssh-add -L 2>/dev/null; cat ~/.ssh/*.pub 2>/dev/null; } | awk 'NF>=2 {print $2}' | sort -u)
    expr=""
    for k in $keys; do expr="$expr -e '\\#$k#d'"; done
    [ -n "$expr" ] && ssh "$HOST" "[ -f ~/.ssh/authorized_keys ] && sed -i $expr ~/.ssh/authorized_keys" || true

    echo "== Done. Verify (asks for the password now):"
    echo "   ssh $HOST 'ls -la ~; ls ~/.cache'"
    ;;
  *)
    sed -n '2,10p' "$0"; exit 1
    ;;
esac
