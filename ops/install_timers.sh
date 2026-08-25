#!/bin/bash
# Install the monitoring schedule as real systemd timers.
#
#     sudo ops/install_timers.sh /path/to/ml1-governed-fraud
#
# `src/schedule.py` decides WHAT is due and WHEN it may retrain. That logic is
# tested precisely because it takes `now` as a parameter instead of reading the
# clock -- but something still has to call it, and until now nothing did.
#
# WHY A SYSTEMD TIMER RATHER THAN CRON, and it is not taste:
#
#   PERSISTENT=true       a timer that was missed while the box was down fires
#                         ONCE on boot rather than not at all. cron simply skips
#                         it. This is the correct behaviour here only because
#                         `schedule.due()` coalesces missed runs -- a scheduler
#                         that fired 72 catch-up runs would page the on-call for
#                         drift that has already resolved.
#
#   RANDOMIZEDDELAYSEC    every monitoring job on every host firing at exactly
#                         :00 is a thundering herd against whatever they read.
#
#   THE UNIT OWNS THE LOG  `journalctl -u ml1-monitor` is the run history.
#                          A cron job's history is an email nobody reads.
#
#   IT CAN BE INSPECTED    `systemctl list-timers` says when it last ran and
#                          when it runs next. That is the property the README
#                          argued a scheduler embedded in the application does
#                          not have, and the argument still holds -- this stays
#                          OUTSIDE the application.
#
# The cadence here is the FLOOR, not the schedule. The timer wakes the job
# hourly; `schedule.due()` then decides which checks are actually due, so the
# 6-hourly and weekly signals are governed by the code rather than by four
# separate timer files that could disagree with it.
set -euo pipefail

REPO="${1:-}"
if [ -z "$REPO" ] || [ ! -f "$REPO/monitor.py" ]; then
  echo "usage: $0 /path/to/ml1-governed-fraud" >&2
  exit 2
fi
# The interpreter that has the project's dependencies, NOT whichever python3 is
# first on the unit's PATH.
#
# This was found by the unit failing, and the failure is worth keeping: pointed
# at WSL's bare python3 the pass exited 1 with "MONITORING PASS FAILED ... This
# is not drift", which is exactly the distinction run_schedule_tick.py's exit
# codes exist to draw. A scheduler that had swallowed the ImportError and
# written an empty report would have shown a healthy dashboard instead.
#
# On WSL against a repo on the Windows filesystem, that interpreter is the
# Windows one -- systemd can execute it directly.
PY="${PYTHON:-python3}"
if ! "$PY" -c 'import numpy, pandas, sklearn' >/dev/null 2>&1; then
  for cand in /mnt/c/Python314/python.exe /mnt/c/Python313/python.exe /mnt/c/Python312/python.exe; do
    if [ -x "$cand" ] && "$cand" -c 'import numpy, pandas, sklearn' >/dev/null 2>&1; then
      PY="$cand"; break
    fi
  done
fi
if ! "$PY" -c 'import numpy, pandas, sklearn' >/dev/null 2>&1; then
  echo "no interpreter with numpy/pandas/sklearn found; set PYTHON=..." >&2
  echo "Refusing to install a timer that would fail on every fire." >&2
  exit 3
fi
echo "using interpreter: $PY"

# A WINDOWS INTERPRETER NEEDS A WINDOWS PATH. Handed `/mnt/c/...` it resolves
# that as `C:\mnt\c\...` and reports the script missing -- which the unit then
# reports as exit 2, correctly, and which cost a second round to find. systemd
# runs the exe fine; it is the ARGUMENT that has to cross the boundary.
SCRIPT="$REPO/run_schedule_tick.py"
WORKDIR="$REPO"
case "$PY" in
  *.exe)
    # Forward slashes, not backslashes. systemd treats `\` in ExecStart as an
    # ESCAPE -- a Windows path goes in and `` comes out as a carriage return,
    # `` as a formfeed, and the interpreter reports an invalid argument.
    # Windows Python accepts `C:/...` perfectly, so the whole class of problem
    # is avoided rather than escaped around.
    SCRIPT="$(wslpath -w "$REPO/run_schedule_tick.py" | tr '\' '/')"
    ;;
esac

cat > /etc/systemd/system/ml1-monitor.service <<EOF
[Unit]
Description=ML-1 fraud model monitoring pass
Documentation=file://$REPO/docs/SCHEDULE.md
# Do not start a second pass while one is running. The schedule state file is
# read-modify-written, and two passes racing on it would double-count breaches.
StartLimitIntervalSec=0

[Service]
Type=oneshot
WorkingDirectory=$REPO
ExecStart=$PY $SCRIPT
# The job is expected to exit non-zero when a check PAGES. That is a signal,
# not a crash, so the unit must not be marked failed for it.
SuccessExitStatus=0 10
EOF

cat > /etc/systemd/system/ml1-monitor.timer <<'EOF'
[Unit]
Description=Wake the ML-1 monitoring schedule hourly

[Timer]
OnCalendar=hourly
# Fire once after a missed window rather than not at all. Safe ONLY because
# schedule.due() coalesces -- see the header.
Persistent=true
# Do not join the herd at :00.
RandomizedDelaySec=300
AccuracySec=1min
Unit=ml1-monitor.service

[Install]
WantedBy=timers.target
EOF

systemctl daemon-reload
systemctl enable --now ml1-monitor.timer
echo "installed. next run:"
systemctl list-timers ml1-monitor.timer --no-pager
