#!/bin/sh
# The cofactor chunk controller, driven over synthetic device responses with
# NO GPU. It compiles the SHIPPING controller -- gen_chunksim.py lifts it
# verbatim out of cofq_flush, so this cannot drift from the code it reports on.
#
# WHY THIS EXISTS ALONGSIDE chunkcheck.sh. That gate needs an NVIDIA card and a
# 200-q band, and can only exercise whichever trajectory that particular card
# happens to produce -- which is how the valve's defects survived: the branches
# that matter do not fire on healthy hardware, and the cluster is not always
# there when you need it. This drives every device model on purpose, reaches
# all three valve branches, and runs in under a second on any machine.
set -e
cd "$(dirname "$0")"
fail=0

python3 gen_chunksim.py > /dev/null
echo "== shipped configuration =="
c++ -std=c++17 -O1 -o /tmp/chunksim.$$ chunksim.cpp 2>/dev/null
/tmp/chunksim.$$ || fail=1

echo
echo "== hard floor above the grid (reaches the floor park) =="
c++ -std=c++17 -O1 -DCOF_CHUNK_HARD_FLOOR=200000u -o /tmp/chunksim_fp.$$ chunksim.cpp 2>/dev/null
/tmp/chunksim_fp.$$ || fail=1

# THE CONTROL IS A REAL REVISION, not a #define. The pre-fix controller at
# 556dc15 restored a larger chunk on its no-progress park; the invariants below
# say the chunk never rises once the valve has spoken, so it must FAIL here.
# A control that is merely the same code with a flag flipped tests the flag.
echo
echo "== control: the pre-fix controller at 556dc15 (must FAIL) =="
python3 gen_chunksim.py 556dc15 /tmp/chunksim_ctrl.$$.cpp > /dev/null
c++ -std=c++17 -O1 -o /tmp/chunksim_ctrl.$$ /tmp/chunksim_ctrl.$$.cpp 2>/dev/null
if /tmp/chunksim_ctrl.$$ > /tmp/chunksim_ctrl.$$.out 2>&1; then
    echo "  CONTROL PASSED -- the invariants do not discriminate, so the gate"
    echo "  above is vacuous. Fix the invariants, not the control."
    fail=1
else
    echo "  control fails as it must: $(grep -c '^FAIL' /tmp/chunksim_ctrl.$$.out) violation(s)"
    grep '^FAIL' /tmp/chunksim_ctrl.$$.out | head -2 | sed 's/^/    /'
fi

rm -f /tmp/chunksim.$$ /tmp/chunksim_fp.$$ /tmp/chunksim_ctrl.$$ \
      /tmp/chunksim_ctrl.$$.cpp /tmp/chunksim_ctrl.$$.out
echo
[ $fail = 0 ] && echo "CHUNK CONTROLLER GATE: PASS" || { echo "CHUNK CONTROLLER GATE: FAIL"; exit 1; }
