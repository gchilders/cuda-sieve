#!/bin/sh
# Source-level ncu profile of k_apply (finding 101's method) on the NFS@Home
# lpba-34 I16 job and, for comparison at the same build, c183 I15.
# Single-queued so launch order is fixed: per slab, side 1 then side 0.
# NFS job: 4 slabs -> 8 k_apply launches per q; q index 3 = launches 24..31.
# c183: 1 slab -> 2 per q; q index 3 = launches 6..7 (finding 101's choice).
cd /home/kylea/code/cuda-sieve/bench || exit 1
W=/home/kylea/code/cuda-sieve/work/applyprof
git describe --always --dirty > $W/build.txt; grep -o "\-DCF_LMAX=[0-9]* -DBN_LIMBS=[0-9]* -DPIPE_K=[0-9]*" ../work/slabpipe/r2.build.log | sort -u >> $W/build.txt
NFS="--poly $W/nfs.job --fb1 $W/nfs.roots1.m16 --sq-side 1 --logI 16 --J 32768 --qrange 296847000: --nq 5"
C183="--poly ../oracle/input.job --fb1 ../oracle/c183.fb1 --logI 15 --qrange 130000000: --nq 5"
ncu --set full --import-source yes --kernel-name k_apply --launch-skip 24 --launch-count 8 \
    -f -o $W/nfs_apply ./bench --pipeline --cofactor $NFS --no-q-pipeline > $W/nfs_apply.log 2>&1
echo "nfs ncu exit=$?"
ncu --set full --import-source yes --kernel-name k_apply --launch-skip 6 --launch-count 2 \
    -f -o $W/c183_apply ./bench --pipeline --cofactor $C183 --no-q-pipeline > $W/c183_apply.log 2>&1
echo "c183 ncu exit=$?"
echo done
