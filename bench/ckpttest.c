#define _XOPEN_SOURCE 700

#include "ckpt.h"

#include <stdio.h>
#include <string.h>
#include <unistd.h>

static int fail(const char *what)
{
    fprintf(stderr, "ckpttest: %s\n", what);
    return 1;
}

int main(void)
{
    char root[] = "/tmp/cuda-sieve-ckpttest.XXXXXX";
    char relations[CKPT_PATH_MAX], candidate[CKPT_PATH_MAX];
    char checkpoint[CKPT_PATH_MAX];
    poly_t P;
    bench_cfg_t cfg;
    ckpt_t written, loaded;
    int rc = 1;

    if (!mkdtemp(root)) { perror("mkdtemp"); return 1; }
    if (snprintf(relations, sizeof relations, "%s/relations with space", root)
            >= (int)sizeof relations ||
        snprintf(candidate, sizeof candidate, "%s/candidates with space.part",
                 root) >= (int)sizeof candidate ||
        ckpt_ckpt_path(relations, checkpoint, sizeof checkpoint))
        goto out;

    memset(&P, 0, sizeof P);
    memset(&cfg, 0, sizeof cfg);
    memset(&written, 0, sizeof written);
    P.deg = 0;
    strcpy(P.cs[0], "1");
    strcpy(P.y0s, "0");
    strcpy(P.y1s, "1");
    written.next_q = 101;
    written.next_rho = 7;
    written.rel_bytes = 1234;
    written.cand_bytes = 567;
    written.cand_dev = 42;
    written.cand_ino = 99;
    written.nrel = 12;
    written.nqdone = 3;
    written.scale = written.scale0 = 1.0;
    written.allowance = written.allowance0 = 2.0;
    strcpy(written.fp, "0123456789abcdef");
    strcpy(written.cand_part, candidate);

    if (ckpt_write(relations, &P, &cfg, &written))
        goto out;
    if (ckpt_read(relations, &loaded))
        goto out;
    if (strcmp(loaded.cand_part, candidate))
        goto bad_path;
    if (loaded.cand_dev != written.cand_dev ||
        loaded.cand_ino != written.cand_ino)
        goto bad_identity;
    if (loaded.cand_bytes != written.cand_bytes)
        goto bad_bytes;

    printf("ckpttest: candidate path and identity round-trip passed\n");
    rc = 0;
    goto out;

bad_path:
    rc = fail("candidate path did not round-trip");
    goto out;
bad_identity:
    rc = fail("candidate identity did not round-trip");
    goto out;
bad_bytes:
    rc = fail("candidate byte count did not round-trip");

out:
    remove(checkpoint);
    rmdir(root);
    return rc;
}
