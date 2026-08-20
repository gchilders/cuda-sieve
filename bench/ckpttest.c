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


/* The identity recorded in a checkpoint is the whole of what stands between a
 * stored pathname and an unlink, so it has to actually distinguish two files.
 * This is the regression test for the MSVC case specifically: _stat64 reports
 * st_ino as a constant 0, so the (dev, ino) pair this replaced compared equal
 * for every peer .part on the drive and the guard passed anything. Two
 * distinct files here MUST NOT compare equal, and a file MUST compare equal to
 * itself whether it was identified through a path or through an open stream. */
static int check_file_identity(const char *dir)
{
    char pa[CKPT_PATH_MAX], pb[CKPT_PATH_MAX];
    bench_file_id_t ia, ib, ia_stream;
    FILE *fa = NULL, *fb = NULL;
    int rc = 1;

    if (snprintf(pa, sizeof pa, "%s/id-a.part", dir) >= (int)sizeof pa ||
        snprintf(pb, sizeof pb, "%s/id-b.part", dir) >= (int)sizeof pb)
        return fail("identity temp paths too long");
    if (!(fa = fopen(pa, "wb")) || !(fb = fopen(pb, "wb"))) {
        rc = fail("cannot create identity test files");
        goto out;
    }
    /* Same length on purpose: size is not identity, and the guard this backs
     * also compares sizes. If identity were still collapsing to a constant,
     * equal sizes would let these two look like the same file. */
    if (fwrite("xxxx", 1, 4, fa) != 4 || fwrite("xxxx", 1, 4, fb) != 4 ||
        fflush(fa) || fflush(fb)) {
        rc = fail("cannot write identity test files");
        goto out;
    }
    if (bench_file_id_path(pa, &ia) || bench_file_id_path(pb, &ib) ||
        bench_file_id_stream(fa, &ia_stream)) {
        rc = fail("bench_file_id failed on a regular file");
        goto out;
    }
    if (!bench_file_id_valid(&ia) || !bench_file_id_valid(&ib)) {
        rc = fail("file identity unavailable on the test filesystem");
        goto out;
    }
    if (bench_file_id_equal(&ia, &ib)) {
        rc = fail("two distinct files reported the same identity");
        goto out;
    }
    if (!bench_file_id_equal(&ia, &ia_stream)) {
        rc = fail("path and stream identities disagree for one file");
        goto out;
    }
    {   /* An unavailable identity must never compare equal -- not even to
         * another unavailable one, which is the "cannot tell" case that has to
         * refuse rather than authorise a delete. */
        const bench_file_id_t none = { 0, 0 };
        if (bench_file_id_valid(&none) || bench_file_id_equal(&none, &none)) {
            rc = fail("an unavailable identity compared equal");
            goto out;
        }
    }
    rc = 0;
out:
    if (fa) fclose(fa);
    if (fb) fclose(fb);
    remove(pa); remove(pb);
    return rc;
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

    if (check_file_identity(root)) goto out;

    printf("ckpttest: candidate path and identity round-trip passed;"
           " file identity distinguishes peers\n");
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
