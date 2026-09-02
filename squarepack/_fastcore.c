/* Fast core for squarepack: penalty energy/gradient and an L-BFGS minimiser
 * with an internal cell-list pair rebuild.  Compiled on demand by fastcore.py.
 *
 * Layout of z: [x_0..x_{n-1}, y_0..y_{n-1}, theta_0..theta_{n-1}].
 */
#include <math.h>
#include <stdlib.h>
#include <string.h>

#define SQRT2 1.4142135623730951

typedef struct { int *I, *J; int m, cap; } PairList;

static void pl_push(PairList *pl, int i, int j) {
    if (pl->m == pl->cap) {
        pl->cap = pl->cap ? pl->cap * 2 : 1024;
        pl->I = (int *)realloc(pl->I, sizeof(int) * pl->cap);
        pl->J = (int *)realloc(pl->J, sizeof(int) * pl->cap);
    }
    if (i < j) { pl->I[pl->m] = i; pl->J[pl->m] = j; } else { pl->I[pl->m] = j; pl->J[pl->m] = i; }
    pl->m++;
}

/* all pairs with |dx| < cutoff and |dy| < cutoff, via a uniform grid */
static void build_pairs(int n, const double *x, const double *y, double cutoff, PairList *pl) {
    pl->m = 0;
    if (n < 2) return;
    double xmin = x[0], xmax = x[0], ymin = y[0], ymax = y[0];
    for (int i = 1; i < n; i++) {
        if (x[i] < xmin) xmin = x[i]; if (x[i] > xmax) xmax = x[i];
        if (y[i] < ymin) ymin = y[i]; if (y[i] > ymax) ymax = y[i];
    }
    long ncx = (long)((xmax - xmin) / cutoff) + 1, ncy = (long)((ymax - ymin) / cutoff) + 1;
    if (ncx * ncy > 8L * n + 64 || n < 64) {           /* degenerate spread or tiny n: brute force */
        for (int i = 0; i < n; i++)
            for (int j = i + 1; j < n; j++)
                if (fabs(x[i] - x[j]) < cutoff && fabs(y[i] - y[j]) < cutoff) pl_push(pl, i, j);
        return;
    }
    int *head = (int *)malloc(sizeof(int) * ncx * ncy);
    int *next = (int *)malloc(sizeof(int) * n);
    for (long c = 0; c < ncx * ncy; c++) head[c] = -1;
    for (int i = 0; i < n; i++) {
        long cx = (long)((x[i] - xmin) / cutoff), cy = (long)((y[i] - ymin) / cutoff);
        long c = cx * ncy + cy;
        next[i] = head[c]; head[c] = i;
    }
    const int ndx[4] = {1, 0, 1, 1}, ndy[4] = {0, 1, 1, -1};
    for (long cx = 0; cx < ncx; cx++) for (long cy = 0; cy < ncy; cy++) {
        long c = cx * ncy + cy;
        for (int i = head[c]; i != -1; i = next[i]) {
            for (int j = next[i]; j != -1; j = next[j])
                if (fabs(x[i] - x[j]) < cutoff && fabs(y[i] - y[j]) < cutoff) pl_push(pl, i, j);
            for (int k = 0; k < 4; k++) {
                long nx = cx + ndx[k], ny = cy + ndy[k];
                if (nx < 0 || nx >= ncx || ny < 0 || ny >= ncy) continue;
                for (int j = head[nx * ncy + ny]; j != -1; j = next[j])
                    if (fabs(x[i] - x[j]) < cutoff && fabs(y[i] - y[j]) < cutoff) pl_push(pl, i, j);
            }
        }
    }
    free(head); free(next);
}

static inline double sgnd(double v) { return v > 0 ? 1.0 : (v < 0 ? -1.0 : 0.0); }

/* E and gradient; grad may be NULL.  Returns E. */
double energy_grad_c(int n, double s, const double *z, int m, const int *I, const int *J, double *grad) {
    const double *x = z, *y = z + n, *t = z + 2 * n;
    double *gx = grad, *gy = grad ? grad + n : NULL, *gt = grad ? grad + 2 * n : NULL;
    if (grad) memset(grad, 0, sizeof(double) * 3 * n);
    double E = 0.0;
    for (int k = 0; k < m; k++) {
        int i = I[k], j = J[k];
        double dx = x[i] - x[j], dy = y[i] - y[j];
        double ci = cos(t[i]), si = sin(t[i]), cj = cos(t[j]), sj = sin(t[j]);
        double A[4] = {ci * dx + si * dy, -si * dx + ci * dy, cj * dx + sj * dy, -sj * dx + cj * dy};
        int kk = 0; double M = fabs(A[0]);
        for (int q = 1; q < 4; q++) if (fabs(A[q]) > M) { M = fabs(A[q]); kk = q; }
        double phi = t[j] - t[i];
        double cphi = cos(phi), sphi = sin(phi);
        double R = 0.5 * (fabs(cphi) + fabs(sphi));
        double p = 0.5 + R - M;
        if (p <= 0) continue;
        E += p * p;
        if (!grad) continue;
        double sg = sgnd(A[kk]);
        double ax, ay;
        switch (kk) { case 0: ax = ci; ay = si; break; case 1: ax = -si; ay = ci; break;
                      case 2: ax = cj; ay = sj; break; default: ax = -sj; ay = cj; }
        double g = 2.0 * p;
        double gxi = -g * sg * ax, gyi = -g * sg * ay;
        gx[i] += gxi; gx[j] -= gxi; gy[i] += gyi; gy[j] -= gyi;
        double dMi = (kk == 0) ? sg * A[1] : (kk == 1) ? -sg * A[0] : 0.0;
        double dMj = (kk == 2) ? sg * A[3] : (kk == 3) ? -sg * A[2] : 0.0;
        double dR = 0.5 * (-sgnd(cphi) * sphi + sgnd(sphi) * cphi);
        gt[i] += g * (-dR - dMi);
        gt[j] += g * (dR - dMj);
    }
    for (int i = 0; i < n; i++) {
        double c = cos(t[i]), sn = sin(t[i]);
        double w = 0.5 * (fabs(c) + fabs(sn));
        double v0 = w - x[i], v1 = x[i] + w - s, v2 = w - y[i], v3 = y[i] + w - s;
        if (v0 < 0) v0 = 0; if (v1 < 0) v1 = 0; if (v2 < 0) v2 = 0; if (v3 < 0) v3 = 0;
        E += v0 * v0 + v1 * v1 + v2 * v2 + v3 * v3;
        if (grad) {
            gx[i] += 2.0 * (v1 - v0);
            gy[i] += 2.0 * (v3 - v2);
            gt[i] += 2.0 * (v0 + v1 + v2 + v3) * 0.5 * (-sgnd(c) * sn + sgnd(sn) * c);
        }
    }
    return E;
}

/* largest penetration / protrusion of a configuration (exact pair test) */
double max_violation_c(int n, double s, const double *z) {
    PairList pl = {0};
    build_pairs(n, z, z + n, SQRT2 + 1e-9, &pl);
    const double *x = z, *y = z + n, *t = z + 2 * n;
    double worst = -1e300;
    for (int k = 0; k < pl.m; k++) {
        int i = pl.I[k], j = pl.J[k];
        double dx = x[i] - x[j], dy = y[i] - y[j];
        double ci = cos(t[i]), si = sin(t[i]), cj = cos(t[j]), sj = sin(t[j]);
        double A[4] = {ci * dx + si * dy, -si * dx + ci * dy, cj * dx + sj * dy, -sj * dx + cj * dy};
        double M = fabs(A[0]);
        for (int q = 1; q < 4; q++) if (fabs(A[q]) > M) M = fabs(A[q]);
        double phi = t[j] - t[i];
        double p = 0.5 + 0.5 * (fabs(cos(phi)) + fabs(sin(phi))) - M;
        if (p > worst) worst = p;
    }
    for (int i = 0; i < n; i++) {
        double w = 0.5 * (fabs(cos(t[i])) + fabs(sin(t[i])));
        double v = w - x[i]; if (v > worst) worst = v;
        v = x[i] + w - s; if (v > worst) worst = v;
        v = w - y[i]; if (v > worst) worst = v;
        v = y[i] + w - s; if (v > worst) worst = v;
    }
    free(pl.I); free(pl.J);
    return worst;
}

static double dot(int N, const double *a, const double *b) { double r = 0; for (int i = 0; i < N; i++) r += a[i] * b[i]; return r; }

static int needs_rebuild(int n, const double *z, const double *zref, double D) {
    for (int i = 0; i < 2 * n; i++) if (fabs(z[i] - zref[i]) > D) return 1;
    return 0;
}

/* L-BFGS minimisation of E at fixed s.  z is updated in place.  Returns iterations used;
 * *E_out receives the final energy (w.r.t. the final pair list). */
int lbfgs_c(int n, double s, double *z, int maxiter, double gtol, double ftol, double cutoff, double *E_out) {
    const int N = 3 * n, H = 10;
    double D = 0.5 * (cutoff - SQRT2) - 1e-9;
    PairList pl = {0};
    double *zref = (double *)malloc(sizeof(double) * N);
    double *g = (double *)malloc(sizeof(double) * N), *gn = (double *)malloc(sizeof(double) * N);
    double *d = (double *)malloc(sizeof(double) * N), *zn = (double *)malloc(sizeof(double) * N);
    double *S = (double *)malloc(sizeof(double) * N * H), *Y = (double *)malloc(sizeof(double) * N * H);
    double rho[10], alpha[10];
    int hist = 0, head = 0;
    memcpy(zref, z, sizeof(double) * N);
    build_pairs(n, z, z + n, cutoff, &pl);
    double E = energy_grad_c(n, s, z, pl.m, pl.I, pl.J, g);
    int it = 0, stall = 0;
    for (it = 0; it < maxiter; it++) {
        if (E < 1e-26) break;
        double gnorm = sqrt(dot(N, g, g));
        if (gnorm < gtol) break;
        /* two-loop recursion */
        for (int i = 0; i < N; i++) d[i] = -g[i];
        double gamma = 1.0;
        for (int k = 0; k < hist; k++) {
            int idx = (head - 1 - k + H) % H;
            alpha[idx] = rho[idx] * dot(N, S + idx * N, d);
            for (int i = 0; i < N; i++) d[i] -= alpha[idx] * Y[idx * N + i];
        }
        if (hist > 0) {
            int idx = (head - 1 + H) % H;
            gamma = dot(N, S + idx * N, Y + idx * N) / dot(N, Y + idx * N, Y + idx * N);
            for (int i = 0; i < N; i++) d[i] *= gamma;
        }
        for (int k = hist - 1; k >= 0; k--) {
            int idx = (head - 1 - k + H) % H;
            double beta = rho[idx] * dot(N, Y + idx * N, d);
            for (int i = 0; i < N; i++) d[i] += (alpha[idx] - beta) * S[idx * N + i];
        }
        double dg = dot(N, d, g);
        if (dg >= 0) { for (int i = 0; i < N; i++) d[i] = -g[i]; dg = -gnorm * gnorm; hist = 0; }
        /* cap the step so that no square jumps more than ~0.5 in one go */
        double dmax = 0; for (int i = 0; i < N; i++) if (fabs(d[i]) > dmax) dmax = fabs(d[i]);
        double step = (hist == 0) ? fmin(1.0, 0.1 / (dmax + 1e-300)) : 1.0;
        if (step * dmax > 0.5) step = 0.5 / dmax;
        double En = 0; int accepted = 0;
        for (int ls = 0; ls < 40; ls++) {
            for (int i = 0; i < N; i++) zn[i] = z[i] + step * d[i];
            if (needs_rebuild(n, zn, zref, D)) {
                memcpy(zref, zn, sizeof(double) * N);
                build_pairs(n, zn, zn + n, cutoff, &pl);
                E = energy_grad_c(n, s, z, pl.m, pl.I, pl.J, g);   /* re-evaluate the base point */
                dg = dot(N, d, g);
                if (dg >= 0) { for (int i = 0; i < N; i++) d[i] = -g[i]; dg = -dot(N, g, g); hist = 0; step = fmin(step, 0.1 / (dmax + 1e-300)); continue; }
            }
            En = energy_grad_c(n, s, zn, pl.m, pl.I, pl.J, gn);
            if (En <= E + 1e-4 * step * dg) { accepted = 1; break; }
            step *= 0.5;
        }
        if (!accepted) break;
        /* history update */
        double *Sk = S + head * N, *Yk = Y + head * N;
        for (int i = 0; i < N; i++) { Sk[i] = zn[i] - z[i]; Yk[i] = gn[i] - g[i]; }
        double ys = dot(N, Sk, Yk);
        if (ys > 1e-14) { rho[head] = 1.0 / ys; head = (head + 1) % H; if (hist < H) hist++; }
        double dec = E - En;
        memcpy(z, zn, sizeof(double) * N); memcpy(g, gn, sizeof(double) * N); E = En;
        if (dec <= ftol * fmax(E, 1e-300)) { if (++stall >= 8) break; } else stall = 0;
    }
    *E_out = E;
    free(pl.I); free(pl.J); free(zref); free(g); free(gn); free(d); free(zn); free(S); free(Y);
    return it;
}
