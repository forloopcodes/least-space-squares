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

/* ------------------------------------------------------------------------- */
/* Simulated annealing on E(z; s) with single-square Metropolis moves.        */
/* ------------------------------------------------------------------------- */
#define QUARTER_PI 0.78539816339744831
#define HALF_PI 1.5707963267948966

static inline unsigned long long rng_next(unsigned long long *st) {      /* xorshift64* */
    unsigned long long x = *st;
    x ^= x >> 12; x ^= x << 25; x ^= x >> 27;
    *st = x;
    return x * 0x2545F4914F6CDD1DULL;
}
static inline double rng_u(unsigned long long *st) { return (double)(rng_next(st) >> 11) * (1.0 / 9007199254740992.0); }
static inline double rng_sym(unsigned long long *st) { return 2.0 * rng_u(st) - 1.0; }
static inline double canon(double t) { return t - HALF_PI * floor((t + QUARTER_PI) / HALF_PI); }

/* squared penetration of two unit squares (0 when separated) from the centre offset and cos/sin */
static inline double pen2(double dx, double dy, double ci, double si, double cj, double sj) {
    double A0 = ci * dx + si * dy, A1 = -si * dx + ci * dy, A2 = cj * dx + sj * dy, A3 = -sj * dx + cj * dy;
    double M = fabs(A0), a;
    a = fabs(A1); if (a > M) M = a;
    a = fabs(A2); if (a > M) M = a;
    a = fabs(A3); if (a > M) M = a;
    double cphi = ci * cj + si * sj, sphi = sj * ci - cj * si;
    double p = 0.5 + 0.5 * (fabs(cphi) + fabs(sphi)) - M;
    return p > 0 ? p * p : 0.0;
}

/* energy of square i placed at (xi, yi) with cos/sin (ci, si): pairs with every other square
 * (except `skip`) plus its own container terms */
static double sq_energy(int n, double s, const double *x, const double *y, const double *c, const double *sn,
                        int i, int skip, double xi, double yi, double ci, double si) {
    double e = 0.0;
    for (int j = 0; j < n; j++) {
        if (j == i || j == skip) continue;
        double dx = xi - x[j], dy = yi - y[j];
        if (fabs(dx) >= SQRT2 || fabs(dy) >= SQRT2) continue;
        e += pen2(dx, dy, ci, si, c[j], sn[j]);
    }
    double w = 0.5 * (fabs(ci) + fabs(si)), v;
    v = w - xi; if (v > 0) e += v * v;
    v = xi + w - s; if (v > 0) e += v * v;
    v = w - yi; if (v > 0) e += v * v;
    v = yi + w - s; if (v > 0) e += v * v;
    return e;
}

static double total_energy(int n, double s, const double *x, const double *y, const double *c, const double *sn) {
    double E = 0.0;
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            double dx = x[i] - x[j], dy = y[i] - y[j];
            if (fabs(dx) >= SQRT2 || fabs(dy) >= SQRT2) continue;
            E += pen2(dx, dy, c[i], sn[i], c[j], sn[j]);
        }
        double w = 0.5 * (fabs(c[i]) + fabs(sn[i])), v;
        v = w - x[i]; if (v > 0) E += v * v;
        v = x[i] + w - s; if (v > 0) E += v * v;
        v = w - y[i]; if (v > 0) E += v * v;
        v = y[i] + w - s; if (v > 0) E += v * v;
    }
    return E;
}

/* Simulated annealing of E(z; s).  Temperature decays geometrically from T0 to T1 over
 * `nsweeps` sweeps of n single-square moves (displace / rotate / both / snap the angle to
 * 0 or pi/4 / swap two squares / teleport).  Displacement and rotation steps adapt to keep
 * the acceptance rate in a sensible window.  Whenever the tracked energy drops below `etol`
 * (a valid packing up to ~sqrt(etol)), the state is copied to `zbest` with *s_best = s and,
 * if `shrink` > 0, the container is shrunk by that relative amount (positions scaled).
 * On return z holds the final state at *s_io.  Returns the number of feasible hits. */
int anneal_c(int n, double *z, double *s_io, int nsweeps, double T0, double T1, double step_xy, double step_t,
             double shrink, double etol, unsigned long long seed, double *zbest, double *s_best,
             double *E_best, double *E_final) {
    double *x = z, *y = z + n, *t = z + 2 * n;
    double *c = (double *)malloc(sizeof(double) * n), *sn = (double *)malloc(sizeof(double) * n);
    unsigned long long st = seed * 0x9E3779B97F4A7C15ULL + 0x1234567ULL;
    if (st == 0) st = 1;
    for (int k = 0; k < 8; k++) rng_next(&st);
    double s = *s_io;
    for (int i = 0; i < n; i++) { t[i] = canon(t[i]); c[i] = cos(t[i]); sn[i] = sin(t[i]); }
    double E = total_energy(n, s, x, y, c, sn);
    int hits = 0;
    *s_best = -1.0; *E_best = 1e300;
    if (E < etol) {
        memcpy(zbest, z, sizeof(double) * 3 * n); *s_best = s; *E_best = E; hits++;
        if (shrink > 0) {
            double f = 1.0 - shrink;
            for (int i = 0; i < n; i++) { x[i] *= f; y[i] *= f; }
            s *= f; E = total_energy(n, s, x, y, c, sn);
        }
    }
    double sxy = step_xy, stt = step_t;
    double lratio = (nsweeps > 1) ? log(T1 / T0) / (double)(nsweeps - 1) : 0.0;
    for (int sweep = 0; sweep < nsweeps; sweep++) {
        double T = T0 * exp(lratio * sweep);
        int acc_xy = 0, try_xy = 0, acc_t = 0, try_t = 0;
        for (int m = 0; m < n; m++) {
            int i = (int)(rng_u(&st) * n); if (i >= n) i = n - 1;
            double r = rng_u(&st);
            double xi = x[i], yi = y[i], ti = t[i], ci = c[i], si = sn[i];
            int j = -1; double xj = 0, yj = 0;
            int kind;                                    /* 0 move, 1 rotate, 2 both, 3 snap, 4 swap, 5 teleport */
            if (r < 0.45) kind = 0; else if (r < 0.70) kind = 1; else if (r < 0.82) kind = 2;
            else if (r < 0.88) kind = 3; else if (r < 0.96) kind = 4; else kind = 5;
            double dE, e_old, e_new;
            if (kind == 4) {
                if (n < 2) continue;
                j = (int)(rng_u(&st) * (n - 1)); if (j >= i) j++; if (j >= n) j = n - 1;
                xj = x[j]; yj = y[j];
                e_old = sq_energy(n, s, x, y, c, sn, i, j, xi, yi, ci, si) + sq_energy(n, s, x, y, c, sn, j, i, xj, yj, c[j], sn[j]);
                e_new = sq_energy(n, s, x, y, c, sn, i, j, xj, yj, ci, si) + sq_energy(n, s, x, y, c, sn, j, i, xi, yi, c[j], sn[j]);
            } else {
                e_old = sq_energy(n, s, x, y, c, sn, i, -1, xi, yi, ci, si);
                double scale = (rng_u(&st) < 0.5) ? 1.0 : 0.15;
                if (kind == 0 || kind == 2) {
                    xi += sxy * scale * rng_sym(&st); yi += sxy * scale * rng_sym(&st);
                    try_xy++;
                }
                if (kind == 1 || kind == 2) {
                    ti = canon(ti + stt * scale * rng_sym(&st));
                    try_t++;
                }
                if (kind == 3) ti = (rng_u(&st) < 0.5) ? 0.0 : QUARTER_PI;
                if (kind == 5) {
                    xi = 0.5 + rng_u(&st) * (s - 1.0); yi = 0.5 + rng_u(&st) * (s - 1.0);
                    double q = rng_u(&st);
                    ti = q < 0.4 ? 0.0 : (q < 0.7 ? QUARTER_PI : (rng_u(&st) - 0.5) * HALF_PI);
                }
                if (xi < 0.3) xi = 0.3; if (xi > s - 0.3) xi = s - 0.3;
                if (yi < 0.3) yi = 0.3; if (yi > s - 0.3) yi = s - 0.3;
                if (kind != 0) { ci = cos(ti); si = sin(ti); }
                e_new = sq_energy(n, s, x, y, c, sn, i, -1, xi, yi, ci, si);
            }
            dE = e_new - e_old;
            int accept = (dE <= 0.0) || (T > 0 && rng_u(&st) < exp(-dE / T));
            if (!accept) continue;
            if (kind == 4) {
                x[i] = xj; y[i] = yj; x[j] = xi; y[j] = yi;
            } else {
                x[i] = xi; y[i] = yi; t[i] = ti; c[i] = ci; sn[i] = si;
                if (kind == 0 || kind == 2) acc_xy++;
                if (kind == 1 || kind == 2) acc_t++;
            }
            double Eprev = E;
            E += dE;
            if (E < etol && Eprev >= etol) {
                E = total_energy(n, s, x, y, c, sn);
                if (E < etol) {
                    memcpy(zbest, z, sizeof(double) * 3 * n); *s_best = s; *E_best = E; hits++;
                    if (shrink > 0) {
                        double f = 1.0 - shrink;
                        for (int k = 0; k < n; k++) { x[k] *= f; y[k] *= f; }
                        s *= f; E = total_energy(n, s, x, y, c, sn);
                    }
                }
            }
        }
        /* adapt the step sizes towards ~35% acceptance */
        if (try_xy >= 8) { double a = (double)acc_xy / try_xy; if (a < 0.25) sxy *= 0.85; else if (a > 0.45) sxy *= 1.15; }
        if (try_t >= 8) { double a = (double)acc_t / try_t; if (a < 0.25) stt *= 0.85; else if (a > 0.45) stt *= 1.15; }
        if (sxy < 1e-5) sxy = 1e-5; if (sxy > 0.5) sxy = 0.5;
        if (stt < 1e-5) stt = 1e-5; if (stt > 0.8) stt = 0.8;
        E = total_energy(n, s, x, y, c, sn);         /* kill accumulated round-off once per sweep */
    }
    *E_final = E; *s_io = s;
    free(c); free(sn);
    return hits;
}
