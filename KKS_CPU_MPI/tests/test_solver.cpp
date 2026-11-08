#include "cpu_mpi.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

using namespace microsim;
static int checks = 0;
static void check(FFT &f, bool pass, const std::string &message) {
    require(pass, "TEST FAILURE: " + message, f.comm);
    ++checks;
}
static Controls strict_controls() {
    Controls c;
    c.max_iterations = 2000;
    c.change_tolerance = 1e-12;
    c.equilibrium_tolerance = 1e-10;
    c.mean_stress_tolerance = 1e-12;
    return c;
}
static std::vector<Real> fractions(FFT &f, bool uniform = false) {
    std::vector<Real> h(2, Real(f.real_size));
    for (size_t i = 0; i < f.real_size; ++i) {
        auto x = f.coordinate(i);
        h[1][i] = uniform ? .4
                          : .4 + .13 * std::sin(2 * pi * (x[0] + .5) / f.grid.nx) *
                                     std::cos(2 * pi * (x[1] + .5) / f.grid.ny);
        h[0][i] = 1 - h[1][i];
    }
    return h;
}
static double error(FFT &f, const Real &a, const Real &b) {
    double e = 0;
    for (size_t i = 0; i < a.size(); ++i)
        e = std::max(e, std::abs(a[i] - b[i]));
    return maximum(e, f.comm);
}
static void test_fft(Grid grid) {
    FFT f(grid, MPI_COMM_WORLD);
    Real r(f.real_size);
    for (size_t i = 0; i < r.size(); ++i) {
        auto c = f.coordinate(i);
        r[i] = .3 + std::sin(2 * pi * c[0] / grid.nx) + .2 * std::cos(2 * pi * c[1] / grid.ny);
    }
    Fourier a;
    Real b;
    f.forward(r, a);
    f.inverse(a, b);
    check(f, error(f, r, b) < 2e-14, "R2C/C2R transposed MPI roundtrip");
    for (int axis = 0; axis < 3; ++axis) {
        if ((axis == 0 ? grid.nx : axis == 1 ? grid.ny : grid.nz) == 1)
            continue;
        Real expected(f.real_size);
        double L = axis == 0   ? grid.nx * grid.dx
                   : axis == 1 ? grid.ny * grid.dy
                               : grid.nz * grid.dz;
        for (size_t i = 0; i < r.size(); ++i) {
            auto c = f.coordinate(i);
            double n = axis == 0 ? grid.nx : axis == 1 ? grid.ny : grid.nz;
            double arg = 2 * pi * c[axis] / n;
            r[i] = std::sin(arg);
            expected[i] = 2 * pi / L * std::cos(arg);
        }
        f.forward(r, a);
        for (size_t i = 0; i < a.size(); ++i)
            a[i] *= Complex(0, f.wave(i)[axis]);
        f.inverse(a, b);
        check(f, error(f, b, expected) < 5e-14, "Spectral derivative sign/layout");
    }
    if (grid.nx % 2 == 0) {
        for (size_t i = 0; i < r.size(); ++i)
            r[i] = (f.coordinate(i)[0] % 2) ? -1 : 1;
        f.forward(r, a);
        for (size_t i = 0; i < a.size(); ++i)
            a[i] *= Complex(0, f.wave(i)[0]);
        f.inverse(a, b);
        double v = 0;
        for (double x : b)
            v = std::max(v, std::abs(x));
        check(f, maximum(v, f.comm) < 1e-14, "Even-grid Nyquist derivative is zero");
    }
}
static void test_uniform() {
    FFT f({5, 7, 3, 1, 1, 1}, MPI_COMM_WORLD);
    Phase p0{{300, 100, 80}, {0, 0, 0, 0, 0, 0}},
        p1{{600, 200, 160}, {0, .012, -.003, .002, -.004, .006}};
    Elasticity solver(f, {p0, p1}, strict_controls());
    auto h = fractions(f, true);
    Loading load;
    load.stress = {.3, -.2, .1, .07, -.03, .04};
    auto r = solver.solve(h, load);
    check(f, r.status.converged, "Uniform anisotropic full tensor convergence");
    Tensor eig = .4 * p1.eigenstrain;
    double maxerr = 0;
    for (size_t i = 0; i < f.real_size; ++i) {
        Tensor eps;
        for (int a = 0; a < 6; ++a)
            eps[a] = r.strain[a][i] - eig[a];
        auto sigma = Cubic{420, 140, 112}.stress(eps);
        for (int a = 0; a < 6; ++a)
            maxerr = std::max(maxerr, std::abs(sigma[a] - load.stress[a]));
    }
    check(f, maximum(maxerr, f.comm) < 2e-13,
          "Uniform stress control, pure yy and shear eigenstrain");
    Loading fixed;
    fixed.fixed_strain.fill(true);
    fixed.strain = {.001, .002, -.001, .003, .004, -.002};
    r = solver.solve(h, fixed);
    check(f, r.status.converged, "Uniform fixed strain convergence");
    Tensor expected = Cubic{420, 140, 112}.stress(fixed.strain - eig);
    for (int a = 0; a < 6; ++a)
        check(f, std::abs(r.status.mean_stress[a] - expected[a]) < 2e-13,
              "Fixed strain constitutive response");
}
static void test_compatible() {
    FFT f({9, 7, 5, 1, 1, 1}, MPI_COMM_WORLD);
    Phase m{{300, 100, 80}, {}}, p{{450, 150, 120}, {.01, 0, 0, 0, .003, .004}};
    Elasticity e(f, {m, p}, strict_controls());
    auto h = fractions(f);
    // Make h depend on x only: eps_xx,eps_xy,eps_xz are all compatible.
    for (size_t i = 0; i < f.real_size; ++i) {
        auto c = f.coordinate(i);
        h[1][i] = .4 + .1 * std::sin(2 * pi * (c[0] + .5) / f.grid.nx);
        h[0][i] = 1 - h[1][i];
    }
    auto r = e.solve(h);
    check(f, r.status.converged, "Compatible eigenstrain convergence");
    double maxstress = 0;
    for (auto &s : r.stress)
        for (double x : s)
            maxstress = std::max(maxstress, std::abs(x));
    check(f, maximum(maxstress, f.comm) < 2e-10, "Compatible eigenstrain is stress free");
}
static void test_laminate() {
    FFT f({9, 5, 3, 1, 1, 1}, MPI_COMM_WORLD);
    Phase m{{300, 100, 80}, {0, 0, 0, 0, 0, 0}},
        p{{600, 150, 100}, {.01, .006, -.002, .001, .002, -.003}};
    auto h = fractions(f);
    for (size_t i = 0; i < f.real_size; ++i) {
        h[1][i] = f.coordinate(i)[0] < 3 ? 1 : 0;
        h[0][i] = 1 - h[1][i];
    }
    Loading l;
    l.stress = {.2, 0, 0, 0, .03, .04};
    l.fixed_strain = {false, true, true, true, false, false};
    l.strain = {0, .002, -.001, .0007, 0, 0};
    Elasticity e(f, {m, p}, strict_controls());
    auto r = e.solve(h, l);
    check(f, r.status.converged, "Sharp laminate converged");
    double err = 0;
    for (size_t i = 0; i < f.real_size; ++i) {
        const auto &phase = h[1][i] == 1 ? p : m;
        auto eig = phase.eigenstrain;
        auto C = phase.stiffness;
        Tensor eps = l.strain;
        eps[0] = eig[0] + (l.stress[0] - C.c12 * (eps[1] - eig[1] + eps[2] - eig[2])) / C.c11;
        eps[4] = eig[4] + l.stress[4] / (2 * C.c44);
        eps[5] = eig[5] + l.stress[5] / (2 * C.c44);
        auto sig = C.stress(eps - eig);
        for (int a = 0; a < 6; ++a) {
            err = std::max(err, std::abs(r.strain[a][i] - eps[a]));
            err = std::max(err, std::abs(r.stress[a][i] - sig[a]));
        }
    }
    check(f, maximum(err, f.comm) < 2e-10,
          "Laminate independent pointwise analytical solution and mixed loading");
}
static void test_relaxed_derivative() {
    FFT f({7, 5, 3, 1, 1, 1}, MPI_COMM_WORLD);
    Phase a{{300, 100, 80}, {}}, b{{450, 130, 110}, {-.005, .009, -.003, .002, -.001, .003}};
    auto h = fractions(f);
    Elasticity e(f, {a, b}, strict_controls());
    Loading l;
    l.stress = {.15, -.03, .02, .04, .01, -.02};
    auto r = e.solve(h, l);
    check(f, r.status.converged, "Derivative baseline converged");
    double analytic = 0;
    Real direction(f.real_size);
    for (size_t i = 0; i < f.real_size; ++i) {
        auto c = f.coordinate(i);
        direction[i] =
            .3 + std::cos(2 * pi * c[0] / f.grid.nx) * std::sin(2 * pi * c[1] / f.grid.ny);
        analytic += (r.fraction_derivative[1][i] - r.fraction_derivative[0][i]) * direction[i];
    }
    analytic = sum(analytic, f.comm) / f.grid.points();
    double eps = 1e-5, energies[2];
    for (int s = 0; s < 2; ++s) {
        auto hh = h;
        for (size_t i = 0; i < f.real_size; ++i) {
            double d = (s ? 1 : -1) * eps * direction[i];
            hh[1][i] += d;
            hh[0][i] -= d;
        }
        auto rr = e.solve(hh, l);
        check(f, rr.status.converged, "Perturbed equilibrium converged");
        energies[s] = rr.status.mean_potential;
    }
    double numerical = (energies[1] - energies[0]) / (2 * eps);
    check(f, std::abs(numerical - analytic) < 2e-9,
          "Relaxed stress-controlled elastic potential derivative");
}
static void test_phasefield() {
    FFT f({7, 5, 3, 1, 1, 1}, MPI_COMM_WORLD);
    std::vector<Phase> phases(5);
    phases[0] = {{300, 100, 80}, {}};
    for (int p = 1; p < 5; ++p)
        phases[p] = {{450, 130, 110}, {-.005, .009, -.003, .002, -.001, .003}};
    Elasticity e(f, phases, strict_controls());
    PhaseFieldControls p;
    p.variants = 4;
    p.theta2 = 3;
    p.dt = .0001;
    p.a_matrix = 1.3;
    p.a_precip = .7;
    p.mobility_method = 1;
    Loading load;
    load.stress = {.01, 0, 0, 0, .001, 0};
    PhaseField program(f, e, p, load);
    State s;
    s.composition.resize(f.real_size);
    s.phi.assign(4, Real(f.real_size));
    for (size_t i = 0; i < f.real_size; ++i) {
        auto c = f.coordinate(i);
        s.composition[i] = .19 + .001 * std::cos(2 * pi * c[1] / f.grid.ny);
        for (int a = 0; a < 4; ++a)
            s.phi[a][i] = .12 + .01 * std::sin(2 * pi * (c[0] + a) / f.grid.nx);
    }
    std::vector<Real> force;
    auto d = program.evaluate(s, &force);
    Real direction(f.real_size);
    double analytic = 0;
    Fourier phi, lap;
    f.forward(s.phi[1], phi);
    lap = phi;
    for (size_t i = 0; i < lap.size(); ++i) {
        auto k = f.wave(i, false);
        lap[i] *= p.kappa * (k[0] * k[0] + k[1] * k[1] + k[2] * k[2]);
    }
    Real gradient;
    f.inverse(lap, gradient);
    for (size_t i = 0; i < f.real_size; ++i) {
        auto c = f.coordinate(i);
        direction[i] = .4 + std::cos(2 * pi * c[0] / f.grid.nx);
        analytic += (force[1][i] + gradient[i]) * direction[i];
    }
    analytic = sum(analytic, f.comm) / f.grid.points();
    double energies[2], eps = 1e-6;
    for (int sign = 0; sign < 2; ++sign) {
        auto pert = s;
        for (size_t i = 0; i < f.real_size; ++i)
            pert.phi[1][i] += (sign ? 1 : -1) * eps * direction[i];
        energies[sign] = program.evaluate(pert).potential;
    }
    check(f, std::abs((energies[1] - energies[0]) / (2 * eps) - analytic) < 3e-9,
          "Full phase-field energy derivative: KKS, barrier, gradient, elasticity, one chain rule");
    for (int step = 0; step < 3; ++step)
        program.advance(s);
    auto after = program.evaluate(s);
    check(f, std::abs(d.composition_mean - after.composition_mean) < 2e-14,
          "Variable mobility composition conservation");
    check(f, after.potential <= d.potential + 1e-12, "Small-step coupled potential decreases");
    for (int method : {0, 3}) {
        p.mobility_method = method;
        PhaseField other(f, e, p, load);
        double before = f.mean(s.composition);
        other.advance(s);
        check(f, std::abs(f.mean(s.composition) - before) < 2e-14,
              "Constant/weighted mobility conservation");
    }
}
static void test_failure() {
    FFT f({5, 3, 3, 1, 1, 1}, MPI_COMM_WORLD);
    auto h = fractions(f);
    Controls c = strict_controls();
    c.max_iterations = 1;
    Elasticity e(f, {{{300, 100, 80}, {}}, {{600, 200, 160}, {.01, .02, -.01, 0, 0, 0}}}, c);
    auto r = e.solve(h);
    check(f,
          !r.status.converged && r.status.reason == "iteration_limit" &&
              r.fraction_derivative.empty(),
          "Cap is failure and exposes no force");
    bool caught = false;
    auto bad = h;
    if (!bad[0].empty())
        bad[0][0] = -1;
    try {
        e.solve(bad);
    } catch (const std::exception &) {
        caught = true;
    }
    check(f, caught, "Invalid fraction rejected collectively");
}
int main(int argc, char **argv) {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    try {
        fftw_init_threads();
        fftw_mpi_init();
        {
            test_fft({8, 6, 4, 1, 1, 1});
            test_fft({7, 5, 3, .7, 1.3, 2});
            test_fft({3, 5, 1, 1, 1, 1});
            test_uniform();
            test_compatible();
            test_laminate();
            test_relaxed_derivative();
            test_phasefield();
            test_failure();
        }
        if (!rank)
            std::cout << "PASS " << checks << " collective checks\n";
        fftw_mpi_cleanup();
        fftw_cleanup_threads();
        MPI_Finalize();
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "Rank " << rank << ": " << e.what() << "\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
}
