#include "cpu_mpi.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace microsim {
double interpolation(double p) {
    return p * p * p * (10 + p * (-15 + 6 * p));
}
double interpolation_derivative(double p) {
    return 30 * p * p * (1 - p) * (1 - p);
}
PhaseField::PhaseField(FFT &f, Elasticity &e, PhaseFieldControls p, Loading l)
    : fft_(f), elastic_(e), p_(p), loading_(l) {
    require(p.variants > 0 && size_t(p.variants + 1) == e.phases().size(),
            "Require one matrix phase plus one phase per variant", f.comm);
    require((p.mobility_method == 0 || p.mobility_method == 1 || p.mobility_method == 3) &&
                p.dt > 0 && p.kinetic >= 0 && p.kappa >= 0 && p.omega >= 0 && p.theta >= 0 &&
                p.a_matrix > 0 && p.a_precip > 0 && p.diffusivity >= 0,
            "Invalid phase-field controls (mobility methods 0, 1 or corrected 3)", f.comm);
    bool finite = p.theta2 >= 0;
    for (double x : {p.dt, p.kinetic, p.kappa, p.omega, p.theta, p.theta2, p.a_matrix, p.a_precip,
                     p.c_matrix, p.c_precip, p.diffusivity})
        finite = finite && std::isfinite(x);
    require(finite, "Nonfinite phase-field parameter", f.comm);
}
void PhaseField::fractions(const State &s, std::vector<Real> &h) const {
    bool valid = s.phi.size() == size_t(p_.variants) && s.composition.size() == fft_.real_size;
    for (auto &phi : s.phi)
        valid = valid && phi.size() == fft_.real_size;
    require(valid, "State dimensions differ from grid/variant count", fft_.comm);
    h.assign(p_.variants + 1, Real(fft_.real_size));
    std::fill(h[0].begin(), h[0].end(), 1);
    for (int p = 0; p < p_.variants; ++p)
        for (size_t i = 0; i < fft_.real_size; ++i) {
            double v = s.phi[p][i];
            valid = valid && std::isfinite(v) && v >= -1e-10 && v <= 1 + 1e-10;
            h[p + 1][i] = interpolation(v);
            h[0][i] -= h[p + 1][i];
        }
    require(valid, "Phase field left [0,1]; reduce dt. No clipping is performed.", fft_.comm);
}
Diagnostics PhaseField::evaluate(const State &s, std::vector<Real> *force, Real *mu,
                                 ElasticResult *output) {
    std::vector<Real> h;
    fractions(s, h);
    ElasticResult el;
    if (p_.elasticity)
        el = elastic_.solve(h, loading_);
    else {
        el.status.converged = true;
        el.status.reason = "disabled";
        el.energy.assign(fft_.real_size, 0);
        el.fraction_derivative.assign(h.size(), Real(fft_.real_size, 0));
        for (int a = 0; a < 6; ++a) {
            el.strain[a].assign(fft_.real_size, 0);
            el.stress[a].assign(fft_.real_size, 0);
        }
        for (auto &u : el.displacement)
            u.assign(fft_.real_size, 0);
    }
    require(el.status.converged, "Elastic solve hit iteration limit; phase-field step refused",
            fft_.comm);
    if (force)
        force->assign(p_.variants, Real(fft_.real_size));
    if (mu)
        mu->resize(fft_.real_size);
    Real chemical(fft_.real_size), well(fft_.real_size);
    bool valid = true;
    for (size_t i = 0; i < fft_.real_size; ++i) {
        double hp = 1 - h[0][i], ratio = p_.a_matrix / p_.a_precip, den = 1 - hp + hp * ratio;
        double ca = (s.composition[i] - hp * (p_.c_precip - ratio * p_.c_matrix)) / den;
        double cb = p_.c_precip + ratio * (ca - p_.c_matrix),
               m = 2 * p_.a_matrix * (ca - p_.c_matrix);
        double fa = p_.a_matrix * (ca - p_.c_matrix) * (ca - p_.c_matrix),
               fb = p_.a_precip * (cb - p_.c_precip) * (cb - p_.c_precip);
        chemical[i] = (1 - hp) * fa + hp * fb;
        double chem_force = fb - fa + (ca - cb) * m;
        if (mu)
            (*mu)[i] = m;
        valid = valid && std::isfinite(ca) && std::isfinite(cb) && std::isfinite(m) &&
                std::isfinite(chemical[i]);
        double squares = 0;
        for (auto &phi : s.phi)
            squares += phi[i] * phi[i];
        for (int p = 0; p < p_.variants; ++p) {
            double v = s.phi[p][i], pairs = 0;
            for (int a = 0; a < p_.variants; ++a)
                for (int b = a + 1; b < p_.variants; ++b)
                    if (a != p && b != p)
                        pairs += s.phi[a][i] * s.phi[a][i] * s.phi[b][i] * s.phi[b][i];
            well[i] +=
                p_.omega * (v * v * (1 - v) * (1 - v) + .5 * p_.theta * v * v * (squares - v * v) +
                            (p_.theta2 / 3) * v * v * pairs);
            if (force)
                (*force)[p][i] =
                    p_.omega * (2 * v * (1 - v) * (1 - 2 * v) +
                                2 * p_.theta * v * (squares - v * v) + 2 * p_.theta2 * v * pairs) +
                    interpolation_derivative(v) * (chem_force + el.fraction_derivative[p + 1][i] -
                                                   el.fraction_derivative[0][i]);
        }
    }
    for (double value : well)
        valid = valid && std::isfinite(value);
    if (force)
        for (auto &field : *force)
            for (double value : field)
                valid = valid && std::isfinite(value);
    require(valid, "Nonfinite chemical/barrier/force state", fft_.comm);
    // Gradient coefficient is kappa/2 in the energy, consistent with the
    // original Update_phi denominator 1+L*kappa*k^2*dt.
    double grad_sum = 0;
    Fourier f;
    for (auto &phi : s.phi) {
        fft_.forward(phi, f);
        for (size_t i = 0; i < f.size(); ++i) {
            auto k = fft_.wave(i, false);
            double k2 = k[0] * k[0] + k[1] * k[1] + k[2] * k[2];
            size_t z = i % size_t(fft_.grid.nz / 2 + 1);
            double weight =
                (z == 0 || (fft_.grid.nz % 2 == 0 && z == size_t(fft_.grid.nz / 2))) ? 1 : 2;
            grad_sum += weight * k2 * std::norm(f[i]);
        }
    }
    double grad_mean =
        .5 * p_.kappa * sum(grad_sum, fft_.comm) / (fft_.grid.points() * fft_.grid.points());
    Diagnostics d;
    d.composition_mean = fft_.mean(s.composition);
    d.energy = fft_.mean(chemical) + fft_.mean(well) + grad_mean + el.status.mean_energy;
    d.potential = d.energy + el.status.mean_potential - el.status.mean_energy;
    d.elastic = el.status;
    if (output)
        *output = std::move(el);
    return d;
}
Diagnostics PhaseField::advance(State &state) {
    std::vector<Real> force;
    Real mu;
    Diagnostics d = evaluate(state, &force, &mu);
    // All explicit forces use the same old state. Commit only after every
    // field has passed finite/range validation; never silently project mass.
    State next = state;
    Fourier c, drive;
    fft_.forward(state.composition, c);
    if (p_.evolve_composition) {
        fft_.forward(mu, drive);
        if (p_.mobility_method == 1) {
            Fourier div(fft_.fourier_size, 0);
            Real gradient;
            for (int a = 0; a < 3; ++a) {
                Fourier f = drive;
                for (size_t i = 0; i < f.size(); ++i)
                    f[i] *= Complex(0, fft_.wave(i)[a]);
                fft_.inverse(f, gradient);
                for (size_t i = 0; i < gradient.size(); ++i) {
                    double hp = 0;
                    for (auto &phi : state.phi)
                        hp += interpolation(phi[i]);
                    gradient[i] *=
                        p_.diffusivity * ((1 - hp) / (2 * p_.a_matrix) + hp / (2 * p_.a_precip));
                }
                fft_.forward(gradient, f);
                for (size_t i = 0; i < f.size(); ++i)
                    div[i] += Complex(0, fft_.wave(i)[a]) * f[i];
            }
            for (size_t i = 0; i < c.size(); ++i)
                c[i] += p_.dt * div[i];
        } else {
            double mobility = p_.diffusivity / (2 * p_.a_matrix);
            if (p_.mobility_method == 3) {
                Real hp(fft_.real_size);
                for (size_t i = 0; i < hp.size(); ++i)
                    for (auto &phi : state.phi)
                        hp[i] += interpolation(phi[i]);
                double v = fft_.mean(hp);
                mobility = p_.diffusivity * ((1 - v) / (2 * p_.a_matrix) + v / (2 * p_.a_precip));
            }
            for (size_t i = 0; i < c.size(); ++i) {
                auto k = fft_.wave(i, false);
                double k2 = k[0] * k[0] + k[1] * k[1] + k[2] * k[2];
                c[i] -= p_.dt * mobility * k2 * drive[i];
            }
        }
        fft_.inverse(c, next.composition);
    }
    for (int p = 0; p < p_.variants; ++p) {
        Fourier phi, df;
        fft_.forward(state.phi[p], phi);
        fft_.forward(force[p], df);
        for (size_t i = 0; i < phi.size(); ++i) {
            auto k = fft_.wave(i, false);
            double k2 = k[0] * k[0] + k[1] * k[1] + k[2] * k[2];
            phi[i] =
                (phi[i] - p_.dt * p_.kinetic * df[i]) / (1 + p_.dt * p_.kinetic * p_.kappa * k2);
        }
        fft_.inverse(phi, next.phi[p]);
    }
    require(next.step < std::numeric_limits<int>::max() && std::isfinite(next.time + p_.dt),
            "Step/time overflow", fft_.comm);
    next.step++;
    next.time += p_.dt;
    std::vector<Real> h;
    fractions(next, h);
    bool valid = true;
    for (size_t i = 0; i < next.composition.size(); ++i) {
        valid = valid && std::isfinite(next.composition[i]);
        double total = 0;
        for (auto &f : h) {
            valid = valid && std::isfinite(f[i]) && f[i] >= -1e-12 && f[i] <= 1 + 1e-12;
            total += f[i];
        }
        valid = valid && std::abs(total - 1) < 1e-10;
    }
    require(valid, "Invalid proposed state (including variant overlap); reduce dt", fft_.comm);
    state = std::move(next);
    return d;
}
} // namespace microsim
