#include "cpu_mpi.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace microsim {
void require(bool valid, const std::string &message, MPI_Comm comm) {
    int local = valid ? 1 : 0, global = 0;
    MPI_Allreduce(&local, &global, 1, MPI_INT, MPI_MIN, comm);
    if (!global)
        throw std::runtime_error(message);
}
double sum(double local, MPI_Comm comm) {
    double v;
    MPI_Allreduce(&local, &v, 1, MPI_DOUBLE, MPI_SUM, comm);
    return v;
}
double maximum(double local, MPI_Comm comm) {
    double v;
    MPI_Allreduce(&local, &v, 1, MPI_DOUBLE, MPI_MAX, comm);
    return v;
}
double dot(const Tensor &a, const Tensor &b) {
    double s = 0;
    for (int j = 0; j < 6; ++j)
        s += (j < 3 ? 1 : 2) * a[j] * b[j];
    return s;
}
Tensor operator+(Tensor a, const Tensor &b) {
    for (int j = 0; j < 6; ++j)
        a[j] += b[j];
    return a;
}
Tensor operator-(Tensor a, const Tensor &b) {
    for (int j = 0; j < 6; ++j)
        a[j] -= b[j];
    return a;
}
Tensor operator*(double a, Tensor b) {
    for (double &v : b)
        v *= a;
    return b;
}
Tensor Cubic::stress(const Tensor &e) const {
    return {c11 * e[0] + c12 * (e[1] + e[2]),
            c11 * e[1] + c12 * (e[0] + e[2]),
            c11 * e[2] + c12 * (e[0] + e[1]),
            2 * c44 * e[3],
            2 * c44 * e[4],
            2 * c44 * e[5]};
}
void Cubic::validate() const {
    if (!std::isfinite(c11) || !std::isfinite(c12) || !std::isfinite(c44) || c44 <= 0 ||
        c11 - c12 <= 0 || c11 + 2 * c12 <= 0)
        throw std::runtime_error(
            "Cubic stiffness must be finite and positive definite (no exact void/rigid phases).");
}
FFT::FFT(Grid g, MPI_Comm communicator, bool measure) : grid(g), comm(communicator) {
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &ranks);
    require(g.nx > 0 && g.ny > 0 && g.nz > 0 && g.dx > 0 && g.dy > 0 && g.dz > 0 &&
                std::isfinite(g.dx) && std::isfinite(g.dy) && std::isfinite(g.dz) &&
                std::isfinite(g.nx * g.dx) && std::isfinite(g.ny * g.dy) &&
                std::isfinite(g.nz * g.dz) &&
                g.nx <= std::numeric_limits<ptrdiff_t>::max() / 16 / g.ny / g.nz,
            "Invalid/overflowing grid or spacing", comm);
    const auto n = fftw_mpi_local_size_3d_transposed(g.nx, g.ny, g.nz / 2 + 1, comm, &local_nx,
                                                     &x_start, &local_ny, &y_start);
    real_size = size_t(local_nx) * size_t(g.ny) * size_t(g.nz);
    fourier_size = size_t(local_ny) * size_t(g.nx) * size_t(g.nz / 2 + 1);
    real_ = fftw_alloc_real(size_t(std::max<ptrdiff_t>(1, 2 * n)));
    complex_ = fftw_alloc_complex(size_t(std::max<ptrdiff_t>(1, n)));
    require(real_ && complex_, "FFTW buffer allocation failed", comm);
    unsigned flags = measure ? FFTW_MEASURE : FFTW_ESTIMATE;
    forward_ = fftw_mpi_plan_dft_r2c_3d(g.nx, g.ny, g.nz, real_, complex_, comm,
                                        flags | FFTW_MPI_TRANSPOSED_OUT);
    backward_ = fftw_mpi_plan_dft_c2r_3d(g.nx, g.ny, g.nz, complex_, real_, comm,
                                         flags | FFTW_MPI_TRANSPOSED_IN);
    require(forward_ && backward_, "FFTW MPI plan creation failed", comm);
}
FFT::~FFT() {
    if (forward_)
        fftw_destroy_plan(forward_);
    if (backward_)
        fftw_destroy_plan(backward_);
    fftw_free(real_);
    fftw_free(complex_);
}
void FFT::forward(const Real &input, Fourier &output) {
    require(input.size() == real_size, "Incorrect local real field size", comm);
    const size_t pad = 2 * size_t(grid.nz / 2 + 1);
    for (size_t i = 0; i < size_t(local_nx) * size_t(grid.ny); ++i)
        std::copy_n(input.data() + i * grid.nz, size_t(grid.nz), real_ + i * pad);
    fftw_execute(forward_);
    output.resize(fourier_size);
    for (size_t i = 0; i < fourier_size; ++i)
        output[i] = {complex_[i][0], complex_[i][1]};
}
void FFT::inverse(const Fourier &input, Real &output) {
    require(input.size() == fourier_size, "Incorrect local Fourier field size", comm);
    for (size_t i = 0; i < fourier_size; ++i) {
        complex_[i][0] = input[i].real();
        complex_[i][1] = input[i].imag();
    }
    fftw_execute(backward_);
    output.resize(real_size);
    const size_t pad = 2 * size_t(grid.nz / 2 + 1);
    for (size_t i = 0; i < size_t(local_nx) * size_t(grid.ny); ++i)
        for (size_t z = 0; z < size_t(grid.nz); ++z)
            output[i * grid.nz + z] = real_[i * pad + z] / grid.points();
}
std::array<ptrdiff_t, 3> FFT::coordinate(size_t i) const {
    auto z = ptrdiff_t(i % size_t(grid.nz));
    i /= size_t(grid.nz);
    auto y = ptrdiff_t(i % size_t(grid.ny));
    return {x_start + ptrdiff_t(i / size_t(grid.ny)), y, z};
}
std::array<double, 3> FFT::wave(size_t i, bool derivative) const {
    ptrdiff_t kz = ptrdiff_t(i % size_t(grid.nz / 2 + 1));
    i /= size_t(grid.nz / 2 + 1);
    ptrdiff_t kx = ptrdiff_t(i % size_t(grid.nx)), ky = y_start + ptrdiff_t(i / size_t(grid.nx));
    auto frequency = [derivative](ptrdiff_t k, ptrdiff_t n, double d) {
        if (derivative && n % 2 == 0 && k == n / 2)
            return 0.;
        ptrdiff_t signed_k = (k <= (n - 1) / 2) ? k : k - n;
        return 2 * pi * double(signed_k) / (double(n) * d);
    };
    return {frequency(kx, grid.nx, grid.dx), frequency(ky, grid.ny, grid.dy),
            frequency(kz, grid.nz, grid.dz)};
}
double FFT::mean(const Real &f) const {
    require(f.size() == real_size, "Incorrect field size", comm);
    return sum(std::accumulate(f.begin(), f.end(), 0.), comm) / grid.points();
}
double FFT::rms(const Real &f) const {
    double v = 0;
    for (double x : f)
        v += x * x;
    return std::sqrt(sum(v, comm) / grid.points());
}

static Tensor at(const Fields &f, size_t i) {
    Tensor x;
    for (int a = 0; a < 6; ++a)
        x[a] = f[a][i];
    return x;
}
static void put(Fields &f, size_t i, const Tensor &x) {
    for (int a = 0; a < 6; ++a)
        f[a][i] = x[a];
}
static void resize(Fields &f, size_t n) {
    for (auto &x : f)
        x.assign(n, 0.);
}
// Gaussian elimination is relative-scaled; avoids dimensionful determinant cutoffs.
static std::vector<double> linear(std::vector<std::vector<double>> A, std::vector<double> b) {
    const size_t n = b.size();
    double scale = 0;
    for (auto &r : A)
        for (double v : r)
            scale = std::max(scale, std::abs(v));
    for (size_t k = 0; k < n; ++k) {
        size_t p = k;
        for (size_t j = k + 1; j < n; ++j)
            if (std::abs(A[j][k]) > std::abs(A[p][k]))
                p = j;
        if (!(std::abs(A[p][k]) > 32 * std::numeric_limits<double>::epsilon() * scale))
            throw std::runtime_error("Singular or numerically ill-conditioned elastic system");
        std::swap(A[p], A[k]);
        std::swap(b[p], b[k]);
        for (size_t j = k + 1; j < n; ++j) {
            double t = A[j][k] / A[k][k];
            for (size_t l = k; l < n; ++l)
                A[j][l] -= t * A[k][l];
            b[j] -= t * b[k];
        }
    }
    for (size_t k = n; k-- > 0;) {
        for (size_t j = k + 1; j < n; ++j)
            b[k] -= A[k][j] * b[j];
        b[k] /= A[k][k];
    }
    return b;
}
Elasticity::Elasticity(FFT &fft, std::vector<Phase> phases, Controls controls)
    : fft_(fft), phases_(std::move(phases)), controls_(controls) {
    require(!phases_.empty(), "At least one phase is required", fft_.comm);
    bool valid = controls.max_iterations > 0 && controls.change_tolerance > 0 &&
                 controls.equilibrium_tolerance > 0 && controls.mean_stress_tolerance > 0 &&
                 std::isfinite(controls.change_tolerance) &&
                 std::isfinite(controls.equilibrium_tolerance) &&
                 std::isfinite(controls.mean_stress_tolerance);
    for (auto &p : phases_) {
        p.stiffness.validate();
        for (double v : p.eigenstrain)
            valid = valid && std::isfinite(v);
    }
    require(valid, "Invalid elastic controls/eigenstrain", fft_.comm);
    // Repeated order variants of one material must not change the original
    // two-material midpoint reference. For >2 distinct materials use a
    // positive isotropic reference bounding all phase stiffness eigenvalues.
    std::vector<Cubic> distinct;
    for (auto &p : phases_) {
        bool seen = false;
        for (auto c : distinct)
            seen = seen || (c.c11 == p.stiffness.c11 && c.c12 == p.stiffness.c12 &&
                            c.c44 == p.stiffness.c44);
        if (!seen)
            distinct.push_back(p.stiffness);
    }
    reference_ = {0, 0, 0};
    if (distinct.size() <= 2) {
        for (auto c : distinct) {
            reference_.c11 += c.c11 / distinct.size();
            reference_.c12 += c.c12 / distinct.size();
            reference_.c44 += c.c44 / distinct.size();
        }
    } else {
        double bound = 0;
        for (auto c : distinct)
            bound = std::max({bound, c.c11 + 2 * c.c12, c.c11 - c.c12, 2 * c.c44});
        reference_ = {bound, 0, .5 * bound};
    }
    for (auto &g : green_)
        g.assign(fft_.fourier_size, 0);
    for (size_t i = 0; i < fft_.fourier_size; ++i) {
        auto k = fft_.wave(i);
        double norm = std::sqrt(k[0] * k[0] + k[1] * k[1] + k[2] * k[2]);
        if (norm == 0)
            continue;
        double n[3] = {k[0] / norm, k[1] / norm, k[2] / norm};
        std::vector<std::vector<double>> A(3, std::vector<double>(3));
        for (int a = 0; a < 3; ++a)
            for (int b = 0; b < 3; ++b)
                A[a][b] = (a == b)
                              ? reference_.c11 * n[a] * n[a] + reference_.c44 * (1 - n[a] * n[a])
                              : (reference_.c12 + reference_.c44) * n[a] * n[b];
        auto x = linear(A, {1, 0, 0}), y = linear(A, {0, 1, 0}), z = linear(A, {0, 0, 1});
        green_[0][i] = x[0];
        green_[1][i] = y[1];
        green_[2][i] = z[2];
        green_[3][i] = z[1];
        green_[4][i] = z[0];
        green_[5][i] = y[0];
    }
    reset();
}
void Elasticity::reset() {
    resize(periodic_, fft_.real_size);
    homogeneous_.fill(0);
    for (auto &x : uhat_)
        x.assign(fft_.fourier_size, 0);
}
void Elasticity::material(const std::vector<Real> &h, std::vector<Cubic> &c, Fields &eigen,
                          Cubic &average) const {
    bool valid = h.size() == phases_.size();
    for (auto &f : h)
        valid = valid && f.size() == fft_.real_size;
    require(valid, "Phase fraction array count/size mismatch", fft_.comm);
    c.assign(fft_.real_size, Cubic{0, 0, 0});
    resize(eigen, fft_.real_size);
    average = {0, 0, 0};
    for (size_t i = 0; i < fft_.real_size; ++i) {
        double total = 0;
        Tensor eig{};
        for (size_t p = 0; p < h.size(); ++p) {
            double w = h[p][i];
            valid = valid && std::isfinite(w) && w >= -1e-12 && w <= 1 + 1e-12;
            total += w;
            const auto &cp = phases_[p].stiffness;
            c[i].c11 += w * cp.c11;
            c[i].c12 += w * cp.c12;
            c[i].c44 += w * cp.c44;
            eig = eig + w * phases_[p].eigenstrain;
        }
        valid = valid && std::abs(total - 1) < 1e-10;
        put(eigen, i, eig);
        average.c11 += c[i].c11;
        average.c12 += c[i].c12;
        average.c44 += c[i].c44;
    }
    require(valid, "Fractions must be finite, in [0,1], and sum to 1. No clipping is performed.",
            fft_.comm);
    average.c11 = sum(average.c11, fft_.comm) / fft_.grid.points();
    average.c12 = sum(average.c12, fft_.comm) / fft_.grid.points();
    average.c44 = sum(average.c44, fft_.comm) / fft_.grid.points();
}
Tensor Elasticity::balance(const std::vector<Cubic> &c, const Fields &eigen, const Fields &per,
                           const Cubic &avg, const Loading &load) const {
    Tensor rhs{};
    for (size_t i = 0; i < c.size(); ++i)
        rhs = rhs + c[i].stress(at(eigen, i) - at(per, i));
    for (int a = 0; a < 6; ++a)
        rhs[a] = sum(rhs[a], fft_.comm) / fft_.grid.points() + load.stress[a];
    Tensor result = load.strain;
    std::vector<int> unknown;
    for (int a = 0; a < 6; ++a)
        if (!load.fixed_strain[a]) {
            unknown.push_back(a);
            result[a] = 0;
        }
    Tensor known = avg.stress(result);
    std::vector<std::vector<double>> A(unknown.size(), std::vector<double>(unknown.size()));
    std::vector<double> b(unknown.size());
    for (size_t j = 0; j < unknown.size(); ++j) {
        b[j] = rhs[unknown[j]] - known[unknown[j]];
        for (size_t k = 0; k < unknown.size(); ++k) {
            Tensor unit{};
            unit[unknown[k]] = 1;
            A[j][k] = avg.stress(unit)[unknown[j]];
        }
    }
    auto x = linear(A, b);
    for (size_t j = 0; j < x.size(); ++j)
        result[unknown[j]] = x[j];
    return result;
}
void Elasticity::project(const Fields &tau, FourierVector &u) {
    std::array<Fourier, 6> t;
    for (int a = 0; a < 6; ++a)
        fft_.forward(tau[a], t[a]);
    for (auto &x : u)
        x.assign(fft_.fourier_size, 0);
    for (size_t i = 0; i < fft_.fourier_size; ++i) {
        auto k = fft_.wave(i);
        double norm = std::sqrt(k[0] * k[0] + k[1] * k[1] + k[2] * k[2]);
        if (norm == 0)
            continue;
        double n[3] = {k[0] / norm, k[1] / norm, k[2] / norm};
        Complex q[3] = {t[0][i] * n[0] + t[5][i] * n[1] + t[4][i] * n[2],
                        t[5][i] * n[0] + t[1][i] * n[1] + t[3][i] * n[2],
                        t[4][i] * n[0] + t[3][i] * n[1] + t[2][i] * n[2]};
        u[0][i] = Complex(0, -1) *
                  (green_[0][i] * q[0] + green_[5][i] * q[1] + green_[4][i] * q[2]) / norm;
        u[1][i] = Complex(0, -1) *
                  (green_[5][i] * q[0] + green_[1][i] * q[1] + green_[3][i] * q[2]) / norm;
        u[2][i] = Complex(0, -1) *
                  (green_[4][i] * q[0] + green_[3][i] * q[1] + green_[2][i] * q[2]) / norm;
    }
}
void Elasticity::reconstruct(const FourierVector &u, Fields &strain) {
    const int row[6] = {0, 1, 2, 1, 0, 0}, col[6] = {0, 1, 2, 2, 2, 1};
    Fourier f(fft_.fourier_size);
    for (int a = 0; a < 6; ++a) {
        for (size_t i = 0; i < f.size(); ++i) {
            auto k = fft_.wave(i);
            f[i] = Complex(0, .5) * (k[col[a]] * u[row[a]][i] + k[row[a]] * u[col[a]][i]);
        }
        fft_.inverse(f, strain[a]);
    }
}
void Elasticity::evaluate(const std::vector<Cubic> &c, const Fields &eigen, Fields &strain,
                          Fields &stress, Real &energy) const {
    resize(strain, c.size());
    resize(stress, c.size());
    energy.resize(c.size());
    for (size_t i = 0; i < c.size(); ++i) {
        Tensor total = homogeneous_ + at(periodic_, i), e = total - at(eigen, i),
               s = c[i].stress(e);
        put(strain, i, total);
        put(stress, i, s);
        energy[i] = .5 * dot(e, s);
    }
}
double Elasticity::equilibrium(const Fields &stress, double scale) {
    std::array<Fourier, 6> s;
    for (int a = 0; a < 6; ++a)
        fft_.forward(stress[a], s[a]);
    double total = 0;
    for (size_t i = 0; i < fft_.fourier_size; ++i) {
        auto k = fft_.wave(i);
        Complex d[3] = {s[0][i] * k[0] + s[5][i] * k[1] + s[4][i] * k[2],
                        s[5][i] * k[0] + s[1][i] * k[1] + s[3][i] * k[2],
                        s[4][i] * k[0] + s[3][i] * k[1] + s[2][i] * k[2]};
        size_t z = i % size_t(fft_.grid.nz / 2 + 1);
        double weight =
            (z == 0 || (fft_.grid.nz % 2 == 0 && z == size_t(fft_.grid.nz / 2))) ? 1 : 2;
        for (auto v : d)
            total += weight * std::norm(v);
    }
    double L = std::max(
        {fft_.grid.nx * fft_.grid.dx, fft_.grid.ny * fft_.grid.dy, fft_.grid.nz * fft_.grid.dz});
    return std::sqrt(sum(total, fft_.comm)) / fft_.grid.points() / (2 * pi / L * scale);
}
ElasticResult Elasticity::solve(const std::vector<Real> &h, const Loading &load) {
    bool valid = true;
    for (int a = 0; a < 6; ++a)
        valid = valid && std::isfinite(load.stress[a]) && std::isfinite(load.strain[a]);
    require(valid, "Nonfinite loading", fft_.comm);
    std::vector<Cubic> c;
    Fields eigen;
    Cubic avg;
    material(h, c, eigen, avg);
    double mod = 0, strain_scale = 0, stress_scale = 0;
    for (auto &p : phases_) {
        mod = std::max({mod, p.stiffness.c11, std::abs(p.stiffness.c12), 2 * p.stiffness.c44});
        for (double e : p.eigenstrain)
            strain_scale = std::max(strain_scale, std::abs(e));
    }
    for (int a = 0; a < 6; ++a) {
        stress_scale = std::max(stress_scale, std::abs(load.stress[a]));
        strain_scale = std::max(strain_scale, std::abs(load.strain[a]));
    }
    strain_scale = std::max({strain_scale, stress_scale / mod, 1e-15});
    stress_scale = std::max(stress_scale, mod * strain_scale);
    homogeneous_ = balance(c, eigen, periodic_, avg, load);
    ElasticResult result;
    Fields tau;
    resize(tau, c.size());
    const double L = std::max(
        {fft_.grid.nx * fft_.grid.dx, fft_.grid.ny * fft_.grid.dy, fft_.grid.nz * fft_.grid.dz});
    for (int iteration = 1; iteration <= controls_.max_iterations; ++iteration) {
        for (size_t i = 0; i < c.size(); ++i) {
            Tensor per = at(periodic_, i),
                   source = c[i].stress(at(eigen, i) - homogeneous_ - per) + reference_.stress(per);
            put(tau, i, source);
        }
        FourierVector next;
        project(tau, next);
        VectorFields delta;
        double l2 = 0, linf = 0;
        for (int a = 0; a < 3; ++a) {
            Fourier d(next[a].size());
            for (size_t i = 0; i < d.size(); ++i)
                d[i] = next[a][i] - uhat_[a][i];
            fft_.inverse(d, delta[a]);
            for (double v : delta[a]) {
                l2 += v * v;
                linf = std::max(linf, std::abs(v));
            }
        }
        Tensor old = homogeneous_;
        uhat_ = std::move(next);
        reconstruct(uhat_, periodic_);
        homogeneous_ = balance(c, eigen, periodic_, avg, load);
        double change =
            controls_.use_max_norm
                ? maximum(linf, fft_.comm) / (L * strain_scale)
                : std::sqrt(sum(l2, fft_.comm) / fft_.grid.points()) / (L * strain_scale);
        for (int a = 0; a < 6; ++a)
            change = std::max(change, std::abs(homogeneous_[a] - old[a]) / strain_scale);
        evaluate(c, eigen, result.strain, result.stress, result.energy);
        Status &st = result.status;
        st.iterations = iteration;
        st.change = change;
        st.mean_strain = homogeneous_;
        st.mean_stress_error = 0;
        for (int a = 0; a < 6; ++a) {
            st.mean_stress[a] = fft_.mean(result.stress[a]);
            if (!load.fixed_strain[a])
                st.mean_stress_error =
                    std::max(st.mean_stress_error,
                             std::abs(st.mean_stress[a] - load.stress[a]) / stress_scale);
        }
        st.equilibrium = equilibrium(result.stress, stress_scale);
        bool finite = std::isfinite(change) && std::isfinite(st.equilibrium) &&
                      std::isfinite(st.mean_stress_error);
        for (auto &f : result.stress)
            for (double v : f)
                finite = finite && std::isfinite(v);
        for (double v : result.energy)
            finite = finite && std::isfinite(v);
        require(finite, "Nonfinite elasticity iterate; no force accepted", fft_.comm);
        if (change <= controls_.change_tolerance &&
            st.equilibrium <= controls_.equilibrium_tolerance &&
            st.mean_stress_error <= controls_.mean_stress_tolerance) {
            st.converged = true;
            st.reason = "converged";
            break;
        }
        st.reason = "iteration_limit";
    }
    for (int a = 0; a < 3; ++a)
        fft_.inverse(uhat_[a], result.displacement[a]);
    result.status.mean_energy = fft_.mean(result.energy);
    result.status.mean_potential = result.status.mean_energy;
    Tensor work{};
    for (int a = 0; a < 6; ++a)
        if (!load.fixed_strain[a])
            work[a] = load.stress[a];
    result.status.mean_potential -= dot(work, homogeneous_);
    // Refuse to expose forces from a failed equilibrium solve.
    if (result.status.converged) {
        result.fraction_derivative.assign(phases_.size(), Real(c.size()));
        for (size_t i = 0; i < c.size(); ++i) {
            Tensor e = at(result.strain, i) - at(eigen, i), s = at(result.stress, i);
            for (size_t p = 0; p < phases_.size(); ++p)
                result.fraction_derivative[p][i] =
                    .5 * dot(e, phases_[p].stiffness.stress(e)) - dot(s, phases_[p].eigenstrain);
        }
    }
    return result;
}
} // namespace microsim
