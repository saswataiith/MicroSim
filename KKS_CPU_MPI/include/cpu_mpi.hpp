#pragma once
#include <array>
#include <complex>
#include <cstdint>
#include <fftw3-mpi.h>
#include <map>
#include <mpi.h>
#include <string>
#include <vector>

namespace microsim {
using Real = std::vector<double>;
using Complex = std::complex<double>;
using Fourier = std::vector<Complex>;
using Tensor = std::array<double, 6>; // xx, yy, zz, yz, xz, xy; TENSOR shear
using Fields = std::array<Real, 6>;
using VectorFields = std::array<Real, 3>;
using FourierVector = std::array<Fourier, 3>;
constexpr double pi = 3.1415926535897932384626433832795;

void require(bool valid, const std::string &message, MPI_Comm comm);
double sum(double local, MPI_Comm comm);
double maximum(double local, MPI_Comm comm);
double dot(const Tensor &a, const Tensor &b);
Tensor operator+(Tensor a, const Tensor &b);
Tensor operator-(Tensor a, const Tensor &b);
Tensor operator*(double a, Tensor b);
struct Cubic {
    double c11 = 300, c12 = 100, c44 = 80;
    Tensor stress(const Tensor &strain) const;
    void validate() const;
};
struct Phase {
    Cubic stiffness;
    Tensor eigenstrain{};
};
struct Grid {
    ptrdiff_t nx = 16, ny = 16, nz = 16;
    double dx = 1, dy = 1, dz = 1;
    double points() const {
        return double(nx) * double(ny) * double(nz);
    }
    double volume() const {
        return points() * dx * dy * dz;
    }
};

// Owns distributed reusable plans. Real fields are compact x slabs; Fourier
// fields are transposed y slabs, ordered [local_y][global_x][kz>=0].
class FFT {
    double *real_ = nullptr;
    fftw_complex *complex_ = nullptr;
    fftw_plan forward_ = nullptr, backward_ = nullptr;

  public:
    Grid grid;
    MPI_Comm comm;
    int rank = 0, ranks = 1;
    ptrdiff_t local_nx = 0, x_start = 0, local_ny = 0, y_start = 0;
    size_t real_size = 0, fourier_size = 0;
    FFT(Grid g, MPI_Comm communicator, bool measure = false);
    ~FFT();
    FFT(const FFT &) = delete;
    FFT &operator=(const FFT &) = delete;
    void forward(const Real &input, Fourier &output);
    void inverse(const Fourier &input, Real &output);
    std::array<double, 3> wave(size_t index, bool derivative = true) const;
    std::array<ptrdiff_t, 3> coordinate(size_t index) const;
    double mean(const Real &field) const;
    double rms(const Real &field) const;
};

struct Loading {
    Tensor stress{}, strain{};
    // true: prescribed mean strain; false: prescribed mean stress
    std::array<bool, 6> fixed_strain{};
};
struct Controls {
    int max_iterations = 1000;
    double change_tolerance = 1e-10, equilibrium_tolerance = 1e-8;
    double mean_stress_tolerance = 1e-10;
    bool use_max_norm = false;
};
struct Status {
    bool converged = false;
    int iterations = 0;
    std::string reason = "not_run";
    double change = 0, equilibrium = 0, mean_stress_error = 0;
    Tensor mean_strain{}, mean_stress{};
    double mean_energy = 0, mean_potential = 0;
};
struct ElasticResult {
    Fields strain, stress; // TOTAL strain and constitutive stress
    VectorFields displacement;
    Real energy;
    // dF/dh_p at fixed total strain, before any phase interpolation derivative.
    // Only differences of these partials matter for fractions summing to one.
    std::vector<Real> fraction_derivative;
    Status status;
};

class Elasticity {
    FFT &fft_;
    std::vector<Phase> phases_;
    Controls controls_;
    Cubic reference_;
    Fields periodic_;
    Tensor homogeneous_{};
    FourierVector uhat_;
    std::array<Real, 6> green_; // inverse acoustic tensor, precomputed on the reduced grid
    void material(const std::vector<Real> &h, std::vector<Cubic> &c, Fields &eigen,
                  Cubic &average) const;
    Tensor balance(const std::vector<Cubic> &c, const Fields &eigen, const Fields &periodic,
                   const Cubic &average, const Loading &loading) const;
    void reconstruct(const FourierVector &displacement, Fields &strain);
    void evaluate(const std::vector<Cubic> &c, const Fields &eigen, Fields &strain, Fields &stress,
                  Real &energy) const;
    void project(const Fields &tau, FourierVector &displacement);
    double equilibrium(const Fields &stress, double stress_scale);

  public:
    Elasticity(FFT &fft, std::vector<Phase> phases, Controls controls = {});
    ElasticResult solve(const std::vector<Real> &fractions, const Loading &loading = {});
    void reset();
    const std::vector<Phase> &phases() const {
        return phases_;
    }
};

// Original KKS parabolas, quintic interpolation and Allen-Cahn/Cahn-Hilliard
// updates. No hidden unit conversion: a_matrix/a_precip are energy densities.
struct PhaseFieldControls {
    int variants = 1, mobility_method = 0;
    double dt = .01, kinetic = 1, kappa = .2, omega = 1, theta = 10, theta2 = 0;
    double a_matrix = 1, a_precip = 1, c_matrix = .15, c_precip = .23;
    double diffusivity = .01;
    bool evolve_composition = true;
    bool elasticity = true;
};
struct State {
    int step = 0;
    double time = 0;
    std::uint64_t model_hash = 0;
    Real composition;
    std::vector<Real> phi;
};
struct Diagnostics {
    double composition_mean = 0, energy = 0, potential = 0;
    Status elastic;
};
double interpolation(double phi);
double interpolation_derivative(double phi);
class PhaseField {
    FFT &fft_;
    Elasticity &elastic_;
    PhaseFieldControls p_;
    Loading loading_;
    void fractions(const State &state, std::vector<Real> &h) const;

  public:
    PhaseField(FFT &fft, Elasticity &elastic, PhaseFieldControls controls, Loading loading = {});
    Diagnostics evaluate(const State &state, std::vector<Real> *driving_force = nullptr,
                         Real *chemical_potential = nullptr, ElasticResult *result = nullptr);
    Diagnostics advance(State &state);
};

struct Config {
    std::map<std::string, std::string> values;
    static Config read(const std::string &path);
    std::string text(const std::string &key, const std::string &fallback) const;
    double number(const std::string &key, double fallback) const;
    int integer(const std::string &key, int fallback) const;
    Tensor tensor(const std::string &key, Tensor fallback = {}) const;
};
void write_fields(const std::string &path, FFT &fft,
                  const std::map<std::string, const Real *> &fields,
                  const std::map<std::string, double> &attributes = {});
Real read_field(const std::string &path, const std::string &name, FFT &fft);
void write_state(const std::string &path, FFT &fft, const State &state);
State read_state(const std::string &path, FFT &fft, int variants);
} // namespace microsim
