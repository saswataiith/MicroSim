#include "cpu_mpi.hpp"
#include "original_kernel_reference.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
using namespace microsim;
int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    fftw_mpi_init();
    try {
        {
            FFT fft({7, 5, 3, 1, 1, 1}, MPI_COMM_WORLD);
            double worst = 0, mu_worst = 0;
            for (int variants : {1, 4}) {
                PhaseFieldControls p;
                p.variants = variants;
                p.theta2 = variants == 4 ? 3 : 0;
                p.a_matrix = 1.3;
                p.a_precip = .7;
                p.theta = 10;
                std::vector<Phase> phases(variants + 1);
                phases[0] = {{300, 100, 80}, {}};
                for (int j = 1; j <= variants; ++j)
                    phases[j] = {{450, 150, 120}, {-.005, -.008, -.008, 0, 0, 0}};
                Controls ctl;
                ctl.change_tolerance = 1e-12;
                ctl.equilibrium_tolerance = 1e-10;
                Elasticity elastic(fft, phases, ctl);
                Loading load;
                load.stress = {.1, -.02, .03, .04, .05, -.06};
                PhaseField code(fft, elastic, p, load);
                State s;
                s.composition.resize(fft.real_size);
                s.phi.assign(variants, Real(fft.real_size));
                for (size_t i = 0; i < fft.real_size; ++i) {
                    auto c = fft.coordinate(i);
                    s.composition[i] = .19 + .003 * std::sin(2 * pi * c[1] / fft.grid.ny);
                    for (int a = 0; a < variants; ++a)
                        s.phi[a][i] = .15 + .04 * std::cos(2 * pi * (c[0] + a) / fft.grid.nx);
                }
                std::vector<Real> force;
                Real mu;
                ElasticResult result;
                code.evaluate(s, &force, &mu, &result);
                for (size_t i = 0; i < fft.real_size; ++i) {
                    using Z = legacy_cuda::cufftDoubleComplex;
                    Z phi[4]{}, df[4]{}, comp{s.composition[i], 0}, ca{}, cb{}, eng{}, chemical{};
                    double h = 0, dh[4]{}, well = 0, vol = 0, ceq[2] = {p.c_matrix, p.c_precip},
                           f0[2] = {p.a_matrix, p.a_precip};
                    for (int a = 0; a < variants; ++a)
                        phi[a].x = s.phi[a][i];
                    if (variants == 1)
                        legacy_cuda::Calc_hsum_dhdphi_sv(&h, dh, &well, &vol, phi, phi + 1, df,
                                                         p.omega, 0, 0, 0, 1, 1);
                    else {
                        legacy_cuda::Calc_hphisum_dhdphi(&h, dh, dh + 1, dh + 2, dh + 3, &vol, phi,
                                                         phi + 1, phi + 1, phi + 2, phi + 3, 0, 1,
                                                         1);
                        legacy_cuda::Calc_dgdphi2(df, df + 1, df + 1, df + 2, df + 3, phi, phi + 1,
                                                  phi + 2, phi + 3, p.omega, p.theta, p.theta2, 1,
                                                  1);
                    }
                    legacy_cuda::Calc_phase_comp(&ca, &cb, &comp, &comp + 1, &h, ceq, f0, 1, 1, 1);
                    legacy_cuda::Calc_phase_engs(&eng, &eng + 1, &ca, &cb, &h, &well, ceq, f0,
                                                 p.a_matrix, p.a_precip, 1, 0, 0, 0, 0, 1, 1, 1);
                    legacy_cuda::Calc_mu(&chemical, &chemical + 1, &ca, ceq, p.a_matrix, p.a_precip,
                                         0, 1, 1, 1);
                    mu_worst = std::max(mu_worst, std::abs(chemical.x - mu[i]));
                    if (variants == 1)
                        legacy_cuda::Calc_dfchdphi_sv(df, df + 1, &chemical, &ca, &cb, &eng, dh, 1,
                                                      1, 1, 1);
                    else
                        legacy_cuda::Calc_dfchdphi(df, df + 1, df + 1, df + 2, df + 3, &chemical,
                                                   &ca, &cb, &eng, dh, dh + 1, dh + 2, dh + 3, 1, 1,
                                                   1, 1);
                    Z strain[6];
                    double hom[6];
                    for (int a = 0; a < 6; ++a) {
                        hom[a] = result.status.mean_strain[a];
                        strain[a] = {result.strain[a][i] - hom[a], 0};
                    }
                    double energy;
                    for (int a = 0; a < variants; ++a) {
                        legacy_cuda::Compute_dfeldphi_inhomsv(
                            df + a, df + a + 1, strain, strain + 1, strain + 2, strain + 3,
                            strain + 4, strain + 5, &energy, &h, dh + a, hom, 375, 125, 100, 75, 25,
                            20, -.005, 1.6, 0, 0, 0, 1, 1, 1);
                        worst = std::max(worst, std::abs(df[a].x - force[a][i]));
                    }
                }
            }
            worst = maximum(worst, fft.comm);
            mu_worst = maximum(mu_worst, fft.comm);
            require(worst < 2e-13 && mu_worst < 2e-13, "Original CUDA local-kernel parity failed",
                    fft.comm);
            if (!rank)
                std::cout << "PASS original CUDA local kernels: max force difference=" << worst
                          << " max mu difference=" << mu_worst << " (1 and 4 variants)\n";
        }
        fftw_mpi_cleanup();
        MPI_Finalize();
        return 0;
    } catch (const std::exception &e) {
        std::cerr << e.what() << "\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
}
