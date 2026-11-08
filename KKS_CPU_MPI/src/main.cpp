#include "cpu_mpi.hpp"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>

using namespace microsim;
static bool boolean(const Config &c, const std::string &key, bool fallback) {
    int v = c.integer(key, fallback);
    if (v != 0 && v != 1)
        throw std::runtime_error("Expected 0 or 1: " + key);
    return v;
}
static void save_result(const std::string &path, FFT &fft, const ElasticResult &r,
                        const std::vector<Real> &h) {
    std::map<std::string, const Real *> fields{{"elastic_energy", &r.energy}};
    const char *component[6] = {"xx", "yy", "zz", "yz", "xz", "xy"};
    for (int a = 0; a < 6; ++a) {
        fields["strain_" + std::string(component[a])] = &r.strain[a];
        fields["stress_" + std::string(component[a])] = &r.stress[a];
    }
    for (int a = 0; a < 3; ++a)
        fields["u" + std::to_string(a)] = &r.displacement[a];
    for (size_t p = 0; p < h.size(); ++p) {
        fields["h" + std::to_string(p)] = &h[p];
        if (!r.fraction_derivative.empty())
            fields["dFdh" + std::to_string(p)] = &r.fraction_derivative[p];
    }
    write_fields(path, fft, fields,
                 {{"converged", r.status.converged ? 1. : 0.},
                  {"iterations", double(r.status.iterations)},
                  {"change", r.status.change},
                  {"equilibrium", r.status.equilibrium},
                  {"mean_stress_error", r.status.mean_stress_error},
                  {"mean_energy", r.status.mean_energy},
                  {"mean_potential", r.status.mean_potential}});
}
static State initial_state(const Config &c, FFT &f, int variants) {
    auto restart = c.text("restart", "");
    if (!restart.empty())
        return read_state(restart, f, variants);
    State s;
    s.composition.assign(f.real_size, c.number("composition", .19));
    s.phi.assign(variants, Real(f.real_size));
    double min_length = std::min(f.grid.nx * f.grid.dx, f.grid.ny * f.grid.dy);
    if (f.grid.nz > 1)
        min_length = std::min(min_length, f.grid.nz * f.grid.dz);
    double radius = c.number("radius", .13 * min_length), width = c.number("interface_width", 1.);
    std::string init = c.text("initial", "sphere");
    double value = c.number("phi_uniform", .3), background = c.number("phi_background", 0);
    require(width > 0 && radius > 0 && value >= 0 && value <= 1 && background >= 0 &&
                background < 1,
            "Invalid initial geometry", f.comm);
    require(init == "sphere" || init == "sine" || init == "uniform", "Unknown initial geometry",
            f.comm);
    for (size_t i = 0; i < f.real_size; ++i) {
        auto xyz = f.coordinate(i);
        double pos[3] = {(xyz[0] + .5) * f.grid.dx, (xyz[1] + .5) * f.grid.dy,
                         (xyz[2] + .5) * f.grid.dz};
        double length[3] = {f.grid.nx * f.grid.dx, f.grid.ny * f.grid.dy, f.grid.nz * f.grid.dz};
        for (int p = 0; p < variants; ++p) {
            double v = 0;
            if (init == "uniform")
                v = value / double(variants);
            if (init == "sine")
                v = (.3 + .08 * std::sin(2 * pi * (xyz[0] + .5) / f.grid.nx) *
                              std::cos(2 * pi * (xyz[1] + .5) / f.grid.ny)) /
                    double(variants);
            if (init == "sphere") {
                double d2 = 0;
                for (int a = 0; a < 3; ++a) {
                    if (a == 2 && f.grid.nz == 1)
                        continue;
                    double center = length[a] * .5;
                    if (variants > 1) {
                        if (a == 0)
                            center = length[a] * (.25 + .5 * double(p % 2));
                        if (a == 1)
                            center = length[a] * (.25 + .5 * double((p / 2) % 2));
                    }
                    double d = std::abs(pos[a] - center);
                    d = std::min(d, length[a] - d);
                    d2 += d * d;
                }
                v = .5 * (1 - std::tanh((std::sqrt(d2) - radius) / width));
            }
            s.phi[p][i] = init == "sphere" ? background + (1 - background) * v : v;
        }
    }
    auto input = c.text("initial_file", "");
    if (!input.empty()) {
        s.composition = read_field(input, "composition", f);
        for (int p = 0; p < variants; ++p)
            s.phi[p] = read_field(input, "phi" + std::to_string(p), f);
    }
    return s;
}
int main(int argc, char **argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    try {
        if (argc != 2) {
            if (!rank)
                std::cerr << "Usage: microsim_cpu CONFIG\n";
            MPI_Finalize();
            return 2;
        }
        Config c = Config::read(argv[1]);
        int threads = c.integer("fft_threads", 1);
        require(threads > 0 && provided >= MPI_THREAD_FUNNELED,
                "MPI thread support/FFT thread count invalid", MPI_COMM_WORLD);
        require(fftw_init_threads() != 0, "FFTW thread initialization failed", MPI_COMM_WORLD);
        fftw_mpi_init();
        fftw_plan_with_nthreads(threads);
        {
            Grid grid;
            grid.nx = c.integer("nx", 16);
            grid.ny = c.integer("ny", 16);
            grid.nz = c.integer("nz", 16);
            grid.dx = c.number("dx", 1);
            grid.dy = c.number("dy", 1);
            grid.dz = c.number("dz", 1);
            FFT fft(grid, MPI_COMM_WORLD, boolean(c, "fft_measure", false));
            int variants = c.integer("variants", 1),
                phase_count = c.integer("phases", variants + 1);
            require(variants >= 1 && variants <= 32 && phase_count >= 1 && phase_count <= 64,
                    "Invalid variant/phase count", fft.comm);
            Controls controls;
            controls.max_iterations = c.integer("elastic_max_iterations", 1000);
            controls.change_tolerance = c.number("elastic_change_tolerance", 1e-10);
            controls.equilibrium_tolerance = c.number("elastic_equilibrium_tolerance", 1e-8);
            controls.mean_stress_tolerance = c.number("elastic_mean_stress_tolerance", 1e-10);
            auto norm = c.text("elastic_norm", "rms");
            require(norm == "rms" || norm == "max", "elastic_norm must be rms or max", fft.comm);
            controls.use_max_norm = norm == "max";
            Loading loading;
            loading.stress = c.tensor("applied_stress");
            loading.strain = c.tensor("applied_strain");
            auto mask = c.tensor("fixed_strain");
            for (int a = 0; a < 6; ++a) {
                require(mask[a] == 0 || mask[a] == 1, "fixed_strain components must be 0 or 1",
                        fft.comm);
                loading.fixed_strain[a] = mask[a] == 1;
            }
            std::set<std::string> known = {"mode",
                                           "nx",
                                           "ny",
                                           "nz",
                                           "dx",
                                           "dy",
                                           "dz",
                                           "fft_threads",
                                           "fft_measure",
                                           "variants",
                                           "phases",
                                           "elastic_max_iterations",
                                           "elastic_change_tolerance",
                                           "elastic_equilibrium_tolerance",
                                           "elastic_mean_stress_tolerance",
                                           "elastic_norm",
                                           "applied_stress",
                                           "applied_strain",
                                           "fixed_strain",
                                           "output",
                                           "steps",
                                           "output_every",
                                           "dt",
                                           "kinetic",
                                           "kappa",
                                           "omega",
                                           "theta",
                                           "theta2",
                                           "a_matrix",
                                           "a_precip",
                                           "c_matrix",
                                           "c_precip",
                                           "diffusivity",
                                           "mobility_method",
                                           "evolve_composition",
                                           "elasticity",
                                           "composition",
                                           "initial",
                                           "initial_file",
                                           "restart",
                                           "fraction_file",
                                           "radius",
                                           "interface_width",
                                           "phi_uniform",
                                           "phi_background",
                                           "allow_restart_parameter_change"};
            std::vector<Phase> phases(phase_count);
            for (int p = 0; p < phase_count; ++p) {
                std::string prefix = "phase" + std::to_string(p) + ".";
                for (auto key : {"c11", "c12", "c44", "eigenstrain"})
                    known.insert(prefix + key);
                double factor = p ? 1.5 : 1.;
                phases[p].stiffness = {c.number(prefix + "c11", 300 * factor),
                                       c.number(prefix + "c12", 100 * factor),
                                       c.number(prefix + "c44", 80 * factor)};
                phases[p].eigenstrain = c.tensor(
                    prefix + "eigenstrain", p ? Tensor{-.005, -.005, -.005, 0, 0, 0} : Tensor{});
            }
            for (auto &[key, value] : c.values)
                if (!known.count(key))
                    throw std::runtime_error("Unknown configuration key: " + key);
            Elasticity elastic(fft, phases, controls);
            std::string out = c.text("output", "run");
            if (!rank) {
                std::filesystem::create_directories(out);
                std::ofstream cfg(out + "/input.cfg");
                for (auto &[k, v] : c.values)
                    cfg << k << " = " << v << "\n";
            }
            MPI_Barrier(fft.comm);
            double started = MPI_Wtime();
            std::string mode = c.text("mode", "phasefield");
            require(mode == "phasefield" || mode == "elasticity",
                    "mode must be phasefield or elasticity", fft.comm);
            if (mode == "elasticity") {
                std::vector<Real> h;
                auto input = c.text("fraction_file", "");
                if (!input.empty()) {
                    for (int p = 0; p < phase_count; ++p)
                        h.push_back(read_field(input, "h" + std::to_string(p), fft));
                } else {
                    require(phase_count == variants + 1,
                            "Generated fractions require phases=variants+1", fft.comm);
                    auto state = initial_state(c, fft, variants);
                    h.assign(phase_count, Real(fft.real_size));
                    std::fill(h[0].begin(), h[0].end(), 1);
                    for (int p = 0; p < variants; ++p)
                        for (size_t i = 0; i < fft.real_size; ++i) {
                            h[p + 1][i] = interpolation(state.phi[p][i]);
                            h[0][i] -= h[p + 1][i];
                        }
                }
                auto r = elastic.solve(h, loading);
                save_result(out + "/fields_final.h5", fft, r, h);
                require(r.status.converged,
                        "Elasticity did not converge; saved fields have converged=0 and contain no "
                        "driving forces",
                        fft.comm);
                if (!rank)
                    std::cout << "converged iterations=" << r.status.iterations
                              << " equilibrium=" << r.status.equilibrium
                              << " mean_stress_error=" << r.status.mean_stress_error << "\n";
            } else {
                PhaseFieldControls p;
                p.variants = variants;
                p.dt = c.number("dt", .01);
                p.kinetic = c.number("kinetic", 1);
                p.kappa = c.number("kappa", .2);
                p.omega = c.number("omega", 1);
                p.theta = c.number("theta", 10);
                p.theta2 = c.number("theta2", 0);
                p.a_matrix = c.number("a_matrix", 1);
                p.a_precip = c.number("a_precip", 1);
                p.c_matrix = c.number("c_matrix", .15);
                p.c_precip = c.number("c_precip", .23);
                p.diffusivity = c.number("diffusivity", .01);
                p.mobility_method = c.integer("mobility_method", 0);
                p.evolve_composition = boolean(c, "evolve_composition", true);
                p.elasticity = boolean(c, "elasticity", true);
                PhaseField program(fft, elastic, p, loading);
                State state = initial_state(c, fft, variants);
                int steps = c.integer("steps", 10), every = c.integer("output_every", 10);
                require(steps >= 0 && every > 0 &&
                            steps <= std::numeric_limits<int>::max() - state.step,
                        "Invalid/overflowing steps or output interval", fft.comm);
                std::ostringstream model;
                model << std::setprecision(17) << "microsim_cpu_schema1\n"
                      << grid.nx << " " << grid.ny << " " << grid.nz << " " << grid.dx << " "
                      << grid.dy << " " << grid.dz << "\n";
                for (auto &phase : phases) {
                    model << phase.stiffness.c11 << " " << phase.stiffness.c12 << " "
                          << phase.stiffness.c44;
                    for (double x : phase.eigenstrain)
                        model << " " << x;
                    model << "\n";
                }
                for (int a = 0; a < 6; ++a)
                    model << loading.stress[a] << " " << loading.strain[a] << " "
                          << loading.fixed_strain[a] << "\n";
                model << p.variants << " " << p.mobility_method << " " << p.dt << " " << p.kinetic
                      << " " << p.kappa << " " << p.omega << " " << p.theta << " " << p.theta2
                      << " " << p.a_matrix << " " << p.a_precip << " " << p.c_matrix << " "
                      << p.c_precip << " " << p.diffusivity << " " << p.evolve_composition << " "
                      << p.elasticity << "\n";
                model << controls.max_iterations << " " << controls.change_tolerance << " "
                      << controls.equilibrium_tolerance << " " << controls.mean_stress_tolerance
                      << " " << controls.use_max_norm << "\n";
                std::uint64_t fingerprint = 14695981039346656037ULL;
                for (unsigned char ch : model.str()) {
                    fingerprint ^= ch;
                    fingerprint *= 1099511628211ULL;
                }
                require(c.text("restart", "").empty() || state.model_hash == fingerprint ||
                            boolean(c, "allow_restart_parameter_change", false),
                        "Restart physical/numerical parameters changed; set "
                        "allow_restart_parameter_change=1 only for an intentional new experiment",
                        fft.comm);
                state.model_hash = fingerprint;
                if (!rank) {
                    std::ofstream effective(out + "/model.txt");
                    effective << model.str() << "model_hash " << fingerprint << "\n";
                }
                std::ofstream log;
                if (!rank) {
                    log.open(out + "/diagnostics.csv");
                    log << "step,time,composition_mean,energy_density,potential_density,elastic_"
                           "iterations,elastic_change,equilibrium,mean_stress_error\n";
                    log << std::setprecision(17);
                }
                auto record = [&](const Diagnostics &d) {
                    if (!rank) {
                        log << state.step << "," << state.time << "," << d.composition_mean << ","
                            << d.energy << "," << d.potential << "," << d.elastic.iterations << ","
                            << d.elastic.change << "," << d.elastic.equilibrium << ","
                            << d.elastic.mean_stress_error << "\n";
                        log.flush();
                    }
                };
                for (int j = 0; j < steps; ++j) {
                    int old_step = state.step;
                    double old_time = state.time;
                    Diagnostics d = program.advance(state);
                    if (j % every == 0) { // d belongs to the old state
                        if (!rank)
                            log << old_step << "," << std::setprecision(17) << old_time << ","
                                << d.composition_mean << "," << d.energy << "," << d.potential
                                << "," << d.elastic.iterations << "," << d.elastic.change << ","
                                << d.elastic.equilibrium << "," << d.elastic.mean_stress_error
                                << "\n";
                    }
                    if (state.step % every == 0)
                        write_state(out + "/checkpoint_" + std::to_string(state.step) + ".h5", fft,
                                    state);
                }
                ElasticResult result;
                auto d = program.evaluate(state, nullptr, nullptr, &result);
                record(d);
                write_state(out + "/state_final.h5", fft, state);
                std::vector<Real> h(variants + 1, Real(fft.real_size));
                std::fill(h[0].begin(), h[0].end(), 1);
                for (int pidx = 0; pidx < variants; ++pidx)
                    for (size_t i = 0; i < fft.real_size; ++i) {
                        h[pidx + 1][i] = interpolation(state.phi[pidx][i]);
                        h[0][i] -= h[pidx + 1][i];
                    }
                save_result(out + "/fields_final.h5", fft, result, h);
                if (!rank)
                    std::cout << "completed step=" << state.step << " time=" << state.time
                              << " mean_composition=" << std::setprecision(17) << d.composition_mean
                              << " equilibrium=" << d.elastic.equilibrium << "\n";
            }
            double elapsed = maximum(MPI_Wtime() - started, fft.comm);
            if (!rank)
                std::cout << "MPI ranks=" << fft.ranks << " FFT threads/rank=" << threads
                          << " elapsed_seconds=" << elapsed << "\n";
        }
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
