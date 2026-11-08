#include "cpu_mpi.hpp"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <hdf5.h>
#include <limits>
#include <sstream>
#include <stdexcept>

#ifndef H5_HAVE_PARALLEL
#error "This program requires parallel HDF5 built with MPI."
#endif
namespace microsim {
static std::string trim(std::string s) {
    auto a = s.find_first_not_of(" \t\r\n");
    return a == s.npos ? "" : s.substr(a, s.find_last_not_of(" \t\r\n") - a + 1);
}
Config Config::read(const std::string &path) {
    std::ifstream f(path);
    if (!f)
        throw std::runtime_error("Cannot open configuration: " + path);
    Config c;
    std::string line;
    int n = 0;
    while (std::getline(f, line)) {
        ++n;
        line = trim(line.substr(0, line.find('#')));
        if (line.empty())
            continue;
        auto equal = line.find('=');
        if (equal == line.npos)
            throw std::runtime_error("Expected key = value on line " + std::to_string(n));
        auto key = trim(line.substr(0, equal)), value = trim(line.substr(equal + 1));
        if (key.empty() || value.empty() || c.values.count(key))
            throw std::runtime_error("Empty or duplicate configuration key on line " +
                                     std::to_string(n));
        c.values[key] = value;
    }
    return c;
}
std::string Config::text(const std::string &key, const std::string &fallback) const {
    auto i = values.find(key);
    return i == values.end() ? fallback : i->second;
}
double Config::number(const std::string &key, double fallback) const {
    auto i = values.find(key);
    if (i == values.end())
        return fallback;
    size_t used = 0;
    double x = std::stod(i->second, &used);
    if (used != i->second.size() || !std::isfinite(x))
        throw std::runtime_error("Invalid finite number: " + key);
    return x;
}
int Config::integer(const std::string &key, int fallback) const {
    double x = number(key, fallback);
    if (std::floor(x) != x || x < std::numeric_limits<int>::min() ||
        x > std::numeric_limits<int>::max())
        throw std::runtime_error("Invalid integer: " + key);
    return int(x);
}
Tensor Config::tensor(const std::string &key, Tensor fallback) const {
    auto i = values.find(key);
    if (i == values.end())
        return fallback;
    auto s = i->second;
    std::replace(s.begin(), s.end(), ',', ' ');
    std::istringstream f(s);
    Tensor t;
    for (double &x : t)
        if (!(f >> x) || !std::isfinite(x))
            throw std::runtime_error("Expected six finite tensor components: " + key);
    std::string extra;
    if (f >> extra)
        throw std::runtime_error("Too many tensor components: " + key);
    return t;
}
static void ok(herr_t code, const std::string &action) {
    if (code < 0)
        throw std::runtime_error("HDF5: " + action);
}
static hid_t handle(hid_t h, const std::string &action) {
    if (h < 0)
        throw std::runtime_error("HDF5: " + action);
    return h;
}
static hid_t access(FFT &f) {
    hid_t p = handle(H5Pcreate(H5P_FILE_ACCESS), "file access");
    ok(H5Pset_fapl_mpio(p, f.comm, MPI_INFO_NULL), "MPI IO setup");
    return p;
}
static hid_t memory_space(FFT &f) {
    hsize_t dims[3] = {hsize_t(std::max<ptrdiff_t>(1, f.local_nx)), hsize_t(f.grid.ny),
                       hsize_t(f.grid.nz)};
    hid_t s = handle(H5Screate_simple(3, dims, nullptr), "local dataspace");
    if (!f.local_nx)
        ok(H5Sselect_none(s), "empty local dataspace");
    return s;
}
static void select_slab(hid_t space, FFT &f) {
    if (!f.local_nx) {
        ok(H5Sselect_none(space), "empty slab");
        return;
    }
    hsize_t start[3] = {hsize_t(f.x_start), 0, 0},
            count[3] = {hsize_t(f.local_nx), hsize_t(f.grid.ny), hsize_t(f.grid.nz)};
    ok(H5Sselect_hyperslab(space, H5S_SELECT_SET, start, nullptr, count, nullptr), "select slab");
}
static void attribute(hid_t file, const std::string &key, double value) {
    hid_t sp = handle(H5Screate(H5S_SCALAR), "scalar space"),
          a = handle(H5Acreate2(file, key.c_str(), H5T_IEEE_F64LE, sp, H5P_DEFAULT, H5P_DEFAULT),
                     "create attribute");
    ok(H5Awrite(a, H5T_NATIVE_DOUBLE, &value), "write attribute");
    H5Aclose(a);
    H5Sclose(sp);
}
static double attribute_read(hid_t file, const std::string &key) {
    hid_t a = handle(H5Aopen(file, key.c_str(), H5P_DEFAULT), "missing attribute " + key);
    double value;
    ok(H5Aread(a, H5T_NATIVE_DOUBLE, &value), "read attribute");
    H5Aclose(a);
    return value;
}
void write_fields(const std::string &path, FFT &f,
                  const std::map<std::string, const Real *> &fields,
                  const std::map<std::string, double> &attrs) {
    for (auto &[name, p] : fields)
        require(p && p->size() == f.real_size, "Invalid output field " + name, f.comm);
    hid_t ap = access(f),
          file = handle(H5Fcreate(path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, ap), "create " + path);
    H5Pclose(ap);
    attribute(file, "nx", f.grid.nx);
    attribute(file, "ny", f.grid.ny);
    attribute(file, "nz", f.grid.nz);
    attribute(file, "dx", f.grid.dx);
    attribute(file, "dy", f.grid.dy);
    attribute(file, "dz", f.grid.dz);
    attribute(file, "schema_version", 1);
    for (auto &[name, value] : attrs)
        attribute(file, name, value);
    hsize_t dims[3] = {hsize_t(f.grid.nx), hsize_t(f.grid.ny), hsize_t(f.grid.nz)};
    hid_t mem = memory_space(f),
          transfer = handle(H5Pcreate(H5P_DATASET_XFER), "transfer property");
    ok(H5Pset_dxpl_mpio(transfer, H5FD_MPIO_COLLECTIVE), "collective transfer");
    double dummy = 0;
    for (auto &[name, p] : fields) {
        hid_t sp = handle(H5Screate_simple(3, dims, nullptr), "global space"),
              ds = handle(H5Dcreate2(file, name.c_str(), H5T_IEEE_F64LE, sp, H5P_DEFAULT,
                                     H5P_DEFAULT, H5P_DEFAULT),
                          "create dataset " + name);
        select_slab(sp, f);
        ok(H5Dwrite(ds, H5T_NATIVE_DOUBLE, mem, sp, transfer, p->empty() ? &dummy : p->data()),
           "write " + name);
        H5Dclose(ds);
        H5Sclose(sp);
    }
    H5Pclose(transfer);
    H5Sclose(mem);
    H5Fclose(file);
}
Real read_field(const std::string &path, const std::string &name, FFT &f) {
    hid_t ap = access(f), file = handle(H5Fopen(path.c_str(), H5F_ACC_RDONLY, ap), "open " + path);
    H5Pclose(ap);
    hid_t ds = handle(H5Dopen2(file, name.c_str(), H5P_DEFAULT), "open field " + name),
          sp = handle(H5Dget_space(ds), "field space");
    hsize_t dims[3] = {};
    int rank = H5Sget_simple_extent_ndims(sp);
    if (rank == 3)
        H5Sget_simple_extent_dims(sp, dims, nullptr);
    require(rank == 3 && dims[0] == hsize_t(f.grid.nx) && dims[1] == hsize_t(f.grid.ny) &&
                dims[2] == hsize_t(f.grid.nz),
            "Input field shape mismatch: " + name, f.comm);
    select_slab(sp, f);
    hid_t mem = memory_space(f), transfer = handle(H5Pcreate(H5P_DATASET_XFER), "transfer");
    ok(H5Pset_dxpl_mpio(transfer, H5FD_MPIO_COLLECTIVE), "collective read");
    Real result(f.real_size);
    double dummy = 0;
    ok(H5Dread(ds, H5T_NATIVE_DOUBLE, mem, sp, transfer, result.empty() ? &dummy : result.data()),
       "read " + name);
    H5Pclose(transfer);
    H5Sclose(mem);
    H5Sclose(sp);
    H5Dclose(ds);
    H5Fclose(file);
    return result;
}
void write_state(const std::string &path, FFT &f, const State &s) {
    std::map<std::string, const Real *> fields{{"composition", &s.composition}};
    for (size_t p = 0; p < s.phi.size(); ++p)
        fields["phi" + std::to_string(p)] = &s.phi[p];
    write_fields(path + ".partial", f, fields,
                 {{"step", double(s.step)},
                  {"time", s.time},
                  {"variants", double(s.phi.size())},
                  {"model_hash_lo", double(s.model_hash & 0xffffffffULL)},
                  {"model_hash_hi", double(s.model_hash >> 32)}});
    MPI_Barrier(f.comm);
    if (!f.rank)
        std::filesystem::rename(path + ".partial", path);
    MPI_Barrier(f.comm);
}
State read_state(const std::string &path, FFT &f, int variants) {
    hid_t ap = access(f), file = handle(H5Fopen(path.c_str(), H5F_ACC_RDONLY, ap), "open restart");
    H5Pclose(ap);
    double step = attribute_read(file, "step"), time = attribute_read(file, "time"),
           np = attribute_read(file, "variants"), hashlo = attribute_read(file, "model_hash_lo"),
           hashhi = attribute_read(file, "model_hash_hi");
    bool valid = np == variants && std::isfinite(step) && step >= 0 &&
                 step <= std::numeric_limits<int>::max() && std::floor(step) == step &&
                 std::isfinite(time) && time >= 0;
    for (auto &item :
         std::map<std::string, double>{{"dx", f.grid.dx}, {"dy", f.grid.dy}, {"dz", f.grid.dz}}) {
        double v = attribute_read(file, item.first);
        valid = valid && std::abs(v - item.second) <= 1e-12 * item.second;
    }
    valid = valid && hashlo >= 0 && hashlo <= 4294967295. && std::floor(hashlo) == hashlo &&
            hashhi >= 0 && hashhi <= 4294967295. && std::floor(hashhi) == hashhi;
    H5Fclose(file);
    require(valid, "Restart metadata/spacing mismatch", f.comm);
    State s;
    s.step = int(step);
    s.time = time;
    s.model_hash = std::uint64_t(hashlo) | (std::uint64_t(hashhi) << 32);
    s.composition = read_field(path, "composition", f);
    for (int p = 0; p < variants; ++p)
        s.phi.push_back(read_field(path, "phi" + std::to_string(p), f));
    return s;
}
} // namespace microsim
