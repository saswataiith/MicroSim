#!/usr/bin/env python3
"""Executable acceptance checks: MPI, original CUDA kernels, NumPy step oracle,
parallel output, restart across rank counts, and explicit failure behaviour."""
import csv
import hashlib
import json
import os
import pathlib
import platform
import shlex
import subprocess
import sys
import time
import tarfile

import h5py
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
os.chdir(ROOT)
RUNS = ROOT / "build" / "validation"
RUNS.mkdir(parents=True, exist_ok=True)
MPIEXEC = shlex.split(os.environ.get("MPIEXEC", "mpiexec"))
records = []


def launch(name, ranks, executable="build/microsim_cpu", config=None, fail=False):
    args = MPIEXEC + ["-n", str(ranks), executable]
    if config:
        args.append(str(config))
    start = time.monotonic()
    r = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                       text=True, timeout=120)
    (RUNS / (name + ".log")).write_text(r.stdout)
    passed = (r.returncode != 0) if fail else r.returncode == 0
    records.append(dict(name=name, ranks=ranks, status="PASS" if passed else "FAIL",
                        seconds=time.monotonic()-start, returncode=r.returncode))
    if not passed:
        raise AssertionError(name + "\n" + r.stdout)
    print("PASS", name)
    return r


BASE = dict(mode="phasefield", nx=9, ny=7, nz=5, initial="sine", composition=.19,
            steps=4, output_every=2, dt=.0005, variants=1, a_matrix=1.3,
            a_precip=.7, diffusivity=.02, theta2=3,
            **{"phase1.eigenstrain": "-.005,.009,-.003,.002,-.001,.003",
               "applied_stress": ".02,-.01,0,.005,0,.003"})


def run(name, ranks=1, values=None, fail=False):
    opts = dict(BASE)
    opts.update(values or {})
    out = RUNS / name
    opts["output"] = str(out)
    config = RUNS / (name + ".cfg")
    config.write_text("\n".join(f"{k} = {v}" for k, v in opts.items())+"\n")
    launch(name, ranks, config=config, fail=fail)
    return out


def fields(path):
    with h5py.File(path) as f:
        return {k: f[k][...] for k in f}


def compare(name, a, b, tolerance=3e-10):
    assert set(a) == set(b)
    err = max(float(np.max(np.abs(a[k]-b[k]))) for k in a)
    assert err < tolerance, (name, err)
    records.append(dict(name=name, status="PASS", max_absolute_error=err))
    print("PASS", name, "max_error", err)


def step_oracle(method):
    before = run(f"step{method}_before", values=dict(steps=0, mobility_method=method))
    after = run(f"step{method}_after", ranks=2, values=dict(steps=1, mobility_method=method))
    state = fields(before / "state_final.h5")
    elastic = fields(before / "fields_final.h5")
    phi, c = state["phi0"], state["composition"]
    h = phi**3*(10-15*phi+6*phi**2)
    dh = 30*phi**2*(1-phi)**2
    Am, Ap = BASE["a_matrix"], BASE["a_precip"]
    ca = (c-h*(.23-(Am/Ap)*.15))/(1-h+h*(Am/Ap))
    cb = .23+(Am/Ap)*(ca-.15)
    mu = 2*Am*(ca-.15)
    df = 2*phi*(1-phi)*(1-2*phi)+dh*(
        Ap*(cb-.23)**2-Am*(ca-.15)**2+(ca-cb)*mu+
        elastic["dFdh1"]-elastic["dFdh0"])
    shape = phi.shape
    axes = [2*np.pi*np.fft.fftfreq(n) for n in shape[:2]]
    axes.append(2*np.pi*np.fft.rfftfreq(shape[2]))
    k = np.meshgrid(*axes, indexing="ij")
    k2 = sum(x*x for x in k)
    dt, D = BASE["dt"], BASE["diffusivity"]
    expected_phi = np.fft.irfftn((np.fft.rfftn(phi)-dt*np.fft.rfftn(df))/
                               (1+dt*.2*k2), s=shape, axes=(0,1,2))
    if method in (0, 3):
        M = D/(2*Am) if method == 0 else D*((1-h.mean())/(2*Am)+h.mean()/(2*Ap))
        expected_c = np.fft.irfftn(np.fft.rfftn(c)-dt*M*k2*np.fft.rfftn(mu), s=shape, axes=(0,1,2))
    else:
        derivative_k = [x.copy() for x in k]
        for a, n in enumerate(shape):
            if n % 2 == 0:
                index = [slice(None)]*3
                index[a] = n//2
                derivative_k[a][tuple(index)] = 0
        M = D*((1-h)/(2*Am)+h/(2*Ap))
        divergence = sum(1j*kk*np.fft.rfftn(M*np.fft.irfftn(1j*kk*np.fft.rfftn(mu), s=shape, axes=(0,1,2)))
                         for kk in derivative_k)
        expected_c = np.fft.irfftn(np.fft.rfftn(c)+dt*divergence, s=shape, axes=(0,1,2))
    result = fields(after / "state_final.h5")
    compare(f"NumPy independent one-step oracle method {method}",
            dict(phi0=expected_phi, composition=expected_c), result, 3e-11)


try:
    for ranks in (1, 2, 4):
        launch(f"solver_{ranks}ranks", ranks, "build/test_solver")
    launch("original_cuda_local_kernels", 2, "build/test_cuda_oracle")
    baseline = run("coupled_1rank")
    for ranks in (2, 4):
        other = run(f"coupled_{ranks}ranks", ranks)
        for filename in ("state_final.h5", "fields_final.h5"):
            compare(f"Rank invariance {ranks}: {filename}", fields(baseline/filename), fields(other/filename))
    split = run("restart_part", 2, dict(steps=2))
    resumed = run("restart_resumed", 4, dict(steps=2, restart=str(split/"state_final.h5")))
    for filename in ("state_final.h5", "fields_final.h5"):
        compare("Restart across 2-to-4 ranks: "+filename, fields(baseline/filename), fields(resumed/filename))
    with h5py.File(resumed/"state_final.h5") as f:
        assert f.attrs["step"] == 4 and abs(f.attrs["time"]-.002)<1e-15
    for method in (0, 1, 3):
        step_oracle(method)
    multi = run("four_variants", 4, dict(variants=4, steps=3, mobility_method=1))
    one = run("four_variants_serial", 1, dict(variants=4, steps=3, mobility_method=1))
    compare("Four-variant MPI evolution", fields(multi/"state_final.h5"), fields(one/"state_final.h5"))
    empty = run("empty_slabs", 4, dict(nx=3, ny=5, nz=1, steps=2))
    serial = run("empty_slabs_serial", 1, dict(nx=3, ny=5, nz=1, steps=2))
    compare("Empty FFTW/HDF5 slabs and z-invariant 3D", fields(empty/"state_final.h5"), fields(serial/"state_final.h5"))
    threaded = run("threaded_fft", 2, dict(fft_threads=2))
    compare("FFTW threaded plans", fields(baseline/"state_final.h5"), fields(threaded/"state_final.h5"))
    maxnorm = run("max_norm", 2, dict(elastic_norm="max"))
    # Different stopping norms can stop on different iterations. Compare at
    # the stated 1e-8 equilibrium accuracy, not bitwise trajectory identity.
    compare("Max displacement norm", fields(baseline/"fields_final.h5"), fields(maxnorm/"fields_final.h5"),1e-8)
    run("restart_mismatch_rejected", 2, dict(restart=str(split/"state_final.h5"), dt=.001), fail=True)
    for name, opts in [
        ("cap_rejected", dict(elastic_max_iterations=1)),
        ("zero_tolerance_rejected", dict(elastic_change_tolerance=0)),
        ("unstable_stiffness_rejected", {"phase0.c44": -1}),
        ("unknown_option_rejected", dict(typo_parameter=1)),
        ("invalid_method_rejected", dict(mobility_method=2)),
        ("malformed_tensor_rejected", dict(applied_stress="1,2,3")),
    ]:
        run(name, 2, opts, fail=True)
    run("elastic_standalone", 2, dict(mode="elasticity", steps=0))
    # A moderate grid smoke run is an execution check, not a scaling benchmark.
    run("grid32_smoke", 4, dict(nx=32, ny=32, nz=32, steps=2))
    # Exercise the exact bundled legacy input and actual HDF dataset mapping.
    with tarfile.open(ROOT/"reference/3DKKSMP.tar.gz") as t:
        (RUNS/"legacy_input").write_bytes(t.extractfile("3DKKSMP_OPT/bin/InputParams").read())
    r=subprocess.run([sys.executable,"tools/convert_cuda_input.py",str(RUNS/"legacy_input"),str(RUNS/"converted.cfg"),"--example-initialization"],text=True,capture_output=True,check=True)
    converted=(RUNS/"converted.cfg").read_text()
    assert "dx = 1\n" in converted and "variants = 4\n" in converted
    assert "a_matrix = " in converted and "phase4.eigenstrain" in converted
    state=fields(baseline/"state_final.h5")
    with h5py.File(RUNS/"legacy_snapshot.h5","w") as f:
        f["COMP"]=state["composition"];f["PHI0"]=state["phi0"]
    subprocess.run([sys.executable,"tools/import_cuda_h5.py",str(RUNS/"legacy_snapshot.h5"),str(RUNS/"imported.h5"),"--variants","1"],check=True,capture_output=True)
    compare("Original CUDA floating-point snapshot import",state,fields(RUNS/"imported.h5"),1e-15)
    run("imported_initial_state",2,dict(initial_file=str(RUNS/"imported.h5"),steps=0))
    records.append(dict(name="Bundled CUDA parameter conversion",status="PASS"))
    with open(baseline/"diagnostics.csv") as f:
        rows=list(csv.DictReader(f))
    mass=[float(r["composition_mean"]) for r in rows]
    assert max(mass)-min(mass)<2e-14
    print("ALL ACCEPTANCE CHECKS PASSED")
finally:
    report=dict(platform=platform.platform(), python=sys.version, numpy=np.__version__,
                h5py=h5py.__version__, checks=records,
                source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                               for d in ("src", "include", "tests", "tools") for p in sorted((ROOT/d).glob("*"))
                               if p.is_file()})
    (ROOT/"VALIDATION.json").write_text(json.dumps(report, indent=2)+"\n")
