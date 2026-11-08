#!/usr/bin/env python3
"""Translate the documented Non_dim_calc=1 CUDA input to explicit CPU units.
Initial geometry is never guessed to reproduce the original nucleation run.
Use --initial-file for a converted frozen field, or explicitly opt into a demo.
"""
import argparse
import math
import pathlib


def convert(path, initial_file=None, demo=False):
    values = {}
    for line in pathlib.Path(path).read_text().splitlines():
        tokens = line.split("#", 1)[0].split()
        i = 0
        while i < len(tokens):
            key = tokens[i]
            count = 6 if key == "applsigma" else 1
            if i+count >= len(tokens):
                raise ValueError("Incomplete CUDA input line: "+line)
            raw = tokens[i+1:i+1+count]
            if key in values:
                raise ValueError("Duplicate CUDA key: "+key)
            try:
                values[key] = [float(x) for x in raw] if count > 1 else float(raw[0])
            except ValueError:
                values[key] = raw[0]
            i += 1+count
    def get(key):
        if key not in values:
            raise ValueError("Missing required CUDA parameter: "+key)
        x=values[key]
        if isinstance(x, float) and not math.isfinite(x):
            raise ValueError("Nonfinite CUDA parameter: "+key)
        return x
    if get("Non_dim_calc") != 1:
        raise ValueError("Automatic conversion is limited to Non_dim_calc=1. For other inputs specify explicit CPU energy-density units manually.")
    if get("Method") == 2:
        raise ValueError("CUDA Method=2 separately diffuses phase compositions and can change total solute. Select conservative CPU method 0 or 1 explicitly.")
    if get("EC_calc") != 0:
        raise ValueError("CUDA EC_calc=1 is inconsistent with its live 2h-1 stiffness kernels. Resolve the intended material interpolation before conversion.")
    if get("time_elast") != 0 or (get("appl_stress") and get("time_sa") != 0):
        raise ValueError("Delayed elasticity/loading must be represented as explicit staged CPU runs; this converter accepts activation at step zero only.")
    if not initial_file and not demo:
        raise ValueError("Provide --initial-file from the original CUDA fields, or --example-initialization to explicitly use a different demonstration geometry.")
    L, E, T = get("Lconv"), get("Econv"), get("Tconv")
    if min(L,E,T,get("Vm")) <= 0:
        raise ValueError("Unit conversion scales and molar volume must be positive")
    density_scale=E/L**3
    a_matrix=.5*get("f0A")/get("Vm")*density_scale
    a_precip=.5*get("f0B")/get("Vm")*density_scale
    D=get("diffusivity")*L*L/T
    gamma=get("gamma_intf")*E/(L*L)
    width=get("Ln")*(get("dx")*L if get("Ln_dx") else 1)
    kappa, omega, kinetic=get("kpa"),get("omega"),get("relax_coeff")
    scheme=int(get("kappa_omega_calc"))
    if scheme==1:
        kappa=3*gamma*width/get("alpha")
        omega=6*get("alpha")*gamma/width
    elif scheme in (2,3):
        curvature=a_matrix
        if scheme==3:
            vf=(get("Supersaturation")-get("c_alpha_eq"))/(get("c_beta_eq")-get("c_alpha_eq"))
            curvature=(1-vf)*a_matrix+vf*a_precip
        M=D/(2*curvature)
        factor=(get("c_beta_eq")-get("c_alpha_eq"))**2/M
        lam=math.log(.95*.95/(.05*.05))
        omega=3*gamma*lam/width
        kappa=6*gamma*width/lam
        kinetic=omega/(3*kappa*.7833*factor)
    elif scheme!=0:
        raise ValueError("Unknown kappa_omega_calc")
    def stiffness(suffix):
        if get("elast_consts"):
            return [get("c11"+suffix)*1e9*density_scale,get("c12"+suffix)*1e9*density_scale,get("c44"+suffix)*1e9*density_scale]
        mu=get("mu_"+suffix)*1e9*density_scale
        nu=get("nu_"+suffix);az=get("Az_"+suffix)
        return [mu*(2*(2+az)/(1+az)-(1-4*nu)/(1-2*nu)),mu*(2*az/(1+az)-(1-4*nu)/(1-2*nu)),mu*2*az/(1+az)]
    cm,cp=stiffness("m"),stiffness("p")
    if get("inhom")==1 and any(abs(a-b)>1e-12*max(abs(a),abs(b)) for a,b in zip(cm,cp)):
        raise ValueError("CUDA inhom=1 selects a homogeneous solver despite unequal phase constants. Choose the intended homogeneous stiffness manually.")
    variants=int(get("nof_variants"));ep=get("epszero");aniso=get("misfit_aniso")
    cfg=dict(mode="phasefield",nx=int(get("nx")),ny=int(get("ny")),nz=int(get("nz")),
             dx=get("dx")*L,dy=get("dy")*L,dz=get("dz")*L,variants=variants,
             dt=get("dt"),steps=int(get("numsteps")),output="cpu-converted",output_every=max(1,int(get("TRACK_OBS"))),
             kinetic=kinetic,kappa=kappa,omega=omega,theta=get("theta"),theta2=max(0,get("theta2")),
             a_matrix=a_matrix,a_precip=a_precip,c_matrix=get("c_alpha_eq"),c_precip=get("c_beta_eq"),
             diffusivity=D,mobility_method=int(get("Method")),evolve_composition=int(get("comp_evolution")),elasticity=int(get("elast_int")),
             elastic_max_iterations=max(1000,int(get("nelast"))),elastic_change_tolerance=1e-10,
             elastic_equilibrium_tolerance=1e-8,composition=get("Supersaturation"),initial="sphere",
             interface_width=max(get("dx")*L,width/4))
    cfg["applied_stress"]=",".join(format(v*1e6*density_scale if get("appl_stress") else 0,".17g") for v in get("applsigma"))
    for p in range(variants+1):
        for name,v in zip(("c11","c12","c44"),cp if p else cm):cfg[f"phase{p}.{name}"]=v
        cfg[f"phase{p}.eigenstrain"]=",".join(format(v,".17g") for v in ([ep,ep*aniso,ep*aniso,0,0,0] if p else [0]*6))
    if initial_file:cfg["initial_file"]=str(pathlib.Path(initial_file).resolve())
    notes=["Parameters translated to explicit density/length/time units; no CUDA binaries executed.",
           "CPU uses energy-density parabolas: a=Nv*Am(atom); therefore mu is also a density and M=D/(2a).",
           "The original FE_form=1 common affine energy gauge has the same deterministic KKS forces/fluxes; CPU reports FE_form=0 energy.",
           "CPU convergence uses scaled displacement+mean-strain change AND equilibrium+mean stress; the raw CUDA Tol is not transplanted.",
           "Method 3 uses the local-mobility volume average, with the missing factor 1/2 corrected and updated as the phase fractions evolve.",
           "Legacy noise, smoothing, GPU flags, output precision switches, and stochastic nucleus placement are not copied."]
    if demo:notes.append("DEMONSTRATION GEOMETRY ONLY: original nucleus arrangement/composition profile is not reproduced.")
    return cfg,notes


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("input");p.add_argument("output");g=p.add_mutually_exclusive_group(required=True);g.add_argument("--initial-file");g.add_argument("--example-initialization",action="store_true");a=p.parse_args()
    try:
        cfg,notes=convert(a.input,a.initial_file,a.example_initialization)
        text="\n".join("# "+s for s in notes)+"\n"+"\n".join(f"{k} = {v:.17g}" if isinstance(v,float) else f"{k} = {v}" for k,v in cfg.items())+"\n"
        pathlib.Path(a.output).write_text(text);print("Wrote",a.output);print("\n".join(notes))
    except (ValueError,ZeroDivisionError) as e:p.exit(2,str(e)+"\n")
