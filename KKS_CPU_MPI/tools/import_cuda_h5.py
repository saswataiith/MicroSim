#!/usr/bin/env python3
"""Import a global floating-point CUDA snapshot (COMP, PHI0...) for initial_file.
This imports actual data; it never reconstructs a missing original realization.
Use the CPU program to create subsequent restart checkpoints with provenance.
"""
import argparse
import pathlib
import h5py
import numpy as np

p=argparse.ArgumentParser(description=__doc__);p.add_argument("input");p.add_argument("output");p.add_argument("--variants",type=int,choices=(1,4),required=True);a=p.parse_args()
if pathlib.Path(a.input).resolve()==pathlib.Path(a.output).resolve():p.error("Input and output must differ")
with h5py.File(a.input) as src:
    names={"composition":"COMP",**{f"phi{i}":f"PHI{i}" for i in range(a.variants)}}
    shape=src["COMP"].shape
    for name in names.values():
        if src[name].shape!=shape or len(shape)!=3 or src[name].dtype.kind!="f":p.error("Expected matching 3D floating-point datasets; quantized int16 snapshots require their original scale metadata")
    with h5py.File(a.output,"w") as dst:
        dst.attrs["source_file"]=str(pathlib.Path(a.input).resolve())
        for new,old in names.items():
            target=dst.create_dataset(new,shape,dtype="f8")
            for i in range(shape[0]):
                slab=src[old][i,:,:]
                if not np.isfinite(slab).all():raise ValueError("Nonfinite original field: "+old)
                target[i,:,:]=slab
print("Imported original floating-point fields to",a.output)
