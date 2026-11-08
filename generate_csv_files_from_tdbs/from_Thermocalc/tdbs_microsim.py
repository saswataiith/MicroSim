import os
import sys
from pathlib import Path
from table_utils import output_directory, phase_frame, common_phase_frame, write_composition_tables
from tc_python import *
import pandas as pd
import json
import csv
import shutil
import logging
from tcsetup import *

logging.basicConfig(filename='tdbs_microsim.log', level=logging.INFO)
#, format='%(asctime)s %(message)s', datefmt='%m/%d/%y %I:%M:%S %p')
logging.info('The Calculation starts for composition and hessian files')


#open and read json file and store the variables
with open(sys.argv[1], 'r') as myfile:
    data=myfile.read()

# parse file
obj = json.loads(data)

Thermodatabase	= obj['Thermodynamic_Database']
Kineticdatabase	= obj['Kintic_Database']
solvent		= obj['Base_element']
solutes		= obj['Solutes']
conc		= obj['alloy_composition']
comp_choice	= obj["Composition_unit"]
T_min		= obj['Tmin']
T_max		= obj['Tmax']
matrix_pahse	= obj['Matrix_phase']
Active_phases	= obj['Solid_Phases']
material	= obj['Material']

logging.info('The variables are imported from JSON script')

num_steps = int(T_max-T_min)

current_dir = os.getcwd()

# Reuse a deterministic output directory on repeated exports.
output_dir = output_directory(material).resolve()
os.chdir(output_dir)
manifest = {'status': 'incomplete', 'material': material, 'thermodynamic_database': Thermodatabase, 'kinetic_database': Kineticdatabase, 'temperature_unit': 'K', 'phase_composition_unit': 'mole_fraction', 'hessian_basis': 'phase_tie_line_mole_composition', 'mobility_exported': False}
Path('export_manifest.json').write_text(json.dumps(manifest, indent=2))
logging.info('Output directory: %s', output_dir)

#Strart the thermocalc Calculation
with TCPython() as start:
    # create equilibrium calculation object
    S1 = setup(start, Thermodatabase, Kineticdatabase, solutes, solvent, conc, comp_choice, T_max, T_min, material)
    calculation_setup = S1.Database_selection()
    logging.info('System is generated with Thermodynamic database and Kinetic database')

    #Set condition for equilobrium thermodynamic calculation
    comp_system = S1.Eq_composition(calculation_setup, num_steps)
    logging.info('The instace for equilibrium composition is created from '+str(T_min)+' to '+str(T_max)+' with_property_diagram_calculation')

    # Phase intervals can have different temperature grids. Join on T before
    # exporting any tie-line or Hessian values.
    frames = []
    for phase in Active_phases + [matrix_pahse]:
        groups = comp_system.get_values_grouped_by_quantity_of(
            ThermodynamicQuantity.temperature(), "X(" + phase + ",*)")
        frames.append(phase_frame(groups, phase, solutes))
    df1 = common_phase_frame(frames)
    df1.to_csv('Composition_' + material + '.csv', index=False)
    write_composition_tables(df1, Active_phases, matrix_pahse, solutes)

    #Calculating the Hessian values from the Hessian_setup
    logging.info('Instance for hessian calculation is created with_single_equilibrium_calculation')
    phase_list=[matrix_pahse]+Active_phases
    for phase in phase_list:
        dfphase=pd.DataFrame()
        #get system for Hessian calculation with phase
        Hessian_setup= S1.Hessian_setup(calculation_setup, phase)
        logging.info('The instance for hessian calculation for the '+phase+' is created')


        for i in range(len(df1.index)):
            calc_temp = df1.at[i,"Temperature"]
            dfphase.at[i,"Temperature"]= calc_temp
            
            #set condition for Hessian calculation
            phase_composition = [float(df1.at[i, element + "@" + phase]) for element in solutes]
            l = S1.Hessian_calculation(Hessian_setup, calc_temp, phase_composition)

            for j1, s1 in zip(range(len(solutes)),solutes):
                for j2, s2 in zip(range(len(solutes)),solutes):
                    if j1==j2:
                        ddgdxdx=l.get_value_of("mu("+s1+").x("+s2+")")-l.get_value_of("mu("+solvent+").x("+s2+")")
                        dfphase.at[i,'HSN('+s1+','+s2+')@'+phase]=ddgdxdx  

            for j1, s1 in zip(range(len(solutes)),solutes):
                for j2, s2 in zip(range(len(solutes)),solutes):
                    if j1<j2:
                        ddgdxdx=l.get_value_of("mu("+s1+").x("+s2+")")-l.get_value_of("mu("+solvent+").x("+s2+")")
                        dfphase.at[i,'HSN('+s1+','+s2+')@'+phase]=ddgdxdx
                    
            dfphase.to_csv("HSN_"+phase+".csv",index=False)
            logging.info('Hessian data for '+phase+' is saved to .csv file')

manifest['status'] = 'complete'
Path('export_manifest.json.tmp').write_text(json.dumps(manifest, indent=2))
os.replace('export_manifest.json.tmp', 'export_manifest.json')

# Preserve calculation caches for provenance and avoid deleting unrelated files.
os.chdir(current_dir)
logging.info('Calculation complete. Mobility matrices are not exported by this script.')
