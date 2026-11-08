import os
from tc_python import *


class setup:
    def __init__(self, start, Thermodynamic_database, Kinetic_database, solutes, solvent, composition, comp_choice, Tmax, Tmin, material):
        self.Thermodata = Thermodynamic_database
        self.Kineticdata = Kinetic_database
        self.solutes = solutes
        self.solvent = solvent
        self.composition = composition
        self.Tmax = Tmax
        self.Tmin = Tmin
        self.Fname = material
        self.start = start
        self.comp_choice = comp_choice

    def _set_multiple_conditions(self, composition=None, unit=None):
        values = self.composition if composition is None else composition
        basis = self.comp_choice if unit is None else unit
        if basis not in ('mass', 'mole'):
            raise ValueError("Composition_unit must be 'mass' or 'mole'")
        if len(values) != len(self.solutes):
            raise ValueError('One composition value is required for every solute')
        c = "W(" if basis == 'mass' else "X("
        temp=["s-c"]
        for i in range(len(self.solutes)):
            temp.append(c+self.solutes[i]+")="+str(values[i]))
        return (' '.join(temp))

    def Database_selection(self):
        calculation_setup= (
        self.start
            .set_cache_folder(os.path.basename(self.Fname[:-4]) + "_cache")
            .select_thermodynamic_and_kinetic_databases_with_elements(self.Thermodata, self.Kineticdata, self.solutes+[self.solvent])
            )
        return calculation_setup
    
    def Eq_composition(self, calculation_setup, num_steps):
        condition = setup._set_multiple_conditions(self)
        Tinit = self.Tmin + int(0.5*(self.Tmax - self.Tmin))
        propertyDia=(calculation_setup
                .get_system()
                .with_property_diagram_calculation()
                .run_poly_command(condition)          
                .with_axis(CalculationAxis(ThermodynamicQuantity.temperature()).
                              set_min(self.Tmin).
                              set_max(self.Tmax).
                              with_axis_type(Linear().
                                             set_min_nr_of_steps(num_steps)))
                .set_condition(ThermodynamicQuantity.temperature(), Tinit)
                .enable_global_minimization()
                .calculate())
        return propertyDia
    
    def Hessian_setup(self, calculation_setup, phase):
        Hessian=(calculation_setup
                .without_default_phases()
                .select_phase(phase)
                .get_system()
                .with_single_equilibrium_calculation())
        return Hessian

    def Hessian_calculation(self, Hessian_setup, temperature, phase_mole_fractions):
        # Curvatures must refer to this phase's tie-line composition, even when
        # the nominal alloy was specified on a mass-fraction basis.
        condition = self._set_multiple_conditions(phase_mole_fractions, 'mole')
        l=(Hessian_setup
            .set_condition("T", temperature)
            .run_poly_command(condition)
            .disable_global_minimization()
            .calculate())
        return l
    
    def order(self, phase, some_value):
        order = ['Temperature'] if some_value == 1 else []
        for i in range(len(self.solutes)):
            order.append(self.solutes[i] + phase)
        return order
