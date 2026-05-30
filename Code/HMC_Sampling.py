from pathlib import Path

from ase import units
from ase.io import read, write
from mace.calculators import mace_off

from ase import Atoms
from ase.optimize.lbfgs import LBFGS
from ase.optimize.optimize import Dynamics # Make sure to import this!

from ase.mc.ensembles import NPT, NVT
from ase.mc.moveset import Moveset
from ase.mc.mc import MonteCarlo
from ase.io.trajectory import Trajectory

MODEL_PATH = Path("../mace_model/MACE-omol-0-extra-large-1024.model")
INIT_PATH = Path("trial_init.xyz")
OPT_PATH = Path("../Structure/trial.xyz")

def write_mc_trajectory():
    if dyn.last_accepted_config is None:
        return
    atoms_i = dyn.last_accepted_config
    with Trajectory(str(TRAJ_FILE), mode="a") as traj:
        traj.write(atoms_i)

    symbols = atoms_i.get_chemical_symbols()
    positions = atoms_i.get_positions()
    energy = dyn.calculator_results.get("energy", float("nan"))
    step = dyn.nsteps
    with open(MULTIXYZ_FILE, "a", encoding="ascii") as f:
        f.write(f"{len(symbols)}\n")
        f.write(f"mc_step={step} energy_eV={energy:.10f}\n")
        for sym, (x, y, z) in zip(symbols, positions):
            f.write(f"{sym} {x:.10f} {y:.10f} {z:.10f}\n")

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH.resolve()}")

# CPU is safer by default; set device='cuda' if your runtime has working CUDA.
calc = mace_off(model=str(MODEL_PATH), device="cuda")

Skip_opt = True
RESTART = False
LOG_FILE = Path("npt_mc.log")
TRAJ_FILE = Path("npt_mc.traj")
MULTIXYZ_FILE = Path("npt_mc_multi.xyz")
RUN_STEPS = 10000

if not Skip_opt:
    if not INIT_PATH.exists():
        raise FileNotFoundError(f"Initial structure not found: {INIT_PATH.resolve()}")
    atoms = read(str(INIT_PATH), 0)
    atoms.calc = calc

    print("Starting energy calculation...")
    print(atoms.get_potential_energy())
    print("Starting geometry optimization...")
    opt = LBFGS(atoms)
    opt.run(fmax=1e-2)

    print("Optimized energy:", atoms.get_potential_energy())
    print("Optimized positions:\n", atoms.get_positions())
    OPT_PATH.parent.mkdir(parents=True, exist_ok=True)
    write(str(OPT_PATH), atoms)
else:
    if not OPT_PATH.exists():
        raise FileNotFoundError(
            f"Optimized structure not found: {OPT_PATH.resolve()}. "
            "Set Skip_opt = False to generate it from trial_init.xyz."
        )
    atoms = read(str(OPT_PATH), 0)
    atoms.calc = calc

if atoms.cell.rank == 3 and atoms.pbc.any():
    ensemble = NPT(pressure=1.0 * units.bar, temperature_K=315.0)
    moveset = Moveset(ensemble.get_moves())
    moveset.adjust_parameter("Volume", "max_delta", 0.02)
    moveset.adjust_parameter("Volume", "probability", 0.5)
    moveset.adjust_parameter("HMC", "probability", 0.3)
    moveset.adjust_parameter("Translate", "probability", 0.1)
    moveset.adjust_parameter("Rotate", "probability", 0.1)
else:
    # Non-periodic systems have no defined volume, so NPT/Volume moves are invalid.
    ensemble = NVT(temperature_K=315.0)
    moveset = Moveset(ensemble.get_moves())
    moveset.adjust_parameter("HMC", "probability", 0.5)
    moveset.adjust_parameter("Translate", "probability", 0)
    moveset.adjust_parameter("Rotate", "probability", 0)

if not RESTART:
    LOG_FILE.write_text("")
    if TRAJ_FILE.exists():
        TRAJ_FILE.unlink()
    if MULTIXYZ_FILE.exists():
        MULTIXYZ_FILE.unlink()

dyn = MonteCarlo(
        atoms, moveset, dft_calc = calc,
        # Keep ASE internal trajectory disabled (ASE MC assertion bug with string trajectories).
        trajectory=None,
        logfile=str(LOG_FILE),
        loginterval=4,
    )

dyn.attach(write_mc_trajectory, interval=4)
# ASE MonteCarlo.run currently drops the steps argument internally.
# Call Dynamics.run directly to enforce an exact step count.
Dynamics.run(dyn, steps=RUN_STEPS)
