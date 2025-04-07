"""
Thermal Model Predictive Control (MPC) Module - DAQ Version
Uses real-time DAQ temperature readings rather than MATLAB files
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp
from scipy import linalg
import cvxpy as cp
import random
from dataclasses import dataclass
import logging

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class ThermalZone:
    """Class representing a thermal zone in a building"""
    name: str
    thermal_resistance: float    # Rᵢ in (°F/W)
    thermal_capacitance: float   # Cᵢ in (J/°F)
    initial_temperature: float

class ThermalSystem:
    """Class representing a thermal system with multiple zones"""
    def __init__(self, zones, connections, ambient_temperature):
        """
        Initialize the thermal system
        
        Parameters:
        - zones: List of ThermalZone objects
        - connections: 2D numpy array of conduction coefficients (kᵢⱼ in W/°F)
        - ambient_temperature: Function that returns ambient temp at given time
        """
        self.zones = zones
        self.connections = connections
        self.ambient_temperature = ambient_temperature

def create_real_system(temperature_readings=None):
    """
    Create a thermal system model from real temperature readings
    
    Parameters:
    - temperature_readings: Dictionary with keys TR1, TR2, etc. containing temperatures
    
    Returns:
    - ThermalSystem object
    """
    # Thermal parameters from the original model
    C = 602.8512 * 1000  # thermal capacitance (J/°F) *1000
    R_heated = 0.8294    # thermal resistance (°F/W)
    R_unheated = 0.0288
    R_br3 = 0.2229
    R_br4 = 0.1661

    k_c = 0.9405         # horizontal conduction coefficient
    k_v = 1.2214         # vertical conduction coefficient
    k_br3_extra = 0.1751 # extra coupling for BR3
    k_br4_extra = 1.228  # extra coupling for BR4

    # Default or provided temperatures
    if temperature_readings:
        zone_temps = {
            "TR1": temperature_readings.get("TR1", 75.0),
            "TR2": temperature_readings.get("TR2", 75.0),
            "TR3": temperature_readings.get("TR3", 75.0),
            "TR4": temperature_readings.get("TR4", 75.0),
            "BR1": temperature_readings.get("BR1", 75.0),
            "BR2": temperature_readings.get("BR2", 75.0),
            "BR3": temperature_readings.get("BR3", 75.0),
            "BR4": temperature_readings.get("BR4", 75.0)
        }
        ambient_temp = temperature_readings.get("Ambient", 75.0)
    else:
        # Default values if no readings provided
        zone_temps = {name: 75.0 for name in ["TR1", "TR2", "TR3", "TR4", "BR1", "BR2", "BR3", "BR4"]}
        ambient_temp = 75.0

    # Create a constant ambient temperature function
    ambient_func = lambda t: ambient_temp

    # Create thermal zones
    zones = [
        ThermalZone("TR1", R_unheated, C, zone_temps["TR1"]),
        ThermalZone("TR2", R_unheated, C, zone_temps["TR2"]),
        ThermalZone("TR3", R_heated, C, zone_temps["TR3"]),
        ThermalZone("TR4", R_heated, C, zone_temps["TR4"]),
        ThermalZone("BR1", R_heated, C, zone_temps["BR1"]),
        ThermalZone("BR2", R_heated, C, zone_temps["BR2"]),
        ThermalZone("BR3", R_br3, C, zone_temps["BR3"]),
        ThermalZone("BR4", R_br4, C, zone_temps["BR4"])
    ]

    # Create connection matrix
    n = len(zones)
    connections = np.zeros((n, n))

    # Add horizontal connections
    for i, j in [(0,1), (0,2), (1,3), (2,3), (4,5), (4,6), (5,7), (6,7)]:
        connections[i,j] = connections[j,i] = k_c

    # Add vertical connections
    for i, j in [(0,4), (1,5), (2,6), (3,7)]:
        connections[i,j] = connections[j,i] = k_v

    # Add extra connections
    connections[5,6] = connections[6,5] = k_c + k_br3_extra
    connections[4,7] = connections[7,4] = k_c + k_br4_extra

    logger.info(f"Created thermal system with temperatures: {zone_temps}")
    return ThermalSystem(zones, connections, ambient_func)

def update_thermal_system(system, temperature_readings):
    """
    Update an existing thermal system with new temperature readings
    
    Parameters:
    - system: Existing ThermalSystem object
    - temperature_readings: Dictionary with new temperature readings
    
    Returns:
    - Updated ThermalSystem object
    """
    if not temperature_readings:
        logger.warning("No temperature readings provided for update")
        return system
        
    # Update ambient temperature function if ambient reading is provided
    if "Ambient" in temperature_readings and temperature_readings["Ambient"] is not None:
        ambient_temp = temperature_readings["Ambient"]
        system.ambient_temperature = lambda t: ambient_temp
        logger.info(f"Updated ambient temperature to {ambient_temp}°F")
    
    # Update zone temperatures
    zone_names = ["TR1", "TR2", "TR3", "TR4", "BR1", "BR2", "BR3", "BR4"]
    for i, name in enumerate(zone_names):
        if name in temperature_readings and temperature_readings[name] is not None:
            old_temp = system.zones[i].initial_temperature
            system.zones[i].initial_temperature = temperature_readings[name]
            logger.info(f"Updated {name} temperature: {old_temp}°F -> {temperature_readings[name]}°F")
    
    return system

def thermal_dynamics(t, T, system, cooling):
    """
    ODE function for thermal dynamics
    
    Parameters:
    - t: time in seconds
    - T: array of temperatures for each zone
    - system: ThermalSystem object
    - cooling: array of cooling values for each zone
    
    Returns:
    - dT: rate of change of temperatures
    """
    t_hr = t / 3600
    T_amb = system.ambient_temperature(t_hr)
    
    n = len(system.zones)
    dT = np.zeros(n)
    
    for i in range(n):
        Ri = system.zones[i].thermal_resistance
        Ci = system.zones[i].thermal_capacitance
        
        # Heat exchange with ambient
        dT[i] = (T_amb - T[i]) / (Ri * Ci)
        
        # Heat exchange with connected zones
        for j in range(n):
            if j != i and system.connections[i,j] > 0:
                dT[i] += system.connections[i,j] * (T[j] - T[i]) / Ci
        
        # Cooling effect
        dT[i] += cooling[i] / Ci
    
    return dT

def optimize_ac_damper_control(system, x_current=None, future_amb=None, horizon=12, dt=300.0, T_min=60.0, T_max=110.0, bigM=50.0):
    """
    MPC optimization for AC and damper control using CVXPY
    
    Parameters:
    - system: ThermalSystem object
    - x_current: current temperatures of all zones (if None, use initial temperatures from system)
    - future_amb: predicted ambient temperatures for the horizon (if None, use constant current ambient)
    - horizon: number of steps in prediction horizon
    - dt: time step in seconds
    - T_min, T_max: temperature bounds
    - bigM: big-M value for logical constraints
    
    Returns:
    - ac_first: AC on/off decision for first time step
    - damper_first: damper open/closed decisions for first time step
    - T_pred: predicted temperature trajectories (if available)
    """
    n = len(system.zones)
    N = horizon
    
    # If x_current not provided, use initial temperatures from system
    if x_current is None:
        x_current = np.array([zone.initial_temperature for zone in system.zones])
    
    # If future_amb not provided, use constant current ambient
    if future_amb is None:
        current_amb = system.ambient_temperature(0)  # Current time
        future_amb = [current_amb] * horizon
    
    # Create CVXPY variables
    T = cp.Variable((n, N+1))                # Temperatures
    ac_on = cp.Variable(N, boolean=True)     # AC on/off
    damper = cp.Variable((n, N), boolean=True)  # Damper open/closed
    cool = cp.Variable((n, N), boolean=True)    # Cooling active
    
    critical_temp = 80.0  # Critical temperature threshold
    temp_violation = cp.Variable((n, N), nonneg=True)  # Temperature violations
    
    # Constraints
    constraints = []
    
    # Temperature bounds
    for i in range(n):
        for k in range(N+1):
            constraints.append(T[i, k] >= T_min)
            constraints.append(T[i, k] <= T_max)
    
    # Initial conditions
    for i in range(n):
        constraints.append(T[i, 0] == x_current[i])
    
    # Temperature violation constraints
    for i in range(n):
        for k in range(N):
            constraints.append(T[i, k] - critical_temp <= temp_violation[i, k])
    
    # Damper and cooling logic
    threshold = 75.0  # Temperature threshold for damper opening
    
    for i in range(n):
        for k in range(N):
            # Damper opens if T >= threshold (using big-M formulation)
            constraints.append(T[i, k] - threshold <= bigM * damper[i, k])
            constraints.append(T[i, k] - threshold >= -bigM * (1 - damper[i, k]))
            
            # Cooling requires AC on and damper open
            constraints.append(cool[i, k] <= ac_on[k])
            constraints.append(cool[i, k] <= damper[i, k])
            constraints.append(cool[i, k] >= ac_on[k] + damper[i, k] - 1)
    
    # System dynamics with cooling
    P_cool_zone = 366.0  # Cooling power per zone (W)
    
    for k in range(N):
        T_amb = future_amb[k]
        for i in range(n):
            zone = system.zones[i]
            R = zone.thermal_resistance
            C = zone.thermal_capacitance
            
            # Ambient heat exchange
            ambient_term = dt * (T_amb - T[i, k]) / (R * C)
            
            # Heat exchange with connected zones
            conduction_term = 0
            for j in range(n):
                if j != i and system.connections[i, j] > 0:
                    conduction_term += dt * system.connections[i, j] * (T[j, k] - T[i, k]) / C
            
            # Cooling term
            cooling_term = dt * (-P_cool_zone / C) * cool[i, k]
            
            # Temperature dynamics
            constraints.append(T[i, k+1] == T[i, k] + ambient_term + conduction_term + cooling_term)
    
    # Objective function
    violation_penalty = 100000000.0  # High penalty for temperature violations
    
    energy_cost = 0
    for k in range(N):
        energy_cost += (620 * ac_on[k] + 0.8 * (1 - ac_on[k])) * dt
    
    violation_cost = violation_penalty * cp.sum(temp_violation)
    
    objective = cp.Minimize(energy_cost + violation_cost)
    
    # Create and solve the problem
    prob = cp.Problem(objective, constraints)
    try:
        # Try different solvers until one works
        solvers_to_try = [None, 'GLPK_MI', 'CBC']  # None means default solver first
        
        for solver in solvers_to_try:
            try:
                if solver:
                    solver_method = getattr(cp, solver)
                    prob.solve(solver=solver_method, verbose=False)
                else:
                    prob.solve(verbose=False)
                    
                # If we get here, the solver worked
                break
            except Exception as e:
                logger.warning(f"Solver {solver} failed: {e}")
                continue
        
        # Check if the problem was solved successfully
        if prob.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            ac_first = int(round(ac_on.value[0]))
            damper_first = [int(round(damper.value[i, 0])) for i in range(n)]
            T_pred = T.value
            
            # After solving optimization, before returning results
            if T_pred is not None:
                # Log the prediction and current state for validation
                import os
                import json
                from datetime import datetime
                
                timestamp = datetime.now().isoformat()
                prediction_log = {
                    "timestamp": timestamp,
                    "current_state": x_current.tolist() if x_current is not None else None,
                    "predicted_trajectories": T_pred.tolist() if T_pred is not None else None,
                    "ac_decision": ac_first,
                    "damper_decisions": damper_first,
                    "ambient_temp": future_amb[0] if future_amb else None
                }
                
                # Save to file (ensure directory exists)
                os.makedirs("data/mpc_predictions", exist_ok=True)
                with open(f"data/mpc_predictions/pred_{timestamp.replace(':', '-')}.json", "w") as f:
                    json.dump(prediction_log, f, indent=2)
            
            return ac_first, damper_first, T_pred
        else:
            logger.warning(f"Optimization failed with status: {prob.status}")
            return 0, [0] * n, None
    except Exception as e:
        logger.error(f"Optimization solver error: {e}")
        return 0, [0] * n, None

def run_mpc_simulation(system, dt=300.0, horizon=12, sim_hours=6.0, bigM=50.0):
    """
    Run a closed-loop MPC simulation
    
    Parameters:
    - system: ThermalSystem object
    - dt: time step in seconds
    - horizon: number of steps in prediction horizon
    - sim_hours: total simulation time in hours
    - bigM: big-M value for logical constraints
    
    Returns:
    - times: array of simulation time points
    - T_store: array of temperatures for all zones at all time points
    - ac_log: array of AC on/off decisions at all time points
    - damper_log: array of damper open/closed decisions at all time points
    """
    n = len(system.zones)
    steps = int(round((sim_hours*3600)/dt))
    
    x_true = np.array([z.initial_temperature for z in system.zones])
    
    times = np.arange(0, steps*dt + dt, dt)
    T_store = np.zeros((n, steps+1))
    T_store[:,0] = x_true
    
    ac_log = np.zeros(steps, dtype=bool)
    damper_log = np.zeros((n, steps), dtype=bool)
    
    for k in range(steps):
        t_now = times[k]
        logger.info(f"Simulating step {k+1} of {steps} (time = {t_now/3600:.2f} hours)")
        
        # Generate future ambient predictions
        future_amb = [system.ambient_temperature((t_now + j*dt)/3600) for j in range(horizon)]
        
        # Run MPC optimization
        ac_first, damper_first, _ = optimize_ac_damper_control(
            system, x_true, future_amb, horizon, dt, bigM=bigM
        )
        
        # Log control actions
        ac_log[k] = (ac_first == 1)
        for i in range(n):
            damper_log[i, k] = (damper_first[i] == 1)
        
        # Print damper states
        logger.info(f"Damper states at t = {t_now/3600:.2f} hours:")
        for i in range(n):
            state = "open" if damper_first[i] == 1 else "closed"
            logger.info(f"  {system.zones[i].name}: {state}")
        
        # Apply cooling based on control actions
        cooling_vec = np.zeros(n)
        if ac_first == 1:
            for i in range(n):
                if damper_first[i] == 1:
                    cooling_vec[i] = -366.0  # Cooling power
        
        # Simulate system for one time step
        try:
            # Use scipy's ODE solver
            def dynamics_wrapper(t, y):
                return thermal_dynamics(t, y, system, cooling_vec)
            
            sol = solve_ivp(
                dynamics_wrapper,
                (t_now, t_now+dt),
                x_true,
                method='RK45',
                t_eval=[t_now+dt]
            )
            
            x_true = sol.y[:, -1]
        except Exception as e:
            logger.warning(f"ODE solver failed at step {k}: {e}")
            # Fallback to forward Euler if ODE solver fails
            dt_small = dt / 10
            x_temp = x_true.copy()
            for j in range(10):
                dT = thermal_dynamics(t_now + (j-1)*dt_small, x_temp, system, cooling_vec)
                x_temp += dT * dt_small
            x_true = x_temp
        
        # Store temperatures
        T_store[:,k+1] = x_true
    
    return times, T_store, ac_log, damper_log

def extract_state_space_matrices(system):
    """
    Extract state space matrices (A, B_cool, B_amb) from the thermal system
    
    Parameters:
    - system: ThermalSystem object
    
    Returns:
    - A: state matrix
    - B_cool: input matrix for cooling
    - B_amb: input matrix for ambient temperature
    """
    n = len(system.zones)
    
    A = np.zeros((n, n))
    B_cool = np.zeros((n, n))
    B_amb = np.zeros((n, 1))
    
    for i in range(n):
        Ri = system.zones[i].thermal_resistance
        Ci = system.zones[i].thermal_capacitance
        
        A[i, i] = -1 / (Ri * Ci)
        B_amb[i, 0] = 1 / (Ri * Ci)
        
        for j in range(n):
            if j != i and system.connections[i, j] > 0:
                A[i, j] = system.connections[i, j] / Ci
                A[i, i] -= system.connections[i, j] / Ci
        
        B_cool[i, i] = 1 / Ci
    
    return A, B_cool, B_amb

def analyze_system_properties(system):
    """
    Analyze system properties like stability and controllability
    
    Parameters:
    - system: ThermalSystem object
    
    Returns:
    - A, B_cool, B_amb: State space matrices
    """
    n = len(system.zones)
    
    # Extract state space matrices
    A, B_cool, B_amb = extract_state_space_matrices(system)
    
    # Display state space equations
    logger.info("State Space Equations for the Thermal System:")
    logger.info("--------------------------------------------")
    logger.info("The state vector x represents the temperatures of each zone:")
    for i in range(n):
        logger.info(f"x[{i}] = Temperature of zone {system.zones[i].name}")
    
    logger.info("\nThe input vector u represents the cooling power applied to each zone.")
    logger.info("The ambient temperature T_amb is an external input.")
    
    logger.info("\nContinuous-time state space model:")
    logger.info("dx/dt = A * x + B_cool * u + B_amb * T_amb")
    
    # Calculate discrete-time matrices
    dt = 300.0
    A_d = np.eye(n) + A * dt
    B_cool_d = B_cool * dt
    B_amb_d = B_amb * dt
    
    # Analyze stability
    logger.info("\nSystem Stability Analysis:")
    eigen_vals = linalg.eigvals(A)
    logger.info(f"Eigenvalues of A: {eigen_vals}")
    logger.info(f"Is the system stable? {'Yes (all eigenvalues have negative real parts)' if all(np.real(eigen_vals) < 0) else 'No'}")

    # Analyze controllability
    logger.info("\nControllability Analysis:")
    B_combined = np.hstack([B_cool[:, i].reshape(-1, 1) for i in range(n)])
    C = np.zeros((n, n*n))
    current_block = B_combined
    
    for i in range(n):
        C[:, (i*n):((i+1)*n)] = current_block
        current_block = A @ current_block
    
    controllability_rank = np.linalg.matrix_rank(C)
    logger.info(f"Rank of controllability matrix: {controllability_rank} (out of {n})")
    logger.info(f"Is the system fully controllable? {'Yes' if controllability_rank == n else 'No'}")

    # Analyze time constants
    logger.info("\nTime Constants Analysis:")
    time_constants = -1 / np.real(eigen_vals)
    logger.info(f"Time constants (seconds): {time_constants}")
    logger.info(f"Time constants (minutes): {time_constants / 60}")
    logger.info(f"Time constants (hours): {time_constants / 3600}")
    
    return A, B_cool, B_amb

def visualize_results(times, T_traj, ac_used, dampers, system):
    """Create visualizations of simulation results"""
    n = len(system.zones)
    
    # Create a figure with multiple subplots
    fig = plt.figure(figsize=(16, 16))
    fig.suptitle('Thermal System MPC Simulation Results', fontsize=16)
    
    # Temperature plots
    for i in range(n):
        plt.subplot(3, 3, i+1)
        plt.plot(times/3600, T_traj[i, :], label=system.zones[i].name)
        plt.axhline(y=75, linestyle='--', color='gray', label='75°F')
        plt.axhline(y=85, linestyle='--', color='red', label='85°F (Critical)')
        plt.xlabel('Time (hours)')
        plt.ylabel('Temperature (°F)')
        plt.legend()
        plt.title(f'Zone {system.zones[i].name} Temperature')
    
    # AC status plot
    plt.subplot(3, 3, 7)
    plt.plot(times[:-1]/3600, ac_used, 'k-', linewidth=2, label='AC on=1/off=0')
    plt.xlabel('Time (hours)')
    plt.ylabel('AC Status')
    plt.ylim(-0.1, 1.1)
    plt.yticks([0, 1])
    plt.legend()
    plt.title('AC Status')
    
    # Number of open dampers plot
    plt.subplot(3, 3, 8)
    num_open = np.sum(dampers, axis=0)
    max_dampers = len(system.zones)
    plt.plot(times[:-1]/3600, num_open, 'b-', linewidth=2, label='# Dampers Open')
    plt.xlabel('Time (hours)')
    plt.ylabel('Count')
    plt.ylim(0, max_dampers)
    plt.yticks(range(max_dampers+1))
    plt.legend()
    plt.title('Number of Open Dampers')
    
    # Combined damper states
    plt.subplot(3, 3, 9)
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange']
    for i in range(n):
        # Offset each damper state slightly for better visibility
        offset = i * 0.05
        plt.step(times[:-1]/3600, dampers[i, :] + offset, 
                where='post', color=colors[i % len(colors)],
                linewidth=2, label=f'{system.zones[i].name}')
    plt.xlabel('Time (hours)')
    plt.ylabel('Damper State')
    plt.ylim(-0.1, 1.5)
    plt.yticks([0, 1])
    plt.legend(loc='upper right', fontsize=8)
    plt.title('All Damper States')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Make room for suptitle
    plt.savefig("combined_thermal_plots.png")
    
    # Show the figure
    plt.show()

# Simple test to make sure everything works
def run_test():
    """Run a simple test of the thermal model"""
    # Create test temperature readings
    temps = {
        "TR1": 72.5,
        "TR2": 73.1,
        "TR3": 71.8,
        "TR4": 72.2,
        "BR1": 70.9,
        "BR2": 71.5,
        "BR3": 71.0,
        "BR4": 70.5,
        "Ambient": 68.0
    }
    
    # Create system and run optimization
    system = create_real_system(temps)
    ac_on, damper_states, _ = optimize_ac_damper_control(system)
    
    print(f"Test Results:")
    print(f"  AC: {'ON' if ac_on else 'OFF'}")
    print(f"  Dampers:")
    for i, name in enumerate(["TR1", "TR2", "TR3", "TR4", "BR1", "BR2", "BR3", "BR4"]):
        print(f"    {name}: {'OPEN' if damper_states[i] else 'CLOSED'}")
    
    return ac_on, damper_states

# If module is run directly, perform a test
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run test
    run_test()
