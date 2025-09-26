# import pyomo and json
import pyomo.environ as pyo
from pyomo.environ import floor
import json
from contextlib import redirect_stdout

# Load the JSON file using the json module
with open('network5.json', 'r') as file:
    data = json.load(file)

# Create Python variables for the input data
I = data['I']
J = data['J']
K = data['K']
delta = data['delta']
lastT = data['lastT']
I_i_k_minus = {tuple(item['index']): item['value'] for item in data['I_i_k_minus']}
I_i_k_plus = {tuple(item['index']): item['value'] for item in data['I_i_k_plus']}
rho_minus = {tuple(item['index']): item['value'] for item in data['rho_minus']}
rho_plus = {tuple(item['index']): item['value'] for item in data['rho_plus']}
I_i_j_prod = {tuple(item['index']): item['value'] for item in data['I_i_j_prod']}
tau_p = {tuple(item['index']): item['value'] for item in data['tau_p']}
tau = {tuple(item['index']): item['value'] for item in data['tau']}
beta_min = {tuple(item['index']): item['value'] for item in data['beta_min']}
beta_max = {tuple(item['index']): item['value'] for item in data['beta_max']}
cost = {tuple(item['index']): item['value'] for item in data['cost']}
gamma = {item['index']: item['value'] for item in data['gamma']}
S0 = {item['index']: item['value'] for item in data['S0']}
demand = {tuple(item['index']): item['value'] for item in data['demand']}

# Create a concrete model
m = pyo.ConcreteModel()

# Import the sets from the JSON file
m.I = pyo.Set(initialize=I, doc='Set of tasks')
m.J = pyo.Set(initialize=J, doc='Set of Units')
m.K = pyo.Set(initialize=K, doc='Set of states')

# Import the time range sets.
m.delta = pyo.Param(
    initialize=data['delta'],
    doc='length of time periods of discretized time grid [units of time]',
)
m.lastT = pyo.Param(initialize=data['lastT'], doc='last discrete time value')
m.T = pyo.RangeSet(0, m.lastT, 1, doc='Discrete time set')

# Initialize parameters I_i_k_minus and I_i_k_plus
m.I_i_k_minus = pyo.Param(
    m.I,
    m.K,
    initialize={tuple(item['index']): item['value'] for item in data['I_i_k_minus']},
    default=0,
    doc='State-task mapping: outputs from states',
)
m.I_i_k_plus = pyo.Param(
    m.I,
    m.K,
    initialize={tuple(item['index']): item['value'] for item in data['I_i_k_plus']},
    default=0,
    doc="Task-state mapping: inputs to states",
)

# Initialize rho_minus and rho_plus
m.rho_minus = pyo.Param(
    m.I,
    m.K,
    initialize={tuple(item['index']): item['value'] for item in data['rho_minus']},
    default=0,
    doc="Fraction of material in state k consumed by task i ",
)
m.rho_plus = pyo.Param(
    m.I,
    m.K,
    initialize={tuple(item['index']): item['value'] for item in data['rho_plus']},
    default=0,
    doc="Fraction of material in state k produced by task i ",
)

# Initialize I_i_j_prod
m.I_i_j_prod = pyo.Param(
    m.I,
    m.J,
    initialize={tuple(item['index']): item['value'] for item in data['I_i_j_prod']},
    default=0,
    doc="Unit-task mapping (Definition of units that are allowed to perform a given task",
)

m.tau_p = pyo.Param(
    m.I,
    m.J,
    initialize={tuple(item['index']): item['value'] for item in data['tau_p']},
    default=0,
    doc="Physical processing time for tasks [units of time]",
)
m.tau = pyo.Param(
    m.I,
    m.J,
    initialize={tuple(item['index']): item['value'] for item in data['tau']},
    default=0,
    doc="Processing time with respect to the time grid",
)

m.beta_min = pyo.Param(
    m.I,
    m.J,
    initialize={tuple(item['index']): item['value'] for item in data['beta_min']},
    default=0,
    doc="Minimum capacity of unit j for task i",
)

m.beta_max = pyo.Param(
    m.I,
    m.J,
    initialize={tuple(item['index']): item['value'] for item in data['beta_max']},
    default=0,
    doc="Maximum capacity of unit j for task i",
)

m.cost = pyo.Param(
    m.I,
    m.J,
    default=0,
    initialize={tuple(item['index']): item['value'] for item in data['cost']},
    doc='cost to run task i in unit j',
)

m.gamma = pyo.Param(
    m.K,
    initialize={item['index']: item['value'] for item in data['gamma']},
    default=0,
    doc="maximum amount of material k that can be stored",
)

m.S0 = pyo.Param(
    m.K,
    initialize={item['index']: item['value'] for item in data['S0']},
    default=0,
    doc="Initial amount of state k",
)

m.demand = pyo.Param(
    m.K,
    m.T,
    initialize={tuple(item['index']): item['value'] for item in data['demand']},
    default=0,
    doc="Demand of material k at time t",
)

# Define Variables
m.X = pyo.Var(
    m.I,
    m.J,
    m.T,
    within=pyo.Binary,
    initialize=0,
    doc='Task i is assigned to unit j at time t',
)

m.B = pyo.Var(
    m.I,
    m.J,
    m.T,
    within=pyo.NonNegativeReals,
    initialize=0,
    doc='Batch size of task i processed in unit j starting at time t',
)

# Define the variable m.S with a dynamic upper bound set by m.gamma
def _S_bounds(m, K, T):
    return (0, m.gamma[K])  # Lower bound is 0, upper bound is m.gamma[K]

m.S = pyo.Var(
    m.K,
    m.T,
    within=pyo.NonNegativeReals,
    bounds=_S_bounds,  # Apply the dynamic bounds
    initialize=0,
    doc='Inventory of material k at time t',
)


# New Module from Maravelias Paper
m.N = pyo.Var(m.I, m.J, within=pyo.NonNegativeIntegers, initialize=0, doc='Total number of times task i runs in unit j')

def _E4_Binary_Reformulation(m, I, J):
    # Apply the constraint only where the task can be performed on the unit
    if m.I_i_j_prod[I, J] == 1:
        return sum(m.X[I, J, T1] for T1 in m.T) == m.N[I, J]
    else:
        return pyo.Constraint.Skip  # Skip the constraint if the task cannot be performed on the unit

m.E4 = pyo.Constraint(m.I, m.J, rule=_E4_Binary_Reformulation, doc='Total number of times task i runs in unit j')


def _E5_N_Boundary(m, I, J):
    # Apply the constraint only where the task can be performed on the unit
    if m.I_i_j_prod[I, J] == 1: # and m.tau_p[I, J] > 0:
        return m.N[I, J] <= floor(120 / m.tau_p[I, J])
    else:
        return pyo.Constraint.Skip  # Skip if the task cannot be performed on the unit or tau_p is zero

m.E5 = pyo.Constraint(m.I, m.J, rule=_E5_N_Boundary, doc='Limit on number of task runs in a unit')



# Define Constraint
def _E1_UNIT(m, J, T):
    return (
        sum(
            sum(
                m.X[I, J, TP] for TP in m.T if TP <= T and TP >= T - m.tau_p[I, J] + 1 
            )
            for I in m.I
            if m.I_i_j_prod[I, J] == 1
        )
        <= 1
    )

m.E1_UNIT = pyo.Constraint(m.J, m.T, rule=_E1_UNIT, doc='unit utilization constraint')


def _E2_CAPACITY_LOW(m, I, J, T):
    if m.I_i_j_prod[I, J] != 1:
        return pyo.Constraint.Skip
    else:
        return m.beta_min[I, J] * m.X[I, J, T] <= m.B[I, J, T]

m.E2_CAPACITY_LOW = pyo.Constraint(
    m.I, m.J, m.T, rule=_E2_CAPACITY_LOW, doc='unit capacity lower bound'
)

def _E2_CAPACITY_UP(m, I, J, T):
    if m.I_i_j_prod[I, J] != 1:
        return pyo.Constraint.Skip
    else:
        return m.B[I, J, T] <= m.beta_max[I, J] * m.X[I, J, T]

m.E2_CAPACITY_UP = pyo.Constraint(
    m.I, m.J, m.T, rule=_E2_CAPACITY_UP, doc='unit capacity upper bound'
)

def _E3_BALANCE(m, K, T):
    # Production finishing at T: comes from batches started at (T - tau[i,j])
    prod = sum(
        m.rho_plus[I, K] *
        sum(
            m.B[I, J, T - m.tau[I, J]]
            for J in m.J
            if m.I_i_j_prod[I, J] == 1 and (T - m.tau[I, J]) > 0
        )
        for I in m.I if m.I_i_k_plus[I, K] == 1
    )

    # Consumption at T: batches that start at T
    cons = sum(
        m.rho_minus[I, K] *
        sum(m.B[I, J, T] for J in m.J if m.I_i_j_prod[I, J] == 1)
        for I in m.I if m.I_i_k_minus[I, K] == 1
    )

    if T == 0:
        return m.S[K, 0] == m.S0[K] + 0 - cons - m.demand[K, 0]
    else:
        return m.S[K, T] == m.S[K, T-1] + prod - cons - m.demand[K, T]

m.E3_BALANCE = pyo.Constraint(m.K, m.T, rule=_E3_BALANCE, doc="material balance with discrete production lag")


# objective - cost minimization
# def _obj(m):
#     return sum(
#         sum(sum(m.cost[I, J] * m.X[I, J, T] for J in m.J) for I in m.I) for T in m.T
#     )

# Reformulation of the objective function
def _obj(m):
    return sum(m.cost[I, J] * m.N[I, J] for I in m.I for J in m.J)

m.obj = pyo.Objective(rule=_obj, sense=pyo.minimize)


# Solve the model
solver = pyo.SolverFactory('gurobi')
solver.solve(m, tee=True)

# Print the objective value
print("Objective value: ", pyo.value(m.obj))




print(demand)
m.demand.pprint()