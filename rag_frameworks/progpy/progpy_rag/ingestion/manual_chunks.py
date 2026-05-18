"""
ingestion/manual_chunks.py

Manually curated chunks for critical ProgPy APIs that the auto-crawler
consistently truncates or misses. These are added during every ingestion run.

When to add a manual chunk:
- The auto-crawler truncates a critical method mid-body
- A usage pattern requires combining info from multiple source locations
- The correct pattern is non-obvious and Claude consistently hallucinates it
"""

from .github_crawler import RawChunk

MANUAL_CHUNKS_PHYSICS = [
    # ------------------------------------------------------------------
    # UnscentedKalmanFilter — correct usage pattern
    # ------------------------------------------------------------------
    RawChunk(
        text="""
# UnscentedKalmanFilter — complete correct usage pattern
from progpy.state_estimators import UnscentedKalmanFilter

# --- CONSTRUCTION ---
# Required: model + x0. Noise as dicts keyed by state/output name.
filt = UnscentedKalmanFilter(
    model,                                        # PrognosticsModel instance
    x0,                                           # initial state: dict or StateContainer
    process_noise={"state_key": 1e-8},            # dict keyed by model.states
    measurement_noise={"output_key": 0.1},        # dict keyed by model.outputs
)

# --- ONE STEP OF ESTIMATION ---
# Call once per timestep. t must always be greater than previous t.
filt.estimate(
    t,    # float: current time (strictly increasing)
    u,    # InputContainer: model.InputContainer({'input_key': value})
    z,    # OutputContainer: model.OutputContainer({'output_key': value})
)

# --- READING STATE ---
state_mean = filt.x.mean          # dict-like: {'state_key': float_value}
state_dist = filt.x               # MultivariateNormalDist (.mean, .cov)

# --- CRITICAL: WHAT DOES NOT EXIST ---
# filt.predict()       → AttributeError — does not exist
# filt.update()        → AttributeError — does not exist  
# filt.initialize()    → AttributeError — does not exist
# filt.P               → AttributeError — does not exist
# filt.x['key']        → Wrong — filt.x is UncertainData, use filt.x.mean['key']

# --- NOISE MATRIX RULES ---
# Process noise (Q): how much states change between steps
#   Degradation states (wear, buildup): ~1e-8 to 1e-10 (changes slowly)
#   Pass-through states: ~1e-4 (tracks inputs quickly)
# Measurement noise (R): sensor variance from real data
#   Temperature: ~1.0-4.0, Pressure: ~0.01-0.1, Time: ~0.5-1.0
# Rule: R should be >> Q for slowly degrading systems

# --- COMPLETE WORKING EXAMPLE ---
from progpy.models import ThrownObject
from progpy.state_estimators import UnscentedKalmanFilter

m = ThrownObject(process_noise=0, measurement_noise=0)
x0 = m.initialize()

filt = UnscentedKalmanFilter(
    m, x0,
    process_noise={"x": 1e-4, "v": 1e-4},
    measurement_noise={"x": 0.1},
)

u = m.InputContainer({})
z = m.OutputContainer({"x": 7.5})

filt.estimate(0.1, u, z)
print(filt.x.mean)  # {'x': float, 'v': float}
""",
        source="manual:ukf_usage_pattern",
        chunk_type="example",
        name="UnscentedKalmanFilter_correct_usage",
        domain="general",
        pattern="component",
    ),

    # ------------------------------------------------------------------
    # CompositeModel parameter access — correct pattern
    # ------------------------------------------------------------------
    RawChunk(
        text="""
# CompositeModel parameter access — correct patterns

from progpy import CompositeModel

composite = CompositeModel(
    [('Grinder', grinder), ('Brewer', brewer)],
    connections=[('Grinder.grind_size_mm', 'Brewer.grind_size_mm')]
)

# --- READ/WRITE via dot-notation string key ---
value = composite['Grinder.wear_rate']           # read
composite['Grinder.wear_rate'] = 1e-5            # write

# --- GET COMPONENT MODEL OBJECT ---
models_dict = dict(composite.parameters['models'])
grinder_model = models_dict['Grinder']
grinder_model.parameters['wear_rate'] = 1e-5     # direct parameter access

# --- WHAT EXISTS ON COMPOSITE ---
composite.inputs    # unconnected inputs only, namespaced: 'Grinder.input_key'
composite.states    # ALL states, namespaced: 'Grinder.state_key'
composite.outputs   # ALL outputs, namespaced: 'Grinder.output_key'
composite.events    # ALL events, namespaced: 'Grinder.EventName'

# --- WHAT DOES NOT EXIST ---
# composite.connections    → AttributeError — does not exist
# composite.models         → use composite.parameters['models'] instead

# --- FUTURE LOADING FOR COMPOSITE ---
# Only pass UNCONNECTED inputs (those in composite.inputs)
def future_loading(t, x=None):
    return composite.InputContainer({
        'Grinder.grinder_position': 2,      # unconnected input
        'Water.ambient_temp_c': 22.0,        # unconnected input
        # DO NOT include connected inputs — they come from component outputs
    })
""",
        source="manual:composite_parameter_access",
        chunk_type="example",
        name="CompositeModel_parameter_access",
        domain="general",
        pattern="composite",
    ),

    # ------------------------------------------------------------------
    # estimate_params — correct usage for component calibration
    # ------------------------------------------------------------------
    RawChunk(
        text="""
# estimate_params — correct usage for calibrating component parameters

# Basic usage
component.estimate_params(
    times=times,        # list of floats (time in hours)
    inputs=inputs,      # list of dicts, one per timestep, unnamespaced keys
    outputs=outputs,    # list of dicts, one per timestep, unnamespaced keys
    keys=['wear_rate', 'base_temp'],   # which params to fit
    bounds={
        'wear_rate': (1e-8, 1e-3),    # must be positive
        'base_temp': (80.0, 100.0),   # physical range
    },
    dt=1.0,             # timestep in hours
    method='nelder-mead',
)

# CRITICAL: estimate_params is called on the COMPONENT, not the composite
# For composite models, get the component first:
models_dict = dict(composite.parameters['models'])
grinder = models_dict['Grinder']
grinder.estimate_params(times=times, inputs=inputs, outputs=outputs, ...)

# inputs and outputs must use UNNAMESPACED keys (component's own keys)
# NOT 'Grinder.grind_size_mm' — just 'grind_size_mm'

# times must be monotonically increasing floats
# Convert from datetime: df['time_hours'] = (df['ts'] - df['ts'].min()).dt.total_seconds() / 3600
""",
        source="manual:estimate_params_usage",
        chunk_type="example",
        name="estimate_params_correct_usage",
        domain="general",
        pattern="component",
    ),
]
MANUAL_CHUNKS_DATA_DRIVEN = [
    # ------------------------------------------------------------------
    # LSTMStateTransitionModel — correct usage pattern
    # ------------------------------------------------------------------
    RawChunk(
        text="""
# LSTMStateTransitionModel — correct usage pattern
from progpy.data_models import LSTMStateTransitionModel

# --- TRAINING ---
# inputs and outputs must be lists of np.ndarray, shape (n_timesteps, n_features)
# Each element in the list = one run of data
import numpy as np

hi_array = hi_series.values.reshape(-1, 1).astype(np.float64)  # shape (n, 1)

model = LSTMStateTransitionModel.from_data(
    inputs=[hi_array],              # list of arrays — NOT a dict, NOT a DataFrame
    outputs=[hi_array],             # autoregressive: predicts next HI from past HI
    input_keys=['health_index'],    # must match number of input columns
    output_keys=['health_index'],   # must match number of output columns
    window=10,                      # NOT sequence_length — use window
    epochs=50,
    validation_percentage=0.2,      # NOT validation_split
    early_stopping=True,
)

# --- CRITICAL: DO NOT pass time as an input ---
# time_array as input causes time to become a lagged state variable
# (e.g., health_index_t-9) which breaks all filter initialization

# --- SAVE / LOAD ---
model.save('path/to/model_dir')
model = LSTMStateTransitionModel.load('path/to/model_dir')

# --- WHAT DOES NOT EXIST ---
# LSTMStateTransitionModel.from_data(sequence_length=...)  → KeyError
# LSTMStateTransitionModel.from_data(validation_split=...) → KeyError
# LSTMStateTransitionModel.from_data(times=...)            → wrong format

# --- CRITICAL: FILTER INCOMPATIBILITY ---
# DO NOT use UnscentedKalmanFilter or ParticleFilter with LSTMStateTransitionModel
# The LSTM creates lagged state variables internally (health_index_t-9, etc.)
# These cannot be initialized via x0 — filters will raise KeyError on construction
# Instead: use the batch HI as the observation directly in a plain Kalman fusion
""",
        source="manual:lstm_state_transition_usage",
        chunk_type="example",
        name="LSTMStateTransitionModel_correct_usage",
        domain="general",
        pattern="component",
    ),

    # ------------------------------------------------------------------
    # Data-driven state estimation — correct pattern without ProgPy filters
    # ------------------------------------------------------------------
    RawChunk(
        text="""
# Data-driven state estimation — correct pattern
# DO NOT use ProgPy UKF/PF with LSTMStateTransitionModel — they are incompatible
# Use this plain Kalman fusion pattern instead

import numpy as np

class ComponentStateEstimator:
    \"\"\"
    Kalman-style fusion of batch HI observation with learned trend prediction.
    State: current_hi (scalar). Observation: batch HI from health_index.py.
    \"\"\"
    def __init__(self, component_name, process_noise=0.001, measurement_noise=0.05):
        self.component_name = component_name
        self.process_noise = process_noise       # Q: trust in prediction
        self.measurement_noise = measurement_noise  # R: trust in observation
        self.P = 0.1                             # state error covariance
        self.current_hi = None
        self.hi_history = []                     # unscaled HI values
        self.degradation_rate = 0.0

    def initialize(self, initial_hi: float):
        self.current_hi = float(np.clip(initial_hi, 0.0, 1.0))
        self.hi_history = [self.current_hi]
        self.P = 0.1

    def update(self, observed_hi: float) -> dict:
        \"\"\"
        observed_hi: batch HI from health_index.py for this timestep (the measurement)
        \"\"\"
        observed_hi = float(np.clip(observed_hi, 0.0, 1.0))

        # Predict step: linear trend from recent history
        if len(self.hi_history) >= 2:
            trend = self.hi_history[-1] - self.hi_history[-2]
            predicted_hi = float(np.clip(self.current_hi + trend, 0.0, 1.0))
        else:
            predicted_hi = self.current_hi

        P_pred = self.P + self.process_noise

        # Update step: fuse prediction with observation
        K = P_pred / (P_pred + self.measurement_noise)  # Kalman gain
        self.current_hi = predicted_hi + K * (observed_hi - predicted_hi)
        self.current_hi = float(np.clip(self.current_hi, 0.0, 1.0))
        self.P = (1 - K) * P_pred

        # Track history
        self.hi_history.append(self.current_hi)
        if len(self.hi_history) > 50:
            self.hi_history.pop(0)

        # Degradation rate from recent slope
        if len(self.hi_history) >= 5:
            slope = np.polyfit(range(5), self.hi_history[-5:], 1)[0]
            self.degradation_rate = max(0.0, -slope)

        ci = 1.96 * np.sqrt(self.P)
        return {
            'health_index': float(self.current_hi),
            'confidence_lower': float(np.clip(self.current_hi - ci, 0, 1)),
            'confidence_upper': float(np.clip(self.current_hi + ci, 0, 1)),
            'degradation_rate': float(self.degradation_rate),
            'uncertainty': float(np.sqrt(self.P))
        }

# --- HOW TO USE IN THE ONLINE LOOP ---
# estimator.initialize(val_hi[component].iloc[0])
# for i in range(len(val_features)):
#     observed_hi = val_hi[component].iloc[i]   # batch HI = measurement
#     result = estimator.update(observed_hi)
""",
        source="manual:data_driven_state_estimation_pattern",
        chunk_type="example",
        name="data_driven_state_estimation_correct_pattern",
        domain="general",
        pattern="component",
    ),

]

MANUAL_CHUNKS = MANUAL_CHUNKS_PHYSICS

