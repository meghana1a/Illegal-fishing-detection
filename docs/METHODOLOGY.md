# Technical Methodology

## System Architecture

This project implements a **two-unit integrated detection system**:

```
UNIT 1: AIS BEHAVIOR ANALYSIS          UNIT 2: TELEMETRY ANALYSIS
┌─────────────────────────┐            ┌──────────────────────────┐
│ AIS Vessel Tracking     │            │ Shark Acoustic Telemetry │
│ (MMSI, position, speed) │            │ (Location, timestamp)    │
└────────────┬────────────┘            └────────────┬─────────────┘
             │                                      │
             ↓                                      ↓
    ┌────────────────────┐            ┌──────────────────────┐
    │ Data Preprocessing │            │ Anomaly Detection    │
    │ - Distance calc    │            │ - Isolation Forest   │
    │ - Heading calc     │            │ - Movement patterns  │
    │ - AR imputation    │            │ - Loitering events   │
    └────────────┬───────┘            └──────────────┬───────┘
                 │                                   │
                 ↓                                   ↓
    ┌────────────────────┐            ┌──────────────────────┐
    │ 1D CNN Classifier  │            │ Neural Network       │
    │ - Conv1D layer     │            │ - Feature integration│
    │ - MaxPooling       │            │ - Classification     │
    │ - Dense output     │            │ - Risk scoring       │
    └────────────┬───────┘            └──────────────┬───────┘
                 │                                   │
                 └───────────────┬────────────────────┘
                                 ↓
                    ┌────────────────────────┐
                    │ Decision Support       │
                    │ - Risk aggregation     │
                    │ - Spatial/temporal viz │
                    │ - Enforcement priority │
                    └────────────────────────┘
```

---

## Unit 1: 1D CNN for AIS Vessel Behavior Detection

### Problem Statement
Detect whether a fishing vessel is actively fishing based on its movement patterns in AIS data. This enables risk-based surveillance where enforcement resources can focus on vessels most likely to be engaged in illegal fishing.

### Data Flow

**1. Data Loading**
```python
# Load AIS data by vessel type
df = pd.read_csv('ais_vessel_data/trawlers.csv')
# Schema: mmsi, timestamp, lat, lon, speed, course, 
#         distance_from_shore, distance_from_port, is_fishing
```

**2. Feature Engineering**

**Haversine Distance Calculation:**
```
distance = 2R × arcsin(√[sin²(Δlat/2) + cos(lat1)cos(lat2)sin²(Δlon/2)])

Where:
- R = Earth's radius (6,371 km)
- Δlat = latitude difference
- Δlon = longitude difference
```

Calculates great-circle distance between consecutive AIS points.

**Heading/Bearing Calculation:**
```
heading = atan2(sin(Δlon)cos(lat2), 
                cos(lat1)sin(lat2) - sin(lat1)cos(lat2)cos(Δlon))
heading = (heading + 360) % 360  // normalize to [0, 360)
```

Determines compass direction of vessel movement.

**3. Missing Value Imputation**

**AutoRegressive (AR) Model:**
```
For each numeric column:
  1. Fit AR(1) model: X_t = c + φ₁X_{t-1} + ε_t
  2. Predict missing values using model
  3. Fill NaN positions with predictions

Parameters:
  - Lag: 1 (previous value predicts current)
  - Method: statsmodels.tsa.ar_model.AutoReg
```

Why AR?
- Respects temporal dependencies in vessel data
- Preserves speed/course patterns
- Better than static imputation (mean, median)
- Single lag adequate for AIS (short-term patterns)

**4. Feature Normalization**

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)
# Transforms: X_scaled = (X - mean) / std_dev
# Result: Zero mean, unit variance
```

Why normalize?
- 1D CNN requires similar feature scales
- Speed: 0-25 knots, Distance: 0-1000+ km
- Prevents features with large ranges from dominating

**5. Sequence Preparation for LSTM/CNN**

```
Original features (per vessel):
[feature1, feature2, ... feature10]  (1D vector)

Reshape for Conv1D:
[[feature1], [feature2], ... [feature10]]  (2D: samples × features)

Grouping by MMSI:
Each vessel's timesteps become one training sequence
Pad/truncate to consistent length
```

**6. 1D CNN Model**

```
Architecture:
  Input: (batch_size, num_features, 1)
    ↓
  Conv1D(filters=64, kernel_size=3, activation='relu')
    - Learns 64 different 3-element patterns
    - Captures local temporal features (speed changes, etc.)
    Output shape: (batch_size, features-2, 64)
    ↓
  MaxPooling1D(pool_size=2)
    - Reduces dimension by 50%
    - Keeps strongest activations
    Output shape: (batch_size, (features-3)//2, 64)
    ↓
  Flatten()
    - Convert 3D → 1D for dense layers
    Output shape: (batch_size, flattened_size)
    ↓
  Dense(2, activation='softmax')
    - Binary classification: [P(not_fishing), P(fishing)]
    - Softmax: probabilities sum to 1
    Output: [0.95, 0.05] or [0.20, 0.80]
```

**Why 1D CNN?**
- Exploits sequential structure of time series
- Learns hierarchical patterns (low-level: speed changes, high-level: fishing behavior)
- Faster than LSTM, fewer parameters
- Excellent for sensor/tracking data

**7. Training**

```python
Hyperparameters:
  - Loss: sparse_categorical_crossentropy (integer labels)
  - Optimizer: Adam (learning_rate=0.005)
  - Epochs: 10
  - Batch size: 32
  - Validation split: 0.2

Training loop:
  1. Forward pass: predictions = model(X_train)
  2. Compute loss: L = -Σ[y_true × log(y_pred)]
  3. Backward pass: compute gradients via backprop
  4. Update weights: w_new = w_old - lr × gradient
  5. Repeat 10 times through entire dataset
```

**Results:**
- Training accuracy: ~98%
- Validation accuracy: **98.38%**
- Loss: 18.44%

### Output & Interpretation

```python
# Predictions on validation set
predictions = model.predict(X_val)
# Shape: (num_samples, 2)
# Example: [[0.98, 0.02], [0.15, 0.85], ...]

# Extract fishing vessels
fishing_threshold = 0.5
fishing_mmsi = X_val[predictions[:, 1] >= fishing_threshold]['mmsi'].unique()
# Result: MMSIs of vessels predicted to be fishing
```

---

## Unit 2: Isolation Forest + Neural Network for Telemetry Analysis

### Problem Statement
Detect interactions between illegal fishing vessels and marine protected species (sharks) by identifying anomalous vessel behavior near acoustic detection arrays.

### Stage 1: Anomaly Detection with Isolation Forest

**Why Isolation Forest?**
- Efficient for high-dimensional data
- No distance calculations (unlike KNN)
- Naturally handles isolated points (anomalies are rare)
- Outliers often isolated quickly → shorter path lengths
- Robust to feature scaling

**Algorithm:**

```
1. Randomly select a feature and split value
2. Recursively partition data until isolated
3. Anomalies isolated in fewer splits (shorter trees)
4. Anomaly score = average path length

For each point:
  score = 2^(-average_depth / c(n))
  Where c(n) = average depth of unsuccessful searches
  
  If score → 1: Anomaly
  If score → 0: Normal
```

**Parameters:**
```python
IsolationForest(
    contamination=0.05,  # Expect 5% anomalies
    random_state=42,     # Reproducibility
    n_estimators=100     # Number of trees
)
```

**Application to Shark Telemetry:**

```
Input features: [longitude, latitude, speed]
Dataset size: 356,000 detection points

Results:
  - Anomalies detected: 17,819 (4.99%)
  - Average anomaly score: -0.0639
  - Normal points: 338,181 (95.01%)
```

### Stage 2: Loitering Detection

**Hypothesis:** Illegal fishing vessels slow down or stop when actively fishing or committing poaching.

```
Loitering_threshold = 2 knots (nautical miles/hour)
                    = 3.7 km/h
                    = Very slow movement

Filter: anomalies[anomalies['speed'] < 2]

Results:
  - Total anomalies: 17,819
  - Loitering events: 222 (1.2% of anomalies)
  - Duration: Minutes to hours at same location
```

### Stage 3: Spatial-Temporal Correlation

**Objective:** Match loitering vessel events with shark detection locations

```
For each shark detection:
  1. Get location (lat, lon) and timestamp
  2. Define buffer: ±1 hour time window
  3. Find vessels loitering nearby in that period
  4. Calculate distance to nearest receiver/animal
  5. Store matches
```

**Results:**
```
Shark detections: N points
Loitering events: 222 points
Matches found: X correlations

Each match indicates:
- Vessel present near protected animal
- Temporal overlap (likely coincidence if far apart)
- Suspicious behavior (loitering vs. normal transit)
```

### Stage 4: Neural Network Classification

**Model Purpose:** Predict probability of illegal fishing given:
- Shark telemetry anomalies
- Vessel loitering events
- Environmental factors (optional)
- Historical IUU incident data (when available)

**Architecture:**

```
Input Features (dimension: 20-30)
  ↓
Dense(128, activation='relu')
  - 128 neurons with ReLU
  - Learns complex patterns
  ↓
Dropout(0.3)
  - Randomly disable 30% of neurons during training
  - Reduces overfitting
  ↓
Dense(64, activation='leaky_relu')
  - 64 neurons with LeakyReLU
  - Allows small gradients for dead neurons
  - Non-zero slope when x < 0
  ↓
Dense(32, activation='elu')
  - 32 neurons with ELU (Exponential Linear Unit)
  - Smooth gradient, learns faster
  ↓
Dense(1, activation='sigmoid')
  - Binary output: P(illegal_fishing)
  - Sigmoid: smooth, differentiable, output ∈ [0, 1]
```

**Activation Functions:**
- **ReLU:** f(x) = max(0, x) - Fast, sparse activation
- **LeakyReLU:** f(x) = max(0.01x, x) - Prevents dead neurons
- **ELU:** f(x) = x if x>0, α(e^x - 1) if x≤0 - Smoother gradient
- **Sigmoid:** f(x) = 1/(1+e^(-x)) - Probability output

**Regularization:**

```python
regularizer = L1L2(l1=0.01, l2=0.01)
# L1: Sparse weights (feature selection)
# L2: Small weights (smooth model)
# Equation: Loss = MSE + 0.01×Σ|w| + 0.01×Σ(w²)
```

**Class Weighting:**

```python
class_weight = {
    0: 1.0,           # Normal events (common)
    1: 10.0           # IUU events (rare, 1:10 ratio)
}
# Penalizes misclassifying rare IUU events more heavily
```

**Training:**

```python
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',  # For binary classification
    metrics=['accuracy']
)

model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    class_weight=class_weight
)
```

### Output & Decision Making

```
Predicted probability: 0.78
Interpretation:
  - 78% confidence this is illegal fishing event
  - Threshold = 0.5: Classified as POSITIVE
  - Risk level: HIGH (> 0.7)
  
Decision:
  ✓ Flag for further investigation
  ✓ Prioritize for enforcement patrol
  ✓ Cross-reference with other risk factors
```

---

## Data Integration & Decision Framework

### Combining Unit 1 & Unit 2

```
Vessel identified as FISHING (Unit 1)
  ↓
Is vessel in marine protected area?
  ↓
Does vessel show LOITERING behavior (Unit 2)?
  ↓
Positive match with SHARK DETECTION?
  ↓
Calculate RISK SCORE:
  Risk = α×(AIS_confidence) + β×(Telemetry_anomaly) + γ×(Correlation)
  Where α + β + γ = 1.0

Risk → Enforcement Priority
  High (>0.7): Immediate patrol/inspection
  Medium (0.4-0.7): Monitor closely, intercept if opportunity
  Low (<0.4): Background monitoring
```

### Spatial-Temporal Indexing

```
Grid: 0.1° × 0.1° cells (≈11 km × 11 km)
Time: Daily aggregation

For each cell × day:
  - Count fishing vessels (Unit 1)
  - Sum anomaly points (Unit 2)
  - List shark detections
  - Correlation strength
  → Heat map of risk

Enforcement can target high-risk cells geographically
```

---

## Model Validation & Performance

### Unit 1 Validation Strategy

```
Train/Val/Test Split: 80/10/10
  - Training: Optimize weights
  - Validation: Monitor overfitting, tune hyperparameters
  - Test: Final performance assessment

Metrics:
  - Accuracy: Correct predictions / Total
  - Precision: TP / (TP + FP) - False positive rate
  - Recall: TP / (TP + FN) - False negative rate
  - F1: Harmonic mean of Precision & Recall

Result: 98.38% accuracy indicates excellent vessel classification
```

### Unit 2 Validation Strategy

```
Challenges:
  - Imbalanced classes (few IUU events)
  - Limited ground truth data
  - Temporal dependencies

Solutions:
  - Class weighting
  - Anomaly detection (unsupervised baseline)
  - Expert review of flagged events
  - Cross-validation with held-out time periods
```

---

## Computational Considerations

### Memory Usage
```
Trawlers.csv: 497M
  × 10 features after preprocessing
  × float32 (4 bytes)
  ≈ 20 GB if fully loaded

Solutions:
  - Load with chunksize parameter
  - Process per-MMSI groups
  - Use sparse matrices where applicable
```

### Runtime
```
Demo_Model training: ~15-30 minutes on GPU
  - Depends on batch size, epochs, hardware
  - CPU training: 2-4 hours
  - GPU acceleration: 10-20× faster

new2.ipynb telemetry analysis: ~5-10 minutes
  - Isolation Forest: O(n log n) complexity
  - Neural network training: O(samples × features)
```

### Scalability
```
Current: ~5 million AIS points per vessel type
Scaled: Billions of points across all vessels/years

Optimization:
  - Distributed processing (Spark, Dask)
  - Online learning (streaming data)
  - Model quantization (reduce precision)
  - Feature selection (reduce dimensionality)
```

---

## Future Methodological Improvements

1. **Multimodal Learning:** Combine AIS, VMS, satellite, social media signals
2. **Transfer Learning:** Pretrain on one region, adapt to another
3. **Ensemble Methods:** Stack Unit 1 + Unit 2 + external models
4. **Active Learning:** Prioritize which events to label for retraining
5. **Causal Inference:** Distinguish correlation from causation
6. **Adversarial Robustness:** Test against evasion tactics

