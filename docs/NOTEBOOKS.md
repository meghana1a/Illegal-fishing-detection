# Notebook Documentation

## Overview

This project contains 5 Jupyter notebooks implementing two complementary illegal fishing detection systems. Each notebook has a specific purpose in the research pipeline.

---

## 🎯 Production Notebooks

### `Demo_Model.ipynb` ⭐ MAIN MODEL

**Purpose:** Production implementation of Unit 1 - AIS-based fishing vessel detection

**What it does:**
1. Loads trawler AIS data (vessel positions, speeds, courses)
2. Preprocesses data: calculates distances traveled and heading angles
3. Fills missing values using AutoRegressive (AR) time series models
4. Trains a 1D CNN model for fishing/non-fishing classification
5. Achieves **98.38% accuracy** on validation set
6. Outputs predictions and identified fishing vessel MMSIs

**Key Performance Metrics:**
- Accuracy: 98.38%
- Loss: 18.44%
- Epochs: 10
- Batch Size: 32
- Learning Rate: 0.005

**Model Architecture:**
```
Input (features: mmsi, lat, lon, speed, course, distance, heading, etc.)
    ↓
Conv1D (64 filters, kernel size 3, ReLU activation)
    ↓
MaxPooling1D (pool size 2)
    ↓
Flatten
    ↓
Dense (2 units, softmax) → [not_fishing_prob, fishing_prob]
```

**Data Used:**
- `data/ais_vessel_data/trawlers.csv` - Primary input
- `data/training_data/training_classes.csv` - Labels
- `data/training_data/real_data.csv` - Preprocessed features

**Output:**
- Model predictions for each vessel
- List of MMSIs identified as fishing
- Accuracy/loss metrics
- Model weights saved

**When to Use:**
- Production predictions on new AIS data
- Baseline model for comparison
- Demonstrating 1D CNN effectiveness on sequential vessel data

**Dependencies:**
```python
pandas, numpy, scikit-learn, tensorflow/keras, geopy
```

---

### `new2.ipynb` ⭐ TELEMETRY ANALYSIS

**Purpose:** Unit 2 - Shark telemetry anomaly detection and illegal fishing correlation

**What it does:**
1. Loads shark acoustic telemetry detections
2. Applies Isolation Forest to detect anomalies in movement patterns
3. Identifies loitering events (ships moving slowly, < 2 knots)
4. Correlates shark detection locations with vessel positions
5. Matches timestamps to find potential interactions
6. Trains neural network to classify suspicious events
7. Outputs risk scores for illegal fishing activity

**Key Components:**

**Anomaly Detection (Isolation Forest):**
- Contamination: 5%
- Features: Longitude, latitude, speed
- Detects: 17,819 anomalies (~5% of 356,000 points)
- Average anomaly score: -0.0639

**Loitering Detection:**
- Speed threshold: < 2 knots
- Identifies: 222 loitering events among anomalies
- Spatial correlation with shark detections

**Neural Network Classification:**
```
Input (telemetry + environmental features)
    ↓
Dense (ReLU activation) + Dropout
    ↓
Dense (LeakyReLU activation) + L1/L2 regularization
    ↓
Dense (ELU activation) + Regularization
    ↓
Dense (1 unit, sigmoid) → [IUU probability]
```

**Class Weighting:** Handles imbalanced data (few IUU events vs. normal activity)

**Data Used:**
- `data/telemetry_data/labelled_shark_data.csv` - Shark detections
- `data/telemetry_data/IMOS_-_Animal_Tracking...csv` - Receiver locations
- Historical IUU incident data (when available)

**Output:**
- Anomaly scores for each telemetry event
- Identified loitering events
- Probability of illegal fishing per location
- Shark-vessel interaction matches

**When to Use:**
- Analyzing shark/marine life vulnerability to fishing
- Detecting suspicious vessel behavior near protected areas
- Research on effectiveness of marine protected areas
- Understanding vessel evasion tactics

**Advantages:**
- Detects behavior changes not visible in AIS alone
- Incorporates ecosystem health (shark presence/behavior)
- Temporal correlation between events
- Identifies specific areas and times of concern

---

## 📚 Development & Reference Notebooks

### `Creating_Models.ipynb`

**Purpose:** Development notebook - earlier version of Demo_Model with experimental variations

**What it does:**
1. Similar pipeline to Demo_Model.ipynb
2. Includes multiple model architecture attempts
3. Different preprocessing strategies
4. Experimental feature engineering
5. Comparison of different hyperparameters

**Key Differences from Demo_Model:**
- More exploratory code
- Multiple model attempts (LSTM, CNN variants)
- Verbose output and debugging cells
- Intermediate data saving

**When to Use:**
- Understanding development process
- Testing new preprocessing approaches
- Baseline comparison for modifications
- Learning the complete data pipeline

**Status:** Reference/Development - Use Demo_Model.ipynb for production

---

### `fixed_gear.ipynb`

**Purpose:** Focused analysis on one vessel type - fixed gear fishing vessels

**What it does:**
1. Loads `data/ais_vessel_data/fixed_gear.csv`
2. Applies same preprocessing as main model
3. Analyzes fixed gear vessel-specific patterns
4. Identifies loitering and anomalous behavior
5. Visualizes spatial distribution and temporal patterns

**Key Insights:**
- Fixed gear vessels have distinct behavior signatures
- Different loitering patterns than trawlers
- Specific geographic hotspots of activity
- Temporal patterns by season/month

**Use Cases:**
- Understanding vessel-type-specific behavior
- Developing specialized detection rules for fixed gear
- Comparative analysis across vessel types
- Domain expertise validation

**Vessel Type:** Fixed Gear
- Includes: Pots and traps, set longlines, set gillnets
- Characteristics: Stationary fishing, repeated locations
- Detection: Position clusters and long dwell times

**Data Used:**
- `data/ais_vessel_data/fixed_gear.csv`
- `data/vessel_metadata/fishing-vessels-v2.csv`

---

### `new5.ipynb`

**Purpose:** Advanced telemetry features and experimental analysis methods

**What it does:**
1. Extends new2.ipynb with additional feature engineering
2. Tests alternative anomaly detection methods
3. Explores environmental factor integration
4. Advanced temporal analysis
5. Experimental visualization techniques

**Experimental Features:**
- Velocity-based anomalies
- Acceleration detection
- Environmental correlation analysis
- Multi-scale temporal patterns
- Network analysis of vessel interactions

**Status:** Experimental - Results may not be production-ready

**When to Use:**
- Research on new detection methods
- Improving model accuracy beyond 98.38%
- Exploring environmental integration
- Academic publications

---

## 🔄 Workflow & Execution Order

### Recommended Order for New Users:
1. **Start:** `Demo_Model.ipynb` - Understand main AIS detection approach
2. **Learn:** `Creating_Models.ipynb` - See development process
3. **Explore:** `fixed_gear.ipynb` - Analyze specific vessel types
4. **Advanced:** `new2.ipynb` - Understand telemetry integration
5. **Experimental:** `new5.ipynb` - See cutting-edge approaches

### For Predictions:
```
Raw AIS Data → Demo_Model.ipynb → Fishing Predictions
Telemetry Data → new2.ipynb → Risk Scores
```

### For Research:
```
Exploratory Analysis → Creating_Models.ipynb
Vessel-Specific Study → fixed_gear.ipynb
Ecosystem Integration → new2.ipynb + new5.ipynb
```

---

## 📊 Common Usage Patterns

### Loading Data
```python
# AIS vessel data
import pandas as pd
data = pd.read_csv('data/ais_vessel_data/trawlers.csv')

# Metadata
vessels = pd.read_csv('data/vessel_metadata/fishing-vessels-v2.csv')

# Telemetry
sharks = pd.read_csv('data/telemetry_data/labelled_shark_data.csv')
```

### Key Preprocessing Steps
```python
# Sort by vessel and time
data = data.sort_values(by=['mmsi', 'timestamp'])

# Calculate distances (Haversine)
from geopy.distance import geodesic
distances = [geodesic((lat[i], lon[i]), (lat[i+1], lon[i+1])).meters 
             for i in range(len(lat)-1)]

# Fill missing values (AutoRegressive)
from statsmodels.tsa.ar_model import AutoReg
model = AutoReg(data['speed'].dropna(), lags=1)
model_fit = model.fit()

# Normalize features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)
```

### Model Training
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense

model = Sequential([
    Conv1D(64, 3, activation='relu', input_shape=(n_features, 1)),
    MaxPooling1D(2),
    Flatten(),
    Dense(2, activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy', 
              optimizer='adam', 
              metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)
```

### Anomaly Detection
```python
from sklearn.ensemble import IsolationForest

iso_forest = IsolationForest(contamination=0.05)
iso_forest.fit(data[['lon', 'lat', 'speed']])
anomalies = iso_forest.predict(data[['lon', 'lat', 'speed']])
```

---

## 🛠️ Troubleshooting

| Issue | Solution |
|-------|----------|
| Out of memory on large CSVs | Use `pd.read_csv()` with `chunksize` parameter |
| NaN values in calculations | Apply AR model imputation, then dropna() |
| Slow Haversine distance calc | Vectorize with numpy or use geopy's vectorized version |
| Model not converging | Reduce learning rate, increase epochs, check data normalization |
| Imbalanced classes | Use `class_weight` parameter in model.fit() |
| Poor predictions on new data | Check data preprocessing matches training pipeline |

---

## 📈 Performance Tuning

### For Faster Execution
- Use subset of data: `df.sample(frac=0.1)`
- Reduce epochs for prototyping
- Use smaller batch sizes for feedback faster

### For Better Accuracy
- Increase epochs (diminishing returns after 20)
- Experiment with learning rates (0.001 - 0.01)
- Add more Conv1D layers
- Use batch normalization
- Increase regularization (L1/L2)

### For Different Vessel Types
- Load different CSV from `data/ais_vessel_data/`
- Retrain model with vessel-specific data
- Consider separate models per vessel class
- Adjust hyperparameters per class

---

## 📝 Citation & References

If using these notebooks in research, cite:

```bibtex
@misc{fyp2024,
  title={Potential Detection of Illegal Fishing: Integrating Passive Acoustic Telemetry, Historical Data, AIS Tracking and Environmental Factors},
  author={Srikanthi, Meghana and Gattani, Khyati and Sivamurugan, V},
  year={2024},
  institution={Sri Sivasubramaniya Nadar College of Engineering}
}
```

For data sources, see main [README.md](../README.md) Acknowledgments section.

---

**Last Updated:** 2026-04-28  
**Maintenance Status:** Active Research
