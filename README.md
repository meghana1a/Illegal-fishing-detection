# Illegal Fishing Detection System

## Project Overview

This project implements a machine learning system to detect and classify illegal, unreported, and unregulated (IUU) fishing activities using two complementary approaches:

1. **Unit 1: AIS-Based Vessel Behavior Analysis** - Detects suspicious fishing patterns from Automatic Identification System (AIS) vessel tracking data using 1D Convolutional Neural Networks
2. **Unit 2: Acoustic Telemetry Correlation** - Analyzes shark acoustic telemetry data to identify potential illegal fishing activities near protected marine areas using Isolation Forest anomaly detection and Neural Networks

## Key Achievements

- **98.38% accuracy** on AIS-based fishing detection model
- **Multi-vessel type analysis** - Processes 6 different fishing vessel classifications (trawlers, purse seines, drifting longlines, etc.)
- **Cross-dataset integration** - Correlates vessel behavior, shark telemetry, and environmental factors
- **Scalable architecture** - Processes millions of AIS data points efficiently

## Project Structure

```
.
├── docs/                          # Documentation
│   ├── FYP_Panel4_Team13.doc     # Original research paper
│   ├── Illegal_Fishing_Detection_using_Neural_Network.pdf
│   └── README-*.txt               # Data source documentation
│
├── notebooks/                     # Jupyter notebooks
│   ├── Demo_Model.ipynb          # ⭐ Main production model (Unit 1)
│   ├── Creating_Models.ipynb     # Development/training notebook
│   ├── fixed_gear.ipynb          # Focused analysis on fixed gear vessels
│   ├── new2.ipynb                # ⭐ Shark telemetry analysis (Unit 2)
│   └── new5.ipynb                # Advanced telemetry features
│
└── data/                          # Datasets (5.2 GB) - NOT INCLUDED IN REPO
    ├── ais_vessel_data/          # AIS tracking by vessel type
    ├── telemetry_data/           # Shark acoustic tracking
    ├── vessel_metadata/          # Vessel classifications
    ├── fishing_effort/           # Historical patterns
    ├── training_data/            # ML training datasets
    └── reference/                # Search results & metadata
```

## Quick Start

### Requirements
- Python 3.8+
- pandas, numpy, scikit-learn, tensorflow/keras
- geopy, matplotlib, geopandas (for visualization)

### Running the Main Model
```bash
# Unit 1: AIS Detection
jupyter notebook notebooks/Demo_Model.ipynb

# Unit 2: Telemetry Analysis
jupyter notebook notebooks/new2.ipynb
```

## Technical Approach

### Unit 1: 1D CNN for AIS Vessel Analysis
**Input:** Vessel movement time series (MMSI, speed, course, heading, distance from shore/port)
**Model:** 1D Convolutional Neural Network with MaxPooling
**Output:** Binary fishing/non-fishing classification

**Key Features:**
- AutoRegressive imputation for missing values
- Haversine distance calculations for vessel movement
- Per-vessel normalization and sequence padding
- 10 epochs, batch size 32, learning rate 0.005

### Unit 2: Isolation Forest + Neural Network for Telemetry
**Input:** Shark acoustic telemetry detections + vessel positions
**Anomaly Detection:** Isolation Forest on movement patterns (4.99% contamination)
**Classification:** Neural Network with LeakyReLU + L1/L2 regularization
**Output:** Probability of illegal fishing activity near protected areas

**Key Features:**
- Real-time anomaly detection in telemetry arrays
- Loitering event identification (speed < 2 knots)
- Temporal correlation with fish detections
- Environmental factor integration

## Data Overview

| Category | Size | Purpose |
|----------|------|---------|
| AIS Vessel Data | 4.4 GB | Train 1D CNN for behavior analysis |
| Shark Telemetry | 27 MB | Detect anomalies near protected areas |
| Training Data | 139 MB | ML model training and validation |
| Fishing Effort | 610 MB | Historical patterns and baselines |
| Vessel Metadata | 17 MB | Classify and characterize vessels |
| Reference | 20 KB | Search results and citations |

## Getting the Data

**Note:** Large data files (>100MB) are not included in this repository due to GitHub's file size limitations. To run the notebooks, you'll need to obtain the datasets from their original sources:

- **AIS Vessel Data:** Global Fishing Watch - Contact GFW or visit https://globalfishingwatch.org/
- **Shark Telemetry Data:** Dryad Digital Repository - Search for "Jacoby shark telemetry" or contact BIOT MPA
- **Training Data:** Contact the research team or download from the sources listed in the References section

Once downloaded, place the data files in the `data/` directory following the structure shown above. Refer to the original data source documentation for schemas and processing instructions.

## Notebooks Guide

| Notebook | Purpose | Status |
|----------|---------|--------|
| **Demo_Model.ipynb** | Main production AIS detection model | ✅ Production |
| **Creating_Models.ipynb** | Earlier development version | 📚 Reference |
| **fixed_gear.ipynb** | Specialized analysis for fixed gear vessels | 🔍 Analysis |
| **new2.ipynb** | Shark telemetry with Isolation Forest | ✅ Production |
| **new5.ipynb** | Advanced telemetry feature engineering | 📊 Experimental |

See [docs/NOTEBOOKS.md](docs/NOTEBOOKS.md) for detailed notebook documentation.

## Model Performance

### Unit 1 - AIS Detection
- **Accuracy:** 98.38%
- **Loss:** 18.44%
- **Precision:** High confidence in fishing detection
- **Anomalies Detected:** 17,819 (4.99% of dataset)
- **Average Anomaly Score:** -0.0639

### Unit 2 - Telemetry Analysis
- **Anomaly Detection Rate:** ~5% of telemetry events flagged
- **Shark Detection Correlation:** Identifies vessel-animal interactions
- **Temporal Precision:** 1-hour buffer for timestamp matching
- **Environmental Integration:** Sea surface temperature, currents

## Key Algorithms

### AIS Analysis
- **Distance Calculation:** Haversine formula for geodesic distances
- **Missing Value Imputation:** AutoRegressive (AR) models with lag=1
- **Feature Extraction:** Speed, heading, distance metrics per vessel
- **Classification:** 1D CNN (Conv1D → MaxPooling → Flatten → Dense)

### Telemetry Analysis
- **Anomaly Detection:** Isolation Forest (contamination=0.05)
- **Loitering Detection:** Speed-based thresholding (< 2 knots)
- **Correlation:** Temporal and spatial matching between datasets
- **Classification:** Neural Network (ReLU → LeakyReLU → ELU)

## Data Sources

### AIS Vessel Data
- **Provider:** Global Fishing Watch (GFW)
- **Coverage:** 2012-2020 global fishing fleet
- **Resolution:** Variable temporal (minutes to hours)
- **Vessels:** ~70,000-114,000 active per year
- **License:** CC BY-SA 4.0

### Shark Telemetry
- **Provider:** BIOT MPA Research / Dryad Digital Repository
- **Species:** Grey reef sharks, silvertip sharks
- **Coverage:** 2013-2018, British Indian Ocean Territory
- **Arrays:** 5 isolated atoll systems
- **Tags:** 101 individuals tracked continuously

### Ocean Environment Data
- **Provider:** Data.World, NOAA
- **Variables:** Sea surface temperature, currents, wind
- **Resolution:** Daily at 0.1° grid

## Research Context

This project addresses a critical conservation challenge: **detecting and preventing illegal fishing in marine protected areas and exclusive economic zones**. Traditional enforcement relies on vessel surveillance and inspection, which is costly and covers only a fraction of global fishing activity.

By integrating:
- Real-time vessel positioning (AIS)
- Marine animal tracking (acoustic telemetry)
- Historical fishing patterns (effort data)
- Environmental factors (weather, currents)

...the system enables **risk-based enforcement** that prioritizes limited enforcement resources on areas and vessels with highest illegal fishing probability.

## Files & Documentation

- **[METHODOLOGY.md](docs/METHODOLOGY.md)** - Detailed technical methodology
- **[NOTEBOOKS.md](docs/NOTEBOOKS.md)** - Per-notebook documentation and usage
- **[data/README.md](data/README.md)** - Data structure and schemas
- **[RESULTS.md](docs/RESULTS.md)** - Model results and performance analysis

## Contributors

**Research Team:**
- Meghana Srikanth
- Khyati Gattani

**Advisors:** Department of Information Technology, Sri Sivasubramaniya Nadar College of Engineering

## Acknowledgments

- Global Fishing Watch for AIS data and open research framework
- BIOT MPA for shark telemetry dataset (Jacoby et al., 2020)
- Indian Ocean Tuna Commission (IOTC) for IUU fishing incident reports
- IMOS (Integrated Marine Observing System) for acoustic infrastructure data

## References

See [docs/FYP_Panel4_Team13.doc](docs/FYP_Panel4_Team13.doc) for complete research paper with 24+ citations including:
- Kroodsma et al. (2018) - Global Fishing Watch methodology
- Jacoby et al. (2020) - Shark movement strategies in MPAs
- Agnew et al. (2009) - Estimating extent of illegal fishing
- Multiple journal articles on marine protected area enforcement

## License

This project is for educational and research purposes. Data usage follows original source licenses (CC BY-SA 4.0 for GFW data, Dryad repository terms for telemetry).
