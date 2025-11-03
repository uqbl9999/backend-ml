# Project Structure Documentation

## 📂 Directory Organization

```
backend-ml/
│
├── 📁 data/                          # Data storage (gitignored)
│   ├── dataset_limpio.csv            # Cleaned data
│   ├── df_clean_to_model.csv         # Encoded features
│   └── dataset_balanceado.csv        # Balanced dataset
│
├── 📁 src/                           # Source code
│   ├── data_preparation.py           # Data processing pipeline
│   ├── train_model.py               # Training script
│   ├── example_prediction.py        # Example usage
│   └── 📁 models/                    # Model modules
│       ├── __init__.py
│       ├── training.py               # Model training logic
│       └── prediction.py             # Prediction logic
│
├── 📁 api/                           # REST API
│   └── main.py                       # FastAPI application
│
├── 📁 models/                        # Trained models
│   └── trained_model.pkl             # Serialized model
│
├── 📁 docs/                          # Documentation & plots
│   ├── evaluation_*.png              # Evaluation plots
│   └── PROJECT_STRUCTURE.md          # This file
│
├── 📁 tests/                         # Unit tests
│   └── test_prediction.py            # Prediction tests
│
├── 📁 notebooks/                     # Jupyter notebooks
│   └── parcialfinal.ipynb           # Original exploration
│
├── 📄 requirements.txt               # Python dependencies
├── 📄 README.md                      # Main documentation
├── 📄 QUICKSTART.md                  # Quick start guide
└── 📄 .gitignore                     # Git ignore rules
```

## 🔄 Data Flow

```
┌─────────────────┐
│  tamizajes.csv  │  Raw Data
└────────┬────────┘
         │
         ▼
┌─────────────────────────────┐
│  data_preparation.py        │
│  ┌─────────────────────┐   │
│  │ 1. Load Data        │   │
│  │ 2. Calculate Rate   │   │
│  │ 3. Clean Data       │   │
│  │ 4. Feature Eng.     │   │
│  │ 5. Balance Data     │   │
│  └─────────────────────┘   │
└──────────┬──────────────────┘
           │
           ▼
┌─────────────────────────────┐
│  training.py                │
│  ┌─────────────────────┐   │
│  │ 1. Split Data       │   │
│  │ 2. Train Base Model │   │
│  │ 3. Optimize Params  │   │
│  │ 4. Evaluate         │   │
│  │ 5. Save Model       │   │
│  └─────────────────────┘   │
└──────────┬──────────────────┘
           │
           ▼
┌─────────────────────────────┐
│  trained_model.pkl          │  Saved Model
└──────────┬──────────────────┘
           │
           ▼
┌─────────────────────────────┐
│  prediction.py              │
│  ┌─────────────────────┐   │
│  │ 1. Load Model       │   │
│  │ 2. Prepare Features │   │
│  │ 3. Make Prediction  │   │
│  │ 4. Interpret Result │   │
│  └─────────────────────┘   │
└──────────┬──────────────────┘
           │
           ▼
┌─────────────────────────────┐
│  FastAPI (main.py)          │
│  ┌─────────────────────┐   │
│  │ REST Endpoints      │   │
│  │ - /predict          │   │
│  │ - /predict/batch    │   │
│  │ - /model/info       │   │
│  └─────────────────────┘   │
└──────────┬──────────────────┘
           │
           ▼
┌─────────────────────────────┐
│  Client (Web/Mobile/API)    │
└─────────────────────────────┘
```

## 🏗️ Architecture Pattern

This project follows a **simplified layered architecture**:

### 1. Data Layer (`src/data_preparation.py`)
- **Responsibility**: Data loading, cleaning, transformation
- **Input**: Raw CSV files
- **Output**: Processed, balanced datasets ready for ML

### 2. Model Layer (`src/models/`)
- **training.py**
  - **Responsibility**: Model training, optimization, evaluation
  - **Input**: Prepared datasets
  - **Output**: Trained model (.pkl file)

- **prediction.py**
  - **Responsibility**: Load model, make predictions
  - **Input**: Feature dictionary
  - **Output**: Prediction + interpretation

### 3. API Layer (`api/main.py`)
- **Responsibility**: REST endpoints, validation, error handling
- **Input**: HTTP requests (JSON)
- **Output**: HTTP responses (JSON)
- **Framework**: FastAPI

### 4. Interface Layer (External)
- **Responsibility**: User interaction
- **Tools**: Swagger UI, curl, client applications

## 🔌 API Endpoints Architecture

```
FastAPI Application (main.py)
│
├── Middleware
│   └── CORS
│
├── Startup Events
│   └── Load Model
│
├── Health & Info Endpoints
│   ├── GET /
│   ├── GET /health
│   └── GET /model/info
│
├── Prediction Endpoints
│   ├── POST /predict          → predict_single()
│   └── POST /predict/batch    → predict_batch()
│
├── Model Info Endpoints
│   └── GET /model/features    → get_feature_importance()
│
└── Metadata Endpoints
    ├── GET /metadata/departamentos
    ├── GET /metadata/tamizajes
    └── GET /metadata/etapas
```

## 🧩 Class Diagram

```
┌──────────────────────────┐
│   DataPreparation        │
├──────────────────────────┤
│ - data_path              │
│ - df                     │
│ - df_pivot               │
│ - df_clean               │
│ - df_encoded             │
├──────────────────────────┤
│ + load_data()            │
│ + calculate_positivity() │
│ + clean_data()           │
│ + feature_engineering()  │
│ + balance_data()         │
│ + prepare_full_pipeline()│
└──────────────────────────┘

┌──────────────────────────┐
│   ModelTrainer           │
├──────────────────────────┤
│ - model_type             │
│ - model                  │
│ - X_train, X_test        │
│ - y_train, y_test        │
│ - feature_names          │
│ - metrics                │
├──────────────────────────┤
│ + split_data()           │
│ + train_base_model()     │
│ + optimize_hyperparams() │
│ + save_model()           │
│ + load_model()           │
│ + predict()              │
│ + plot_results()         │
└──────────────────────────┘

┌──────────────────────────┐
│   Predictor              │
├──────────────────────────┤
│ - model_path             │
│ - model                  │
│ - feature_names          │
│ - model_type             │
│ - metrics                │
├──────────────────────────┤
│ + load_model()           │
│ + predict_single()       │
│ + predict_batch()        │
│ + validate_input()       │
│ + get_feature_import()   │
│ + get_model_info()       │
└──────────────────────────┘
```

## 🔐 Security Considerations

Current implementation (Development):
- ✅ Input validation (Pydantic models)
- ✅ CORS enabled for all origins
- ❌ No authentication
- ❌ No rate limiting
- ❌ No logging

Recommended for Production:
- 🔒 Add JWT authentication
- 🔒 Implement API key system
- 🔒 Add rate limiting
- 🔒 Restrict CORS origins
- 🔒 Add comprehensive logging
- 🔒 Use HTTPS
- 🔒 Add input sanitization
- 🔒 Implement monitoring

## 📊 Model Pipeline

```
Training Phase:
┌────────────┐    ┌───────────┐    ┌──────────┐    ┌────────────┐
│  Raw Data  │ -> │  Clean &  │ -> │ Balance  │ -> │   Train    │
│            │    │  Encode   │    │          │    │            │
└────────────┘    └───────────┘    └──────────┘    └────────────┘
                                                           │
                                                           ▼
                                                    ┌────────────┐
                                                    │   Save     │
                                                    │  Model.pkl │
                                                    └────────────┘

Prediction Phase:
┌────────────┐    ┌───────────┐    ┌──────────┐    ┌────────────┐
│   Input    │ -> │  Prepare  │ -> │  Predict │ -> │  Interpret │
│   JSON     │    │  Features │    │          │    │            │
└────────────┘    └───────────┘    └──────────┘    └────────────┘
```

## 🎯 Design Decisions

### Why This Structure?

1. **Separation of Concerns**
   - Data prep is independent of model training
   - Prediction logic is separate from API logic
   - Easy to modify one component without affecting others

2. **Simplicity First**
   - Chose simple structure over complex DDD
   - Appropriate for academic/small-scale project
   - Easy to understand and maintain

3. **Scalability Path**
   - Clear structure allows easy migration to DDD if needed
   - Can add layers (caching, queuing) without major refactor
   - API design supports multiple clients

4. **Testability**
   - Each module can be tested independently
   - Mock data/models easily
   - Unit tests for critical functions

### Why FastAPI?

- ✅ Automatic API documentation (Swagger/ReDoc)
- ✅ Type validation with Pydantic
- ✅ Async support (future scalability)
- ✅ Modern Python (3.8+)
- ✅ Fast performance
- ✅ Easy to learn

### Why Pickle for Model?

- ✅ Standard scikit-learn serialization
- ✅ Preserves entire model state
- ✅ Easy to load and use
- ⚠️  Not secure for untrusted sources
- ⚠️  Python version dependent

Alternative: ONNX (for production/cross-platform)

## 📈 Future Enhancements

Potential improvements:

1. **Add Caching Layer** (Redis)
   - Cache frequent predictions
   - Store model in memory

2. **Add Database** (PostgreSQL)
   - Store prediction history
   - User management
   - Analytics

3. **Add Message Queue** (RabbitMQ/Celery)
   - Async batch predictions
   - Model retraining jobs

4. **Add Monitoring** (Prometheus/Grafana)
   - API metrics
   - Model performance drift
   - Error tracking

5. **Containerization** (Docker)
   - Easy deployment
   - Environment consistency

6. **CI/CD Pipeline** (GitHub Actions)
   - Automated testing
   - Automated deployment

## 📚 Related Documentation

- [README.md](../README.md) - Main documentation
- [QUICKSTART.md](../QUICKSTART.md) - Quick start guide
- [API Documentation](http://localhost:8000/docs) - Interactive API docs (when running)
