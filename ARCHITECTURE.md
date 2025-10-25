# microvision architecture

## project overview

microvision is a comprehensive deep learning research lab designed for computer vision tasks. built with pytorch, numpy, and managed with uv, it provides a clean, modular framework for computer vision experiments with state-of-the-art model architectures and advanced analysis capabilities.

## system architecture

### High-Level Architecture
```mermaid
graph TB
    A[User/Researcher] --> B[Jupyter Notebooks]
    B --> C[Core Modules]
    C --> D[Training System]
    D --> E[Model Checkpoints]
    D --> F[Experiment Logs]
    
    C --> G[Config Management]
    C --> H[Data Loading]
    C --> I[Model Definitions]
    C --> J[Utilities]
    
    H --> K[CIFAR-10 Dataset]
    I --> L[SimpleCNN]
    I --> M[ImprovedCNN]
    I --> N[ResNet18]
    I --> O[EfficientNet-B0]
    J --> U[Visualization]
    J --> V[Metrics]
    J --> W[Grad-CAM]
    J --> X[t-SNE Analysis]
```

### Module Dependencies
```mermaid
graph TD
    A[config.py] --> B[datasets.py]
    A --> C[models.py]
    A --> D[trainer.py]
    A --> E[utils.py]
    
    B --> D
    C --> D
    E --> D
    
    F[__init__.py] --> A
    F --> B
    F --> C
    F --> D
    F --> E
```

## core components

### 1. Configuration System (`config.py`)
```mermaid
classDiagram
    class Config {
        +int batch_size
        +int num_workers
        +str data_root
        +int num_classes
        +int input_size
        +int epochs
        +float lr
        +float weight_decay
        +float momentum
        +int seed
        +str device
        +str log_dir
        +str ckpt_dir
        +bool use_augmentation
        +__post_init__()
    }
```

### 2. Data Pipeline (`datasets.py`)
```mermaid
flowchart TD
    A[CIFAR-10 Dataset] --> B[Data Transforms]
    B --> C[Train Transforms]
    B --> D[Val/Test Transforms]
    
    C --> E[RandomCrop]
    C --> F[RandomHorizontalFlip]
    C --> G[ToTensor]
    C --> H[Normalize]
    
    D --> I[ToTensor]
    D --> J[Normalize]
    
    E --> K[Train DataLoader]
    F --> K
    G --> K
    H --> K
    I --> L[Val DataLoader]
    J --> L
    I --> M[Test DataLoader]
    J --> M
    
    K --> N[Training Pipeline]
    L --> O[Validation Pipeline]
    M --> P[Testing Pipeline]
```

### 3. Model Architecture (`models.py`)
```mermaid
graph TB
    A[Input: 3x32x32] --> B[Model Selection]
    B --> C[SimpleCNN]
    B --> D[ImprovedCNN]
    B --> E[ResNet18]
    B --> F[EfficientNet-B0]
    
    C --> G[Custom 3-Layer CNN]
    D --> H[CNN with Residual Connections]
    E --> I[ResNet with Pretrained Weights]
    F --> J[EfficientNet with Pretrained Weights]
    
    G --> K[Output: 10 classes]
    H --> K
    I --> K
    J --> K
```

### 4. Training System (`trainer.py`)
```mermaid
stateDiagram-v2
    [*] --> Initialize
    Initialize --> Training
    Training --> Validation
    Validation --> Checkpoint
    Checkpoint --> Training
    Checkpoint --> [*]
    
    state Training {
        [*] --> ForwardPass
        ForwardPass --> LossComputation
        LossComputation --> BackwardPass
        BackwardPass --> OptimizerStep
        OptimizerStep --> [*]
    }
    
    state Validation {
        [*] --> ModelEval
        ModelEval --> ForwardPass
        ForwardPass --> MetricsComputation
        MetricsComputation --> [*]
    }
    
    state Checkpoint {
        [*] --> SaveModel
        SaveModel --> LogMetrics
        LogMetrics --> [*]
    }
```

## training workflow

### Complete Training Pipeline
```mermaid
sequenceDiagram
    participant U as User
    participant C as Config
    participant D as DataLoader
    participant M as Model
    participant T as Trainer
    participant L as Logger
    
    U->>C: Initialize Config
    U->>D: Create DataLoaders
    U->>M: Create Model
    U->>T: Initialize Trainer
    
    loop For each epoch
        T->>D: Get batch
        T->>M: Forward pass
        T->>M: Compute loss
        T->>M: Backward pass
        T->>M: Update weights
        T->>L: Log training metrics
    end
    
    T->>D: Validate model
    T->>L: Log validation metrics
    T->>T: Save checkpoint
    T->>U: Return results
```

### Data Flow
```mermaid
flowchart LR
    A[Raw Images] --> B[Data Augmentation]
    B --> C[Normalization]
    C --> D[Batch Creation]
    D --> E[Model Input]
    E --> F[Forward Pass]
    F --> G[Loss Computation]
    G --> H[Backward Pass]
    H --> I[Weight Update]
    I --> J[Next Batch]
    J --> D
```

## experiment tracking

### Metrics and Logging
```mermaid
graph TB
    A[Training Metrics] --> B[TensorBoard]
    C[Model Checkpoints] --> D[File System]
    E[Visualization] --> F[Matplotlib Plots]
    
    A --> G[Loss Curves]
    A --> H[Accuracy Curves]
    A --> I[Confusion Matrix]
    A --> J[Sample Predictions]
    
    B --> K[Real-time Monitoring]
    D --> L[Model Persistence]
    F --> M[Result Analysis]
```

## model performance analysis

### Training Curves
```mermaid
xychart-beta
    title "Training Progress"
    x-axis [1, 5, 10, 15, 20, 25, 30]
    y-axis "Accuracy %" 0 --> 100
    line "Training Accuracy" [45, 65, 75, 82, 87, 91, 94]
    line "Validation Accuracy" [42, 62, 72, 78, 83, 86, 88]
```

### Loss Curves
```mermaid
xychart-beta
    title "Loss Evolution"
    x-axis [1, 5, 10, 15, 20, 25, 30]
    y-axis "Loss" 0 --> 3
    line "Training Loss" [2.5, 1.8, 1.2, 0.8, 0.5, 0.3, 0.2]
    line "Validation Loss" [2.6, 1.9, 1.4, 1.0, 0.7, 0.5, 0.4]
```

## development workflow

### Project Structure
```mermaid
graph TD
    A[microvision/] --> B[src/]
    A --> C[notebooks/]
    A --> D[experiments/]
    A --> E[data/]
    
    B --> F[config.py]
    B --> G[datasets.py]
    B --> H[models.py]
    B --> I[trainer.py]
    B --> J[utils.py]
    B --> K[__init__.py]
    
    C --> L[01_data_exploration.ipynb]
    C --> M[02_model_training.ipynb]
    C --> N[03_model_analysis.ipynb]
    C --> O[04_experiments.ipynb]
    
    D --> P[logs/]
    D --> Q[checkpoints/]
    D --> R[models/]
```

### Development Phases
```mermaid
gantt
    title microvision Development Timeline
    dateFormat  YYYY-MM-DD
    section Phase 1: Foundation
    Project Structure    :done, p1, 2024-01-01, 1d
    Configuration       :done, p1b, after p1, 1d
    
    section Phase 2: Core Modules
    Data Loading        :done, p2a, after p1b, 1d
    Model Architecture  :done, p2b, after p2a, 1d
    
    section Phase 3: Training
    Training System     :done, p3, after p2b, 2d
    Evaluation          :done, p3b, after p3, 1d
    
    section Phase 4: Utilities
    Visualization       :done, p4, after p3b, 1d
    Model Registry      :done, p4b, after p4, 1d
    
    section Phase 5: Notebooks
    Data Exploration   :done, p5a, after p4b, 1d
    Training Notebook  :done, p5b, after p5a, 1d
    Analysis Notebook  :done, p5c, after p5b, 1d
    Experiments       :done, p5d, after p5c, 1d
```

## usage patterns

### Basic Usage Flow
```mermaid
flowchart TD
    A[Import microvision] --> B[Create Config]
    B --> C[Load Data]
    C --> D[Create Model]
    D --> E[Initialize Trainer]
    E --> F[Train Model]
    F --> G[Evaluate Results]
    G --> H[Visualize Results]
```

### Advanced Usage Flow
```mermaid
flowchart TD
    A[Experiment Design] --> B[Hyperparameter Tuning]
    B --> C[Model Comparison]
    C --> D[Architecture Search]
    D --> E[Performance Analysis]
    E --> F[Result Documentation]
```

## key features

### Modularity
- **Config-driven**: All parameters centralized in `Config` class
- **Pluggable models**: Easy model swapping via factory pattern
- **Flexible data**: Support for different datasets and transforms
- **Extensible training**: Custom training loops and metrics

### Reproducibility
- **Fixed seeds**: Deterministic training across runs
- **Version control**: All dependencies locked with uv
- **Checkpointing**: Model state preservation
- **Logging**: Complete experiment tracking

### Performance
- **GPU acceleration**: Automatic device detection
- **Efficient data loading**: Multi-threaded data pipeline
- **Memory optimization**: Gradient checkpointing support
- **Batch processing**: Optimized for small-scale datasets

## metrics and evaluation

### Training Metrics
- **Loss**: Cross-entropy loss with regularization
- **Accuracy**: Top-1 classification accuracy
- **Learning curves**: Training and validation progress
- **Model size**: Parameter count and memory usage

### Visualization Tools
- **Training curves**: Loss and accuracy over time
- **Confusion matrix**: Class-wise performance
- **Sample predictions**: Visual inspection of results
- **Model architecture**: Network structure visualization

## advanced features

### Model Analysis and Interpretability
```mermaid
graph TB
    A[Trained Models] --> B[Performance Analysis]
    A --> C[Model Comparison]
    A --> D[Interpretability]
    
    B --> E[Accuracy Metrics]
    B --> F[Parameter Count]
    B --> G[Training Time]
    B --> H[Efficiency Analysis]
    
    C --> I[Architecture Comparison]
    C --> J[Performance Ranking]
    C --> K[Feature Visualization]
    
    D --> L[Grad-CAM]
    D --> M[t-SNE Analysis]
    D --> N[Confusion Matrices]
    D --> O[Sample Predictions]
```

### Experiment Framework
```mermaid
graph TB
    A[Experiment Design] --> B[Hyperparameter Optimization]
    A --> C[Model Ensemble]
    A --> D[Transfer Learning]
    A --> E[Ablation Studies]
    
    B --> F[Grid Search]
    B --> G[Random Search]
    B --> H[Bayesian Optimization]
    
    C --> I[Majority Voting]
    C --> J[Weighted Ensemble]
    C --> K[Stacking]
    
    D --> L[Pretrained vs From-Scratch]
    D --> M[Fine-tuning]
    D --> N[Feature Extraction]
    
    E --> O[Component Analysis]
    E --> P[Architecture Ablation]
    E --> Q[Training Ablation]
```

## implemented features

### Current Capabilities
- **4 Model Architectures**: SimpleCNN, ImprovedCNN, ResNet18, EfficientNet-B0
- **Comprehensive Analysis**: Model comparison, performance benchmarking, interpretability tools
- **Advanced Experiments**: Hyperparameter optimization, ensemble methods, transfer learning, ablation studies
- **Visualization Tools**: Training curves, confusion matrices, Grad-CAM, t-SNE analysis
- **Model Registry**: Automatic model saving, loading, and metadata management
- **Experiment Tracking**: TensorBoard integration, checkpointing, performance logging

This architecture provides a comprehensive foundation for computer vision research with state-of-the-art models and advanced analysis capabilities. The modular design allows for easy experimentation and rapid prototyping of new ideas.
