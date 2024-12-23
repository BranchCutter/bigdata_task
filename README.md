# Model Quantization Project

## Project Description
This project demonstrates the use of model quantization techniques, including K-means quantization and linear quantization, to reduce model size and improve inference speed. The script supports quantization-aware training (QAT) and evaluates the performance difference between original and quantized models. The implementation targets resource-constrained environments such as mobile and embedded systems.

## Project Structure
```
project/
 ├── Dockerfile                  # Docker container configuration
 ├── requirements.txt            # Python dependencies
 ├── quantization_utils.py       # Functions for model quantization
 ├── performance_evaluation.py   # Functions for model evaluation and comparison
 ├── main.py                     # Main script for training and evaluation
 └── README.md                   # Project documentation
```

## Setup Instructions

### 1. Using a Virtual Environment

#### Option 1: Using `venv`
1. Create a virtual environment:
   ```bash
   python -m venv venv
   ```
2. Activate the virtual environment:
   - On Linux/Mac:
     ```bash
     source venv/bin/activate
     ```
   - On Windows:
     ```bash
     venv\Scripts\activate
     ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the main script:
   ```bash
   python main.py
   ```

#### Option 2: Using `conda`
1. Create a new conda environment:
   ```bash
   conda create --name quantization_env python=3.9 -y
   ```
2. Activate the environment:
   ```bash
   conda activate quantization_env
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the main script:
   ```bash
   python main.py
   ```

### 2. Using Docker

#### Steps to Build and Run the Docker Container
1. Build the Docker image:
   ```bash
   docker build -t model-quantization .
   ```
2. Run the container:
   ```bash
   docker run --gpus all -it model-quantization
   ```

#### (Optional) Mount Local Data Directory
If you have a local dataset you want to use:
```bash
docker run --gpus all -it -v $(pwd)/data:/app/data model-quantization
```

## Files to Execute

1. **Main Script**:
   - `main.py` is the entry point. It performs the following:
     - Loads and preprocesses the CIFAR-10 dataset.
     - Trains a ResNet18 model with quantization-aware training (QAT).
     - Applies K-means quantization to the trained model.
     - Compares the performance of the original and quantized models.

   - To execute:
     ```bash
     python main.py
     ```

2. **Evaluation Script** (if isolated evaluation is needed):
   - Use `performance_evaluation.py` to evaluate specific models independently.

## Performance Comparison

### Metrics:
- **Loss**: Measures the error in predictions. Lower is better.
- **Accuracy**: Measures the proportion of correct predictions. Higher is better.

### Example Results:
| Metric      | Original Model | Quantized Model |
|-------------|----------------|-----------------|
| Loss        | 0.3            | 0.35            |
| Accuracy    | 85%            | 83%             |

### Visualization:
- Comparison graphs for loss and accuracy are generated to highlight the trade-offs between efficiency and performance.

## Practical Applications
- Deploying quantized models in resource-constrained environments (e.g., smartphones, embedded systems).
- Real-time applications where inference speed and energy efficiency are critical.

## Educational Value
This project provides hands-on experience with:
- Model quantization techniques.
- Quantization-aware training (QAT).
- Performance evaluation and trade-off analysis.

