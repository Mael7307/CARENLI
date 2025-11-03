# CARENLI

This repository contains the core for "CARENLI: Compartmentalised Agentic Reasoning for Clinical NLI".

## Repository Structure

- **./agents/**: Core agent implementations including:
  - `critiques/`: Formal verification agents (Isabelle, Prover9)
  - `generation/`: Text generation models (API and local models)
  - `meta_agents/`: Higher-level planning and coordination agents (initializer/portfolio, planner)
  - `solvers/`: Formal solver implementations (LP, FOL, SMT, CSP) for various reasoning tasks. **Our experiments are conducted using these solvers.**
  - `reasoners/`: Other extensible agent modules for information post-processing, analysis, and reasoning.
- **./base/**: Shared utilities and logging infrastructure
- **./comparing_methods/**: Baseline comparison methods including LLM-based approaches
- **./data/**: Training and evaluation datasets.
- **./libs/**: External libraries (Prover9, pyke knowledge engine)
- **./prompt/**: Prompt templates for different reasoning tasks
- **config.yaml**: Main configuration file

## System Requirements

**Python Version**: 3.11.13

**Some solvers use external solvers that need a separate installation from their official website.**:
- **MiniZinc**: Constraint Programming toolkit
- **Isabelle**: Optional - Not required for experimental results reproduction.

**Key Dependencies** (see `requirements.txt` for complete list):
- `transformers==4.52.4` - For sft
- `llamafactory==0.9.4.dev0` - For sft
- `openai==1.91.0` - For LLM-based generation and reasoning
- `z3-solver==4.15.1.0` - For SMT constraint solving
- `pyke==1.1.1` - Knowledge engine for rule-based reasoning (included in libs/)
- `Prover9` - Custom installation in `libs/Prover9/` - First-order logic theorem prover
- `langchain==0.3.25` - For LLM orchestration

For complete dependency, see: `requirements.txt`.


## Special Dependencies Setup

### Prover9 Installation
The Prover9 theorem prover is included in `libs/Prover9/`. Compile it by:
```bash
cd libs/Prover9
make all
```

### PyKE Knowledge Engine
The PyKE rule-based reasoning engine is included in `libs/pyke-1.1.1/`. Install it by:
```bash
cd libs/pyke-1.1.1
python setup.py install
```

### Z3 Solver
Z3 is automatically installed via pip requirements. Ensure system compatibility for optimal performance.

### Isabelle/HOL
Requires separate Isabelle installation. Configure the Isabelle path in `config.yaml` under `agent_config_dsw.isabelle_path`.

### MiniZinc Setup
1. Download MiniZinc from the official website
2. Install following the platform-specific instructions
3. Configure the MiniZinc path in `config.yaml` under `agent_config_dsw.minizinc_path`


## Configuration Setup

Before running the system, you need to configure the environment paths and API keys in `config.yaml`:

### 1. Environment Paths Configuration
Update the `agent_config_dsw` section with your local paths:
```yaml
agent_config:
  minizinc_path: "/path/to/your/MiniZincIDE"
  isabelle_path: "/path/to/your/Isabelle2023/bin" # (Optional - Not required for experimental results reproduction.)
```

### 2. API Keys Configuration
Configure your API keys in the `api_config` section. The framework supports both OpenAI API and Azure OpenAI API. Hugging Face API keys are required for downloading and using models from the Hugging Face Hub.

**For OpenAI API (GPT models)**:
```yaml
api_config:
  gpt-4o-openai: 
    model_name: "gpt-4o"
    api_key: "YOUR_OPENAI_API_HERE"
  

  gpt-4o-azure:
    model_name: "YOUR_AZURE_CONFIG_HERE"
    azure_endpoint: "YOUR_AZURE_CONFIG_HERE"
    openai_api_version: "YOUR_AZURE_CONFIG_HERE"
    api_key: "YOUR_AZURE_CONFIG_HERE"
```

**For Hugging Face Models**:
```yaml
  your-model-name:
    model_name: "/path/to/your/huggingface/model"
    api_key: "YOUR_HUGGINGFACE_API_KEY_HERE"
    lora_path: "/path/to/lora/checkpoint"  # Optional for fine-tuned models
```


## Data Creation

We used the following code create the necessary datasets (mixed, milti-question prompt) for our experiments:

### 1. Multi-Problem Dataset Creation
```bash
python create_multi_problem_dataset.py
```
**Dataset Sources**: [PLACEHOLDER - Dataset sources will be added here]

### 2. Mixed Training Data Creation  
```bash
python create_mixing_data.py
```
**Dataset Sources**: Source data can be found in the following links:
- For folio, logical_deduction, prontoQA, and ProofWriter, we use the processed cata in the github repository of Logic-LM.
- For $TREC_{trial}$, we use the data in the 2021 Clinical Trials Track.

## Training

To train the models, run:
```bash
bash train_scripts.sh
```

**Note**: Before running, you need to configure the following paths in `train_scripts.sh`:
- Set your model path for Llama-3.1-8B (or other supported models)
- Set your LLaMA-Factory installation path
- Set your desired output directory for saved models

The script supports training with different learning rates and uses LoRA fine-tuning with DeepSpeed for efficient training.

## Testing

To run evaluations, execute:
```bash
bash test_scripts.sh
```

This will run tests on both local models and API-based models using the mixed dataset configuration.

