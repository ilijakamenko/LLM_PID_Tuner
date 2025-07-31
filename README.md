# LLM-Based PID Controller Optimization

## Overview
This project presents a novel approach to PID controller tuning using Large Language Models (LLMs) like GPT-4o and DeepSeekV2. The goal is to dynamically adjust PID parameters (Kp, Ti, Td) based on system response metrics using LLM-guided decision-making. This allows for intelligent, automated tuning without manual heuristics.

## Features
- PID controller tuning without requiring the exact system model.
- Integration with OpenAI and Ollama APIs for LLM access.
- Support for multiple models: GPT-4o, GPT-4, DeepSeekV2.
- Adjustable tuning aggressiveness levels: `fine`, `moderate`, `aggressive`.
- Early stopping mechanism based on predefined performance improvements.
- Context-aware conversation history for iterative LLM interactions.
- Clear evaluation metrics: settling time, rise time, overshoot.

## Project Structure
- `LLMClient`: Handles API communication with OpenAI or Ollama.
- `PIDController`: Defines PID control logic and transfer function.
- `LLM_PID_Tuner`: Manages system response evaluation, communicates with the LLM, and applies PID updates.

## Installation
```bash
pip install -r requirements.txt
```
Ensure you have API access set up for OpenAI or Ollama locally.

## Usage
1. Clone the repository:
```bash
git clone https://github.com/ilijakamenko/LLM_PID_Tuner.git
cd LLM_PID_Tuner
```
2. Set up your `.env` file:
```plaintext
OPENAI_API_KEY=your-openai-api-key
OLLAMA_HOST=http://localhost:11434
```
3. Run the main script:
```bash
python main.py
```

## Quickstart (Notebook)

A Jupyter Notebook is provided in the root directory for quick experimentation.

To use it:
```bash
pip install jupyter
jupyter notebook
```
Then open `quickstart.ipynb`.

## Example
```python
from llm_pid_tuner import LLMClient, PIDController, LLM_PID_Tuner
import control as ctrl

# Define plant
plant = ctrl.TransferFunction([1], [1, 3, 2])

# Initialize modules
llm_client = LLMClient(model_name="gpt-4o")
pid_controller = PIDController(Kp=1.0, Ti=0.5, Td=0.1)
tuner = LLM_PID_Tuner(plant, pid_controller, llm_client, reference_signal=1.0, manipulated_variable=0.0, mode="balanced", aggressiveness="moderate")

# Start tuning
for i in range(15):
    Kp, Ti, Td, metrics = tuner.tune_pid()
    print(f"Iteration {i+1}: Kp={Kp:.3f}, Ti={Ti:.3f}, Td={Td:.3f}, Metrics={metrics}")
```

## Contributing
Contributions are welcome! Please open an issue or submit a pull request.

## Citation
If you use this work in your research or projects, please cite it as:

```bibtex
@INPROCEEDINGS{11091752,
  author={Kamenko, Ilija and Ilic, Slobodan and Congradac, Velimir},
  booktitle={2025 10th International Conference on Smart and Sustainable Technologies (SpliTech)}, 
  title={LLM-based PID controller optimization}, 
  year={2025},
  volume={},
  number={},
  pages={1-6},
  keywords={Adaptation models;Tuners;Computational modeling;Manuals;Robustness;Real-time systems;Stability analysis;Prompt engineering;Tuning;Optimization;LLM;AI;PID controller;tuning;optimization},
  doi={10.23919/SpliTech65624.2025.11091752}}
```

## License
This project is licensed under the MIT License.

## Acknowledgements
- OpenAI API
- Ollama Local LLM Hosting
- Python Control Systems Library

