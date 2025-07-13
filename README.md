# Fune-tuning gpt2-kenyan-clinical
This repository demonstrates how Large Language Models (LLMs) can assist nurses in clinical decision-making, especially in resource-limited settings. It supports scenarios where nurses, often lacking specialist support, must handle real-world clinical cases across diverse counties and healthcare facility levels. The repository includes pipelines for data annotation, model training, inference, and evaluation using two open-source LLMs.
## Repository Structure
* data_annotation/

Contains scripts for generating clinician-oriented prompts using Phi-4 on a predefined dataset.
* train/

Includes training scripts for fine-tuning one open-source model:

    GPT-2 (gpt2-medium)

* inference/

Step-by-step pipeline to evaluate the fine-tuned models:

    step1_generate_Response/:
    Generates answers from natural language prompts using the fine-tuned models.

    step2_compute_metrics/:
    Computes quantitative metrics (e.g., ROUGE score) to evaluate model performance.
