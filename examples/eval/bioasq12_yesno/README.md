# BioASQ12 Yes/No Dataset Preprocessing

## Overview
This directory contains preprocessing scripts for the **BioASQ12 Yes/No** dataset. 
The preprocessing pipeline converts raw dataset files into a structured CSV format that we modified for our LLM tournament evaluation.

## Setup Instructions

### **1. Prepare the Data**
Before running the script, download the **training and test data** for **BioASQ Task B (BioASQ 12b: Year 2024)** from the [BioASQ website](https://participants-area.bioasq.org/datasets/).  
**Note:** You must register on the BioASQ website to access the dataset.  

Once downloaded, extract the required dataset files into a designated directory (e.g., `resources`):

Ensure that the extracted folder contains the required input files.
```bash
$ ls resources/
12B1_golden.json  12B2_golden.json  12B3_golden.json  12B4_golden.json  training12b_new.json
```

### **2. Run the Preprocessing Script**
Execute the prepare_data.py script with the dataset directory and dataset name:
```bash
python3 prepare_data.py resources bioasq12-yesno-llm
```

### **3. Output Format**
The script generates a CSV file with the following columns:

* id: Unique identifier for each instance.
* text: The question text.
* exact_answer: The short answer corresponding to the question.
* label: The normalized answer (first letter capitalized).

The dataset is split into training, development, and test sets:

**Note:** 
* The development set consists of **20% of the training instances**, randomly selected with a fixed seed (random seed = 0).
* The test set is identical to the official set provided by the BioASQ organizers.
* Train Data: 1,086 instances
* Dev Data: 271 instances (randomly picked)
* Test Data: 102 instances
