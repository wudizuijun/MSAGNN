# MSAGNN
A Dual-Path GNN Integrating Static and Dynamic Graphs for Soft Sensor Modeling in Industrial Processes

### Model and Results
The figure below illustrates the proposed Multi-Source Attention Graph Neural Network (MSAGNN) framework, which consists of industrial process data sampling, graph structure learning, attention mechanism fusion, GRU-based temporal modeling, and final output prediction.

![Image](https://github.com/user-attachments/assets/d8de9cce-b41d-4650-ab54-baba0b31b93f)



![Image](https://github.com/user-attachments/assets/d3335152-3719-48c4-ba15-324b805f8428)


## Requirements 
We recommend using **Python 3.8 to 3.10**, which has been tested and verified for this project.  
Please ensure that the required dependencies are installed. You can install them all at once using the following command:
```bash
pip install -r requirements.txt
```

##  Dataset

This project uses the following industrial process datasets:
- **TE Dataset** (Tennessee Eastman Process): Commonly used for industrial process modeling and fault diagnosis research;
- **DC Dataset** (Distillation Column Process): A multivariate time-series dataset collected from a distillation system.
All datasets are included in this repository and located in the `./data/` directory. No additional download is required.

## Experiment Configuration and Execution

### Parameter Configuration
The main experimental parameters are defined in the `parser_args.py` file and can be modified manually or overridden via command-line arguments.

### Model Training
Run the training script using the command line:
```bash
python main.py --model_type AttGRU --dataset TE --target v10 --window_size 16 --horizon 1
```
Or use the provided shell script:
```
bash run.sh
```
### Output
After training, the prediction results and logs will be automatically saved in the `./results/` directory.
## Citation
@misc{wang2025MSAGNN,                        
  title  = {A Multi-Source Attention Graph Neural Network for Modeling Long and Short-Term Dependencies in Chemical Process Forecasting},
  author = {Jian Long and bin wang and Haifei Peng and Hengmin Zhang},
  year   = {2025}
}
