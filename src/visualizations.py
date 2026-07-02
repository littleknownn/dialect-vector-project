from src.utils import models_dir
from pathlib import Path
import matplotlib.pyplot as plt
import json

def adapter_training_loss_fig(adapter_name: str, figure_size: tuple[int, int]=(4, 4)) -> plt.Figure:
    trainer_state_path = models_dir / adapter_name / 'checkpoints' / 'trainer_state.json'
    with open(trainer_state_path, 'r') as trainer_state_json:
        trainer_state = json.load(trainer_state_json)
    
    training_steps = [log['step'] for log in trainer_state['log_history'] if 'loss' in log]
    training_losses = [log['loss'] for log in trainer_state['log_history'] if 'loss' in log]

    fig,ax = plt.subplots(figsize=figure_size)
    ax.set_title(f'PEFT Loss on {adapter_name} Corpus')
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Cross-Entropy Loss')
    ax.plot(training_steps, training_losses)
    
    return fig
