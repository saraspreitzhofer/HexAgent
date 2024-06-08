import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import config
import inspect
from facade import MCTS, create_model
import torch.optim as optim

import matplotlib.pyplot as plt
import os
import numpy as np
import config


def save_results(losses, win_rates, policy_losses, value_losses, model_folder, avg_moves, checkpoints):
    epochs = range(1, len(losses) + 1)

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, losses, label='Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss over Epochs')

    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(policy_losses) + 1), policy_losses, label='Policy Loss')
    plt.plot(range(1, len(value_losses) + 1), value_losses, label='Value Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Policy and Value Losses over Epochs')
    plt.savefig(os.path.join(model_folder, 'loss.png'))
    plt.close()

    # Plot for Win Rate and Moves over Checkpoints
    num_plots = (len(win_rates) + 2) // 3  # Anzahl der benötigten Plots

    for plot_idx in range(num_plots):
        plt.figure(figsize=(12, 6))

        for i in range(plot_idx * 3, min((plot_idx + 1) * 3, len(win_rates))):
            if i == 0:
                label = "Random_Agent"
            else:
                label = f"Agent_Checkpoint_Epoch_{(i - 1) * config.CHECKPOINT_INTERVAL + config.CHECKPOINT_INTERVAL}"
            plt.subplot(1, 2, 1)
            plt.plot(range(config.EVALUATION_INTERVAL, config.EVALUATION_INTERVAL * len(win_rates[i]) + 1,
                           config.EVALUATION_INTERVAL), win_rates[i], label=label)

        plt.xlabel('Epoch')
        plt.ylabel('Win Rate')
        plt.legend()
        plt.title('Win Rate over Checkpoints')

        for i in range(plot_idx * 3, min((plot_idx + 1) * 3, len(avg_moves))):
            if i == 0:
                label = "Random_Agent"
            else:
                label = f"Agent_Checkpoint_Epoch_{(i - 1) * config.CHECKPOINT_INTERVAL + config.CHECKPOINT_INTERVAL}"
            plt.subplot(1, 2, 2)
            plt.plot(range(config.EVALUATION_INTERVAL, config.EVALUATION_INTERVAL * len(avg_moves[i]) + 1,
                           config.EVALUATION_INTERVAL), avg_moves[i], label=label)

        plt.xlabel('Epoch')
        plt.ylabel('Moves')
        plt.legend()
        plt.title('Moves over Checkpoints')
        plt.savefig(os.path.join(model_folder, f'wr_moves_{plot_idx + 1}.png'))
        plt.close()

    plt.subplot(1, 2, 2)
    for i, moves in enumerate(avg_moves):
        label = 'Random_Agent' if i == 0 else f'Agent_Checkpoint_Epoch_{(i) * config.CHECKPOINT_INTERVAL}'
        plt.plot(range(config.EVALUATION_INTERVAL, config.EVALUATION_INTERVAL + len(moves) * config.EVALUATION_INTERVAL, config.EVALUATION_INTERVAL), moves, label=label)
    plt.xlabel('Epoch')
    plt.ylabel('Moves')
    plt.legend()
    plt.title('Moves over Checkpoints')
    plt.savefig(os.path.join(model_folder, 'wr_moves.png'))
    plt.close()

def save_config_to_file(config_module, filename="config.py"):
    with open(filename, 'w') as file:
        for name, value in inspect.getmembers(config_module):
            if not name.startswith("__") and not inspect.ismodule(value) and not inspect.isfunction(value):
                file.write(f"{name} = {value}\n")

def save_checkpoint(model, optimizer, epoch, model_folder, filename='checkpoint.pth.tar'):
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    filepath = os.path.join(model_folder, filename)
    torch.save(state, filepath)

def load_checkpoint(filepath, board_size):
    model = create_model(board_size)
    optimizer = optim.Adam(model.parameters())
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer

def setup_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using device: {device} ({torch.cuda.get_device_name(0)})")
    else:
        device = torch.device('cpu')
        print("CUDA device not found. Using CPU.")
    return device
