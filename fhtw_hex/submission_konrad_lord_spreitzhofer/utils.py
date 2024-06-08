import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import config
import inspect
from facade import MCTS, create_model
import torch.optim as optim


def save_results(losses, win_rates, policy_losses, value_losses, model_folder, avg_moves):
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
    num_agents = len(win_rates)
    num_plots = (num_agents + 2) // 3  # Calculate the number of plots needed
    for i in range(num_plots):
        start_index = i * 3
        end_index = min(start_index + 3, num_agents)

        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        for j in range(start_index, end_index):
            agent_win_rate = win_rates[j]
            start_epoch = j * config.CHECKPOINT_INTERVAL + config.CHECKPOINT_INTERVAL
            plt.plot(range(start_epoch, start_epoch + len(agent_win_rate)), agent_win_rate,
                     label=f'Checkpoint {start_epoch}')
        plt.xlabel('Epoch')
        plt.ylabel('Win Rate')
        plt.legend()
        plt.title('Win Rate over Checkpoints')

        plt.subplot(1, 2, 2)
        for j in range(start_index, end_index):
            agent_moves = avg_moves[j]
            start_epoch = j * config.CHECKPOINT_INTERVAL + config.CHECKPOINT_INTERVAL
            plt.plot(range(start_epoch, start_epoch + len(agent_moves)), agent_moves, label=f'Checkpoint {start_epoch}')
        plt.xlabel('Epoch')
        plt.ylabel('Moves')
        plt.legend()
        plt.title('Moves over Checkpoints')
        plt.savefig(os.path.join(model_folder, f'wr_moves_{i + 1}.png'))
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
