import numpy as np
import os
import matplotlib.pyplot as plt
import torch
import inspect
from facade import MCTS, create_model
import torch.optim as optim

def save_results(losses, win_rates, policy_losses, value_losses, path, avg_moves, checkpoints, config):
    board_size = config['BOARD_SIZE']
    epochs = list(range(1, len(losses) + 1))

    # Save loss plot
    plt.figure()
    plt.plot(epochs, losses, label='Total Loss')
    plt.plot(epochs, policy_losses, label='Policy Loss')
    plt.plot(epochs, value_losses, label='Value Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title(f'Losses for {board_size}x{board_size} Hex')
    plt.savefig(f"{path}/loss.png")
    plt.close()

    # Save win rates and avg moves plot
    for i in range(len(win_rates)):
        if i == 0:
            start_epoch = config['EVALUATION_INTERVAL']
            legend_name = 'Random Agent'
        else:
            checkpoint_epoch = config['CHECKPOINT_INTERVAL'] * i
            start_epoch = checkpoint_epoch if checkpoint_epoch % config['EVALUATION_INTERVAL'] == 0 else \
                (checkpoint_epoch // config['EVALUATION_INTERVAL'] + 1) * config['EVALUATION_INTERVAL']
            legend_name = f'Agent Checkpoint Epoch {checkpoint_epoch}'

        agent_epochs = list(range(start_epoch, start_epoch + len(win_rates[i]) * config['EVALUATION_INTERVAL'], config['EVALUATION_INTERVAL']))
        agent_win_rates = win_rates[i]
        plt.plot(agent_epochs, agent_win_rates, label=f' {legend_name}')
        plt.xlabel('Epochs')
        plt.ylabel('Win Rate')
        plt.legend()
        plt.title(f'Win Rate against {legend_name} for {board_size}x{board_size} Hex')
        plt.savefig(f"{path}/win_rate_agent_{i}.png")
        plt.close()

        agent_avg_moves = avg_moves[i]
        plt.figure()
        plt.plot(agent_epochs, agent_avg_moves, label=f'{legend_name}')
        plt.xlabel('Epochs')
        plt.ylabel('Avg Moves')
        plt.legend()
        plt.title(f'Avg Moves against {legend_name} for {board_size}x{board_size} Hex')
        plt.savefig(f"{path}/avg_moves_agent_{i}.png")
        plt.close()

    # Combined plot for win rates
    plt.figure()
    for i in range(len(win_rates)):
        if i == 0:
            start_epoch = config['EVALUATION_INTERVAL']
            legend_name = 'Random Agent'
        else:
            checkpoint_epoch = config['CHECKPOINT_INTERVAL'] * i
            start_epoch = checkpoint_epoch if checkpoint_epoch % config['EVALUATION_INTERVAL'] == 0 else \
                (checkpoint_epoch // config['EVALUATION_INTERVAL'] + 1) * config['EVALUATION_INTERVAL']
            legend_name = f'Agent Checkpoint Epoch {checkpoint_epoch}'

        agent_epochs = list(range(start_epoch, start_epoch + len(win_rates[i]) * config['EVALUATION_INTERVAL'], config['EVALUATION_INTERVAL']))
        agent_win_rates = win_rates[i]
        plt.plot(agent_epochs, agent_win_rates, label=f'{legend_name}')
    plt.xlabel('Epochs')
    plt.ylabel('Win Rate')
    plt.legend()
    plt.title(f'Win Rates for {board_size}x{board_size} Hex')
    plt.savefig(f"{path}/win_rates_combined.png")
    plt.close()

    # Combined plot for avg moves
    plt.figure()
    for i in range(len(avg_moves)):
        if i == 0:
            start_epoch = config['EVALUATION_INTERVAL']
            legend_name = 'Random Agent'
        else:
            checkpoint_epoch = config['CHECKPOINT_INTERVAL'] * i
            start_epoch = checkpoint_epoch if checkpoint_epoch % config['EVALUATION_INTERVAL'] == 0 else \
                (checkpoint_epoch // config['EVALUATION_INTERVAL'] + 1) * config['EVALUATION_INTERVAL']
            legend_name = f'Agent Checkpoint Epoch {checkpoint_epoch}'

        agent_epochs = list(range(start_epoch, start_epoch + len(avg_moves[i]) * config['EVALUATION_INTERVAL'], config['EVALUATION_INTERVAL']))
        agent_avg_moves = avg_moves[i]
        plt.plot(agent_epochs, agent_avg_moves, label=f'{legend_name}')
        plt.xlabel('Epochs')
        plt.ylabel('Avg Moves')
        plt.legend()
        plt.title(f'Avg Moves for {board_size}x{board_size} Hex')
    plt.savefig(f"{path}/avg_moves_combined.png")
    plt.close()

def save_config_to_file(config, filename="config.py"):
    with open(filename, 'w') as file:
        for name, value in config.items():
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
