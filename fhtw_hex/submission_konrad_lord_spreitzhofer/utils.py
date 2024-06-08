import numpy as np
import os
import matplotlib.pyplot as plt
import torch
import config
import inspect
from facade import MCTS, create_model
import torch.optim as optim

def save_results(losses, win_rates, policy_losses, value_losses, best_model_path, avg_moves, checkpoints):
    epochs = len(losses)

    # Plotting loss
    plt.figure()
    plt.plot(range(epochs), losses, label="Total Loss")
    plt.plot(range(epochs), policy_losses, label="Policy Loss")
    plt.plot(range(epochs), value_losses, label="Value Loss")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss over Epochs')
    plt.savefig(os.path.join(best_model_path, 'loss.png'))
    plt.close()

    # Plotting win rates and moves
    for i in range(len(win_rates)):
        agent_win_rates = win_rates[i]
        agent_avg_moves = avg_moves[i]

        # Calculate epochs for this agent
        if i == 0:
            agent_epochs = list(range(config.EVALUATION_INTERVAL, epochs + 1, config.EVALUATION_INTERVAL))
            legend_name = 'Random_Agent'
        else:
            start_epoch = config.CHECKPOINT_INTERVAL * i
            agent_epochs = list(range(start_epoch, epochs + 1, config.EVALUATION_INTERVAL))
            legend_name = f'Agent_Epoch_{start_epoch}'

        plt.figure()
        plt.plot(agent_epochs, agent_win_rates, label=f'{legend_name} Win Rate')
        plt.xlabel('Epoch')
        plt.ylabel('Win Rate')
        plt.legend()
        plt.title('Win Rate over Checkpoints')
        plt.savefig(os.path.join(best_model_path, f'win_rate_{i}.png'))
        plt.close()

        plt.figure()
        plt.plot(agent_epochs, agent_avg_moves, label=f'{legend_name} Avg. Moves')
        plt.xlabel('Epoch')
        plt.ylabel('Moves')
        plt.legend()
        plt.title('Moves over Checkpoints')
        plt.savefig(os.path.join(best_model_path, f'avg_moves_{i}.png'))
        plt.close()

    # Plotting combined win rates and moves
    plt.figure()
    for i in range(len(win_rates)):
        agent_win_rates = win_rates[i]
        if i == 0:
            agent_epochs = list(range(config.EVALUATION_INTERVAL, epochs + 1, config.EVALUATION_INTERVAL))
            label = 'Random_Agent'
        else:
            start_epoch = config.CHECKPOINT_INTERVAL * i
            agent_epochs = list(range(start_epoch, epochs + 1, config.EVALUATION_INTERVAL))
            label = f'Agent_Epoch_{start_epoch}'
        plt.plot(agent_epochs, agent_win_rates, label=label)
    plt.xlabel('Epoch')
    plt.ylabel('Win Rate')
    plt.legend()
    plt.title('Win Rate over Checkpoints')
    plt.savefig(os.path.join(best_model_path, 'win_rate_combined.png'))
    plt.close()

    plt.figure()
    for i in range(len(avg_moves)):
        agent_avg_moves = avg_moves[i]
        if i == 0:
            agent_epochs = list(range(config.EVALUATION_INTERVAL, epochs + 1, config.EVALUATION_INTERVAL))
            label = 'Random_Agent'
        else:
            start_epoch = config.CHECKPOINT_INTERVAL * i
            agent_epochs = list(range(start_epoch, epochs + 1, config.EVALUATION_INTERVAL))
            label = f'Agent_Epoch_{start_epoch}'
        plt.plot(agent_epochs, agent_avg_moves, label=label)
    plt.xlabel('Epoch')
    plt.ylabel('Moves')
    plt.legend()
    plt.title('Moves over Checkpoints')
    plt.savefig(os.path.join(best_model_path, 'avg_moves_combined.png'))
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
