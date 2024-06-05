import os
import matplotlib.pyplot as plt
from facade import create_model
import torch
import torch.optim as optim
import config
import inspect
import torch.multiprocessing as mp

def save_results(losses, win_rates, policy_losses, value_losses, model_folder, total_moves):
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

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    for i, win_rate in enumerate(win_rates):
        start_epoch = i * config.CHECKPOINT_INTERVAL + 1
        plt.plot(range(start_epoch, start_epoch + len(win_rate)), win_rate, label=f'Checkpoint {start_epoch}')
    plt.xlabel('Epoch')
    plt.ylabel('Win Rate')
    plt.legend()
    plt.title('Win Rate over Checkpoints')
    plt.subplot(1, 2, 2)
    for i, moves in enumerate(total_moves):
        start_epoch = i * config.CHECKPOINT_INTERVAL + 1
        plt.plot(range(start_epoch, start_epoch + len(moves)), moves, label=f'Checkpoint {start_epoch}')
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
