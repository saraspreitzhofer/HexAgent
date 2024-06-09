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
        plt.plot(agent_epochs, agent_win_rates, label=legend_name)
        plt.xlabel('Epochs')
        plt.ylabel('Win Rate')
        plt.legend()
        plt.title(f'Win Rate of {legend_name} for {board_size}x{board_size} Hex')
        plt.savefig(f"{path}/win_rate_agent_{i}.png")
        plt.close()

        agent_avg_moves = avg_moves[i]
        plt.figure()
        plt.plot(agent_epochs, agent_avg_moves, label=legend_name)
        plt.xlabel('Epochs')
        plt.ylabel('Avg Moves')
        plt.legend()
        plt.title(f'Avg Moves of {legend_name} for {board_size}x{board_size} Hex')
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
        plt.plot(agent_epochs, agent_win_rates, label=legend_name)
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
        plt.plot(agent_epochs, agent_avg_moves, label=legend_name)
    plt.xlabel('Epochs')
    plt.ylabel('Avg Moves')
        plt.legend()
        plt.title(f'Avg Moves for {board_size}x{board_size} Hex')
    plt.savefig(f"{path}/avg_moves_combined.png")
    plt.close()
