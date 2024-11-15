from game import DotsAndBoxes
import model
import player
import wandb

def main():
    is_human = False
    do_train = True
    use_wandb = True
    if not is_human and do_train and use_wandb:
        wandb.login()

    epochs = 10000
    verification_epochs = 10
    score = {'P1': 0, 'P2': 0, 'Tie': 0}
    extend_table = False

    rows = 2
    alpha = 0.1
    gamma = 0.4
    epsilon = 0.1

    debug = False

    model_name = 'q_table_' + str(rows)

    if not is_human and do_train and use_wandb:
        run = wandb.init(
            # Set the project where this run will be logged
            project="box-em-all",
            # Track hyperparameters and run metadata
            config={
                "rows": rows,
                "alpha": alpha,
                "gamma": gamma,
                "epsilon": epsilon,
                "epochs": epochs,
                "verification_epochs": verification_epochs,
                "game-state": "try with rotation invariant state",
            },
            tags=["q-learning", "dots-and-boxes"]
        )
    
    # Model
    if do_train:
        q_learning = model.QLearning(alpha=alpha, gamma = gamma, epsilon = epsilon, q_table={} if not extend_table else model.load(model_name).q_table)
    else:
        q_learning = model.load(model_name)

    if is_human:
        player_1 = player.Human(1, 'Human1')
        player_2 = player.ComputerQTable(2, 'QTable2', q_learning)
        game = DotsAndBoxes(rows=rows, cols=rows, player_1=player_1, player_2=player_2)
        game.play(print_board=is_human)
        return

    for epoch in range(epochs if do_train else 1):
        score = {'P1': 0, 'P2': 0, 'Tie': 0}
        if debug or epoch + 1 % 1000 == 0: 
            print(f"epoch: {epoch} Training: {do_train} ****************************************************")
        # Player 1
        if do_train:
            player_1 = player.ComputerRandom(1, 'Random1')
            #player_1 = player.ComputerGreedy(1, 'Greedy1')
            player_2 = player.ComputerQLearning(2, 'QLearning2', q_learning)
            # Play game
            game = DotsAndBoxes(rows=rows, cols=rows, player_1=player_1, player_2=player_2)
            game.play()
        
        verification_player_1 = player.ComputerGreedy(1, 'Greedy1')
        verification_player_2 = player.ComputerQTable(2, 'QTable2', q_learning)
        for _ in range(verification_epochs):
            verification_game = DotsAndBoxes(rows=rows, cols=rows, player_1=verification_player_1, player_2=verification_player_2)
            verification_game.play()
            # Update player score
            if verification_game.player_1.player_score > verification_game.player_2.player_score:
                score['P1'] += 1
            elif verification_game.player_1.player_score < verification_game.player_2.player_score:
                score['P2'] += 1
            else:
                score['Tie'] += 1

        if debug or (((epoch % 1000) == 0) and not use_wandb): 
            # Print final score accross all verification epochs
            print("--------------------------------------------------------------------------------")
            print(f"Final Score in epoch {epoch}: {score}")
            print(f"P1 ({verification_game.player_1.player_name}) Win: {round(score['P1'] / verification_epochs * 100, 2)}%")
            print(f"P2 ({verification_game.player_2.player_name}) Win: {round(score['P2'] / verification_epochs * 100, 2)}%")
            print("--------------------------------------------------------------------------------")    

        if not is_human and do_train and use_wandb:
            wandb.log({"epoch": epoch, "win-greedy": round(score['P1'] / verification_epochs * 100, 2), "win-qplayer": round(score['P2'] / verification_epochs * 100, 2), "tie": round(score['Tie'] / verification_epochs * 100, 2)})

    

        
    # Save model
    if do_train:
        model.save(model_name, q_learning)

if __name__ == "__main__":
    main()