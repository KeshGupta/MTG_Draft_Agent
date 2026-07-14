

# MODEL ARCHITECHTURE

    # Draft a batch of decks
    # Play all against a chosen/random combination of decks from "best drafted decks"
    # Compute Reward:
        # r = -1 if loss, 1 if win, 0 if draw   
        # R = #wins/#loss
    # compute loss = PPOClipLoss()
    # back propagate and update model weights by learning rate
