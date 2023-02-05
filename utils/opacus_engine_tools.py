from opacus import PrivacyEngine


def get_privacy_dataloader(privacy_engine, model, optimizer, train_loader, EPOCHS, EPSILON, DELTA, MAX_GRAD_NORM):
    if EPSILON < 100000.0:
        model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            epochs=EPOCHS,
            target_epsilon=EPSILON,
            target_delta=DELTA,
            max_grad_norm=MAX_GRAD_NORM,
        )

        print(f"Using sigma={optimizer.noise_multiplier} and C={MAX_GRAD_NORM}")
    return model, optimizer, train_loader