import torch
import numpy as np
from opacus.utils.batch_memory_manager import BatchMemoryManager
from tqdm import tqdm
from utils.model_loader import get_resnet18, get_cnn, get_PBS_LSTM, get_PBS_FF
from utils.opacus_engine_tools import get_privacy_dataloader
from opacus import PrivacyEngine
from torch.utils.tensorboard import SummaryWriter

def accuracy(preds, labels):
    return (preds == labels).mean()

def classification_valid(model_type, model, valid_loader, criterion, epoch,
                            device,
                            valid_logger_step=20,
                            other_configs=None):
    valid_losses = []
    valid_acc = []
    model.eval()
    with torch.no_grad():
        for i, (inputs, target) in enumerate(valid_loader):
            inputs = inputs.to(device)
            target = target.to(device)
            if model_type.find("LSTM") != -1:
                n_layer = other_configs['n_layer']
                hidden_size = other_configs['hidden_size']
                h0 = torch.randn(n_layer, inputs.shape[0], hidden_size).to(device)
                c0 = torch.randn(n_layer, inputs.shape[0], hidden_size).to(device)
                output, _ = model(inputs, h0, c0)
            else:
                output = model(inputs)
            loss = criterion(output, target)
            preds = np.argmax(output.detach().cpu().numpy(), axis=1)
            labels = target.detach().cpu().numpy()
            acc = accuracy(preds, labels)
            valid_losses.append(loss.item())
            valid_acc.append(acc)
            
            # valid_logger_step = int(train_logger_step / 10) if int(train_logger_step / 10) > 1 else 1
            if (i + 1) % valid_logger_step == 0:
                print(
                    f"\tValid Epoch: {epoch} \t"
                    f"Loss: {np.mean(valid_losses):.6f} "
                    f"Acc@1: {np.mean(valid_acc) * 100:.6f}"
                )
    '''
    if early_stop is not None:
        early_stop(np.mean(valid_losses), model)
        early_stop_result =  early_stop.early_stop
    else:
        early_stop_result = False
    '''
    early_stop_result = False
    return np.mean(valid_acc), np.mean(valid_losses), early_stop_result

def classification_train(model_type, model, train_loader, optimizer, criterion, epoch, 
                            device, privacy_engine,
                            MAX_PHYSICAL_BATCH_SIZE, DELTA,
                            train_logger_step=200,
                            other_configs=None):
    model.train()

    train_losses = []
    train_acc = []
    
    
    if privacy_engine is not None:
        with BatchMemoryManager(
            data_loader=train_loader, 
            max_physical_batch_size=MAX_PHYSICAL_BATCH_SIZE, 
            optimizer=optimizer
        ) as memory_safe_data_loader:
            for i, (inputs, labels) in enumerate(memory_safe_data_loader):   
                optimizer.zero_grad()
                inputs = inputs.to(device)
                labels = labels.to(device)

                if model_type.find("LSTM") != -1:
                    n_layer = other_configs['n_layer']
                    hidden_size = other_configs['hidden_size']
                    h0 = torch.randn(n_layer, inputs.shape[0], hidden_size).to(device)
                    c0 = torch.randn(n_layer, inputs.shape[0], hidden_size).to(device)
                    output, _ = model(inputs, h0, c0)
                else:
                    output = model(inputs)
                loss = criterion(output, labels)

                preds = np.argmax(output.detach().cpu().numpy(), axis=1)
                labels = labels.detach().cpu().numpy()

                # measure accuracy and record loss
                acc = accuracy(preds, labels)

                train_losses.append(loss.item())
                train_acc.append(acc)

                loss.backward()
                optimizer.step()

                if (i+1) % train_logger_step == 0:
                    epsilon = privacy_engine.get_epsilon(DELTA)
                    print(
                        f"\tTrain Epoch: {epoch} \t"
                        f"Loss: {np.mean(train_losses):.6f} "
                        f"Acc@1: {np.mean(train_acc) * 100:.6f} "
                        f"(ε = {epsilon:.2f}, δ = {DELTA}"
                    )  
    else:
        for i, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            inputs = inputs.to(device)
            labels = labels.to(device)

            if model_type.find("LSTM") != -1:
                n_layer = other_configs['n_layer']
                hidden_size = other_configs['hidden_size']
                h0 = torch.randn(n_layer, inputs.shape[0], hidden_size).to(device)
                c0 = torch.randn(n_layer, inputs.shape[0], hidden_size).to(device)
                output, _ = model(inputs, h0, c0)
            else:
                output = model(inputs)
            loss = criterion(output, labels)

            preds = np.argmax(output.detach().cpu().numpy(), axis=1)
            labels = labels.detach().cpu().numpy()

            # measure accuracy and record loss
            acc = accuracy(preds, labels)

            train_losses.append(loss.item())
            train_acc.append(acc)

            loss.backward()
            optimizer.step()

            if (i+1) % train_logger_step == 0:
                print(
                    f"\tTrain Epoch: {epoch} \t"
                    f"Loss: {np.mean(train_losses):.6f} "
                    f"Acc@1: {np.mean(train_acc) * 100:.6f}"
                )
    if privacy_engine is not None:
        epsilon = privacy_engine.get_epsilon(DELTA)
    else:
        epsilon = 0.0

    return np.mean(train_acc), np.mean(train_losses), epsilon

def privacy_model_train_valid(model_name, target_label, target_train_loader, valid_loader,
                    device, label_num, summary_writer_path,
                    LR, EPSILON, EPOCH_SET_EPSILON, DELTA, MAX_GRAD_NORM, MAX_PHYSICAL_BATCH_SIZE, EPOCHS,
                    other_configs=None):
    privacy_engine = PrivacyEngine() if EPSILON > 0 else None
    if model_name == 'resnet-18-split':
        model, criterion, optimizer = get_resnet18(device, LR, num_classes=label_num)
    elif model_name == 'cnn-split':
        model, criterion, optimizer = get_cnn(device, LR, num_classes=label_num)
    elif model_name == 'LSTM-split':
        n_layer = other_configs['n_layer']
        hidden_size = other_configs['hidden_size']
        vocab_size = other_configs['vocab_size']
        embedding_size = other_configs['embedding_size']
        label_distributions = other_configs['label_distributions']
        opacus_flag = (privacy_engine is not None)
        model, criterion, optimizer = get_PBS_LSTM(device, LR, vocab_size, label_distributions, embedding_size, hidden_size, n_layer, opacus_flag)
    elif model_name == 'FF-split':
        hidden_size = other_configs['hidden_size']
        assert len(hidden_size) == 2
        vocab_size = other_configs['vocab_size']
        embedding_size = other_configs['embedding_size']
        label_distributions = other_configs['label_distributions']
        model, criterion, optimizer = get_PBS_FF(device, LR, vocab_size, label_distributions, embedding_size, hidden_size[0], hidden_size[1])
    else:
        model, criterion, optimizer = None, None, None
    if not EPOCH_SET_EPSILON:
        model, optimizer, target_train_loader = get_privacy_dataloader(privacy_engine, model, optimizer, target_train_loader, EPOCHS, EPSILON, DELTA, MAX_GRAD_NORM)
    # for epoch in tqdm(range(EPOCHS), desc="Epoch", unit="epoch"):
    if summary_writer_path != "":
        summary_writer = SummaryWriter(summary_writer_path)
    else:
        summary_writer = None
    for epoch in range(EPOCHS):
        print("epoch: {}".format(epoch))
        model.train()
        optimizer.zero_grad()
        if EPOCH_SET_EPSILON:
            model, optimizer, target_train_loader = get_privacy_dataloader(privacy_engine, model, optimizer, target_train_loader, 1, EPSILON, DELTA, MAX_GRAD_NORM)
        train_acc, train_loss, epsilon_consume = classification_train(model_name, model, target_train_loader, optimizer, criterion, epoch + 1, 
                                                                        device, privacy_engine,
                                                                        MAX_PHYSICAL_BATCH_SIZE, DELTA, other_configs=other_configs)
        valid_acc, valid_loss, is_early_stop = classification_valid(model_name, model, valid_loader, criterion, epoch+1,
                                                                        device, other_configs=other_configs)

        # all_epsilon_consume += epsilon_consume
        if summary_writer is not None:
            summary_writer.add_scalar('train_acc', train_acc, epoch)
            summary_writer.add_scalar('train_loss', train_loss, epoch)
            summary_writer.add_scalar('valid_acc', valid_acc, epoch)
            summary_writer.add_scalar('valid_loss', valid_loss, epoch)
            summary_writer.add_scalar('all_epsilon_consume', epsilon_consume, epoch)
        if is_early_stop:
            print('Early Stop!')
    if summary_writer is not None:
        summary_writer.add_text('{}_train_acc'.format(target_label), str(train_acc), 0)
        summary_writer.add_text('{}_train_loss'.format(target_label), str(train_loss), 0)
        summary_writer.add_text('{}_valid_acc'.format(target_label), str(valid_acc), 0)
        summary_writer.add_text('{}_valid_loss'.format(target_label), str(valid_loss), 0)
        summary_writer.add_text('{}_all_epsilon_consume'.format(target_label), str(epsilon_consume), 0)
    print('{}_train_acc: {}'.format(target_label, train_acc))
    print('{}_train_loss: {}'.format(target_label, train_loss))
    print('{}_valid_acc: {}'.format(target_label, valid_acc))
    print('{}_valid_loss: {}'.format(target_label, valid_loss))
    print('{}_all_epsilon_consume: {}'.format(target_label, epsilon_consume))
    print('=========================================================================')
    # del model, criterion, optimizer
    return train_acc, train_loss, valid_acc, valid_loss, epsilon_consume

    