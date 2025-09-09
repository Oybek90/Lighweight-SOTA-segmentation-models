import torch
import torch.nn as nn
import torch.optim as optim
from utils import dice_score, iou_score
import numpy as np
import pandas as pd
import json
import time
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score
from thop import profile

def train_model(model, model_name, criterion, train_loader, test_loader, device, datset_name):
    """
    Train and evaluate multiple models on the provided data loaders.    
    Args:
        models (list): List of model instances to train.
        models_name (list): List of names corresponding to each model.
        train_loader (DataLoader): DataLoader for training data.
        test_loader (DataLoader): DataLoader for validation data.
        device (torch.device): Device to run the training on (CPU or GPU).
    """
    Results = {}
    model = model.float()
    model.to(device)  # Move model to device
    # Loss and Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-5)
    # Learning Rate Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    # Training Loop
    num_epochs = 200
    patience = 10
    best_val_loss = float('inf')
    early_stopping_counter = 0
    train_losses = []
    val_losses = []
    all_results = []

    start_time = time.time()
    for epoch in range(num_epochs):
        model.train()
        running_train_loss = 0.0
        for images, masks in tqdm(train_loader):
            images, masks = images.to(device).float(), masks.to(device).float()
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item()
        
        avg_train_loss = running_train_loss / len(train_loader)
        # print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}")
        train_losses.append(avg_train_loss)

        # Validation phase
        model.eval()
        ranning_val_loss = 0.0
        with torch.no_grad():
            for images, masks in test_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                ranning_val_loss += loss.item()
        avg_val_loss = ranning_val_loss / len(test_loader)
        val_losses.append(avg_val_loss)
        # print(f"Epoch [{epoch+1}/{num_epochs}], Val Loss: {avg_val_loss:.4f}")

        # Step the scheduler
        scheduler.step(avg_val_loss)

        # print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            early_stopping_counter = 0
            torch.save(model.state_dict(), f"trained_models/best_{model_name}_{datset_name}_model.pth")
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= patience:
                # print(f"Early stopping triggered at epoch {epoch+1}")
                break
    # Save loss history to csv file
    df_losses = pd.DataFrame({
        'epoch': list(range(1, len(train_losses) + 1)),
        'train_loss': train_losses, 
        'val_loss': val_losses
        })
    df_losses.to_csv(f"models_history/loss_history_{model_name}_{datset_name}.csv", index=False)
    # train_time = (time.time() - start_time) / (epoch+1)
    # print(f"Training Time per epoch: {train_time:.2f} s")

    # Evaluation on test set
    model.eval()
    test_loss = 0.0
    all_labels = []  # Store all ground truth masks
    all_preds = []  # Store all predicted masks
    dice_total = 0.0
    iou_total = 0.0  
    with torch.no_grad():
        for images, masks in tqdm(test_loader):
            images, masks = images.to(device).float(), masks.to(device).float()
            start_time = time.time()
            outputs = model(images)
            infer_time = time.time() - start_time
            loss = criterion(outputs, masks)
            test_loss += loss.item()

            outputs = torch.sigmoid(outputs) > 0.5  # Convert to binary mask
            outputs_binary = outputs.squeeze(1)  # Remove extra channel dimension
            masks = masks.squeeze(1)  # Ensure ground truth also has correct dimensions

            mem_used = torch.mps.current_allocated_memory() / 1e9
            
            all_labels.extend(masks.cpu().numpy().tolist())
            all_preds.extend(outputs_binary.cpu().numpy().tolist())
            dice = dice_score(outputs_binary, masks)
            iou = iou_score(outputs_binary, masks)

            dice_total += dice.item()
            iou_total += iou.item()

    all_preds_np = np.array(all_preds).astype(np.uint8).flatten()
    all_labels_np = np.array(all_labels).astype(np.uint8).flatten()

    avg_test_loss = test_loss / len(test_loader)
    avg_iou = iou_total / len(test_loader)
    avg_dice = dice_total / len(test_loader)
    precision = precision_score(all_labels_np, all_preds_np)
    recall = recall_score(all_labels_np, all_preds_np)
    f1 = f1_score(all_labels_np, all_preds_np)
    total_params = sum(p.numel() for p in model.parameters())
    # Calculate FLOPs and parameters
    flops, params = profile(model, inputs=(images,))
    
    
    all_results.append({
        'model': model_name,
        'avg_test_loss': avg_test_loss,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'avg_iou': avg_iou,
        'avg_dice': avg_dice,
        'params': total_params,
        'flops': flops,
        'infer_time': infer_time,
        'mem_used': mem_used
    })
    Results[model_name] = all_results

            
    print(f"\nTest Loss: {avg_test_loss:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"IoU: {avg_iou:.4f}")
    print(f"Dice: {avg_dice:.4f}")
    
    print("Training complete! 🎉")
    # Save Results to JSON file
    with open(f"models_history/results_of_{model_name}_on_{datset_name}.json", "w") as f:
        json.dump(Results, f, indent=4)

