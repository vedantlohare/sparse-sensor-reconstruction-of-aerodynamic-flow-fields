import torch
import torch.optim as optim
import copy

def train_model(model, train_loader, val_loader, criterion, num_epochs=100, 
                lr=1e-3, patience=15, weight_decay=1e-4):
    """
    Standard PyTorch training loop with validation early stopping and learning-rate decay.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        print(f"Training on GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Training on CPU")
        
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    best_val_loss = float('inf')
    best_model_weights = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0
    
    train_history = []
    val_history = []
    
    for epoch in range(num_epochs):
        # Training Phase
        model.train()
        running_train_loss = 0.0
        
        for batch_y, batch_x in train_loader:
            batch_y = batch_y.to(device)
            batch_x = batch_x.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            preds = model(batch_y)
            
            # Compute loss
            loss = criterion(preds, batch_x)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            running_train_loss += loss.item() * batch_y.size(0)
            
        epoch_train_loss = running_train_loss / len(train_loader.dataset)
        train_history.append(epoch_train_loss)
        
        # Validation Phase
        model.eval()
        running_val_loss = 0.0
        
        with torch.no_grad():
            for batch_y, batch_x in val_loader:
                batch_y = batch_y.to(device)
                batch_x = batch_x.to(device)
                
                preds = model(batch_y)
                loss = criterion(preds, batch_x)
                running_val_loss += loss.item() * batch_y.size(0)
                
        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        val_history.append(epoch_val_loss)
        
        scheduler.step(epoch_val_loss)
        
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_model_weights = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {epoch_train_loss:.6f} | Val Loss: {epoch_val_loss:.6f}")
            
        if epochs_no_improve >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break
            
    # Load best model weights
    model.load_state_dict(best_model_weights)
    return model, train_history, val_history
