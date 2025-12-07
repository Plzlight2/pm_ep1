import torch
import numpy as np
import math
from torch.utils.data import DataLoader, random_split
from src.dataset.pm25_dataset import PM25Dataset
from sklearn.metrics import r2_score
from src.models.resnet import init_resnet

def main():
    # ================== 参数 ==================
    BATCH_SIZE = 32
    EPOCHS = 100
    LR = 5e-4

    torch.manual_seed(42)
    np.random.seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # ================== 数据集 ==================
    dataset = PM25Dataset()

    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=4, pin_memory=True)

    # ================== 模型 ==================
    model, criterion, optimizer = init_resnet(device, LR)

    # ================== 训练 ==================
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device).unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)

        epoch_train_loss = train_loss / len(train_loader.dataset)

        # ---------- val ----------
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device).unsqueeze(1)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)

        epoch_val_loss = val_loss / len(val_loader.dataset)

        # ---------- save best model ----------
        if epoch == 0:
            best_val = epoch_val_loss
            torch.save(model.state_dict(), "best_model.pth")
        else:
            if epoch_val_loss < best_val:
                best_val = epoch_val_loss
                torch.save(model.state_dict(), "best_model.pth")
                print(">>> Saved new best model")

        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f}")

    # ================== 测试 ==================
    # ---------- load best model ----------
    model.load_state_dict(torch.load("best_model.pth", map_location=device))
    print("Loaded best_model.pth")

    model.eval()
    preds, gts = [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device).unsqueeze(1)

            outputs = model(images)
            preds.extend(outputs.cpu().numpy())
            gts.extend(labels.cpu().numpy())

    preds = np.array(preds).flatten()
    gts = np.array(gts).flatten()

    mse = np.mean((preds - gts) ** 2)
    rmse = math.sqrt(mse)
    r2 = r2_score(gts, preds)
    mae = np.mean(np.abs(preds - gts))
    print(f"Test MAE: {mae:.2f}, MSE: {mse:.2f}, RMSE: {rmse:.2f}, R²: {r2:.3f}")

if __name__ == '__main__':
    main()

