import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

batch_size = 512
lr = 0.001
epochs = 1000
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def fen2matrix(fen):
    fen_spl = fen.split()
    piece_plac = fen_spl[0]
    is_white = fen_spl[1] == "w"
    matrix_pos = np.zeros([12, 8, 8], dtype=np.float32)

    pieces = list("RNBQKPrnbqkp")
    col = 0
    row = 0
    for c in piece_plac:
        if c.isdigit():
            col += int(c)
        elif c == '/':
            row += 1
            col = 0
        else:
            for i, p in enumerate(pieces):
                if c == p:
                    if is_white:
                        matrix_pos[i, row, col] = 1
                    else:
                        matrix_pos[i, 7 - row, col] = 1
                    break
            col += 1
    return matrix_pos

def augment_fen(matrix_pos):
    if np.random.rand() > 0.5:
        matrix_pos = np.flip(matrix_pos, axis=1)  # Horizontale Spiegelung
    if np.random.rand() > 0.5:
        matrix_pos = np.flip(matrix_pos, axis=2)  # Vertikale Spiegelung
    return torch.tensor(matrix_pos.copy())

def eval2score(evaluation):
    evaluation = str(evaluation)
    if evaluation.startswith("#"):
        mate_moves = int(evaluation[1:])
        cp = (21 - min(10, abs(mate_moves))) * 100
    else:
        cp = float(evaluation)
    score = 1 / (1 + np.exp(-0.004 * cp))
    return score

def convert_evaluation(fen, evaluation):
    score = eval2score(evaluation)
    if fen.split()[1] == 'b':
        return 1 - score
    return score

class ChessPositionsDataset(Dataset):
    def __init__(self, csv_file):
        self.positions = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, idx):
        fen = self.positions.iloc[idx]["FEN"]
        evaluation = self.positions.iloc[idx]["Bewertung"]
        matrix_pos = fen2matrix(fen)
        matrix_pos = augment_fen(matrix_pos)
        score = convert_evaluation(fen, evaluation)
        return matrix_pos, torch.tensor([score], dtype=torch.float32)

class ChessNet(nn.Module):
    def __init__(self):
        super(ChessNet, self).__init__()
        self.conv1 = nn.Conv2d(12, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(256)

        self.fc1 = nn.Linear(256 * 1 * 1, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 1)

        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))  
        x = F.max_pool2d(x, 2, 2)  
        x = F.relu(self.bn2(self.conv2(x)))  
        x = F.max_pool2d(x, 2, 2)  
        x = F.relu(self.bn3(self.conv3(x)))  
        x = F.max_pool2d(x, 2, 2)

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


if __name__ == '__main__':
    data_url = "train.csv"
    chess_dataset = ChessPositionsDataset(csv_file=data_url)
    dataloader = DataLoader(chess_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    net = ChessNet().to(device)
    criterion = nn.SmoothL1Loss(beta=0.01)
    optimizer = optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter()

    net.train()
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(epochs):
        running_loss = 0.0
        for i, (inputs, scores) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}")):
            inputs, scores = inputs.to(device), scores.to(device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = net(inputs)
                loss = criterion(outputs, scores)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

            if i % 100 == 0:
                writer.add_scalar('Loss/train', loss.item(), epoch * len(dataloader) + i)

        print(f"Epoch {epoch + 1}, Loss: {running_loss / len(dataloader):.4f}")
        scheduler.step()

        os.makedirs("model", exist_ok=True)
        torch.save(net.state_dict(), f"model/epoch_{epoch + 1}.pth")

    writer.close()
    print('Finished Training')
