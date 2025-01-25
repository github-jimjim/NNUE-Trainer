import torch
from nnue import ChessNet, fen2matrix

def evaluate_fen(fen, model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = ChessNet().to(device)
    net.load_state_dict(torch.load(model_path, map_location=device))
    net.eval()

    matrix_pos = fen2matrix(fen)
    matrix_pos = torch.tensor(matrix_pos).unsqueeze(0).to(device)
    
    with torch.no_grad():
        score = net(matrix_pos)
    return score.item()

if __name__ == '__main__':
    #fen = "3r2k1/3r1p2/3P2n1/p4QP1/8/2p5/P7/1K3R2 w - - 0 49"  # should be +3
    #fen = "r1b3k1/p3b1pp/2p1pr2/3p4/2P4P/3B4/q2BQPP1/2R2RK1 b - - 5 21" # should be +0.6
    fen = "1r4r1/R1p1bk2/4P3/1p1np2p/1P5N/2P2PP1/3BK2P/8 b - - 0 30"  # should be -5.9 
    model_path = "model/epoch_300.pth"
    score = evaluate_fen(fen, model_path)
    print(f"Evaluation Score: {score}")
    input()
