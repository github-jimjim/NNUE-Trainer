import chess
import chess.engine
import pandas as pd
from tqdm import tqdm

stockfish_9_path = "stockfish9.exe"
jomfish_10_path = "jomfish10.exe"
stockfish_17_path = "stockfish17.exe"

csv_file = "engine_battle_fens.csv"
target_fens = 500000
fen_save_interval = 100
max_moves = 200

# Engines starten
stockfish_9 = chess.engine.SimpleEngine.popen_uci(stockfish_9_path)
jomfish_10 = chess.engine.SimpleEngine.popen_uci(jomfish_10_path)
stockfish_17 = chess.engine.SimpleEngine.popen_uci(stockfish_17_path)


def play_game(fen_positions):
    board = chess.Board()
    move_count = 0

    while not board.is_game_over() and move_count < max_moves:
        engine = stockfish_9 if board.turn == chess.WHITE else jomfish_10

        result = engine.play(board, chess.engine.Limit(time=0.1))
        board.push(result.move)
        move_count += 1

        fen_positions.append(board.fen())

        if len(fen_positions) >= target_fens:
            return


def evaluate_positions(fen_positions):
    evaluated_positions = []

    for fen in tqdm(fen_positions, desc="Bewerte Positionen mit Stockfish 17"):
        board = chess.Board(fen)
        analysis = stockfish_17.analyse(board, chess.engine.Limit(time=0.1))
        score = analysis["score"].relative

        if score.is_mate():
            evaluation = 10000 if score.mate() > 0 else -10000
        else:
            evaluation = score.score()

        evaluated_positions.append((fen, evaluation))

    return evaluated_positions


def save_to_csv(evaluated_positions, mode='a', header=False):
    """
    Speichert bewertete FENs und Bewertungen in eine CSV-Datei.
    Schreibt Spaltenüberschriften nur, wenn der Header aktiviert ist.
    """
    df = pd.DataFrame(evaluated_positions, columns=["FEN", "Bewertung"])
    df.to_csv(csv_file, mode=mode, header=header, index=False)
    print(f"{len(evaluated_positions)} bewertete FENs gespeichert in {csv_file}.")


def main():
    fen_positions = []
    first_save = True

    try:
        while len(fen_positions) < target_fens:
            print(f"Starte neue Partie. Bisher gesammelte FENs: {len(fen_positions)}")
            play_game(fen_positions)

            evaluated_positions = evaluate_positions(fen_positions)

            save_to_csv(evaluated_positions, mode='a', header=first_save)
            first_save = False
            fen_positions.clear()

        print("Ziel von 500.000 bewerteten FENs erreicht.")
    finally:
        stockfish_9.quit()
        jomfish_10.quit()
        stockfish_17.quit()
        print("Alle Engines geschlossen.")


if __name__ == "__main__":
    main()
