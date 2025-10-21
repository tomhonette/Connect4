import math
import random
import numpy as np

ROW_COUNT = 6
COLUMN_COUNT = 7


class AIPlayer:

    def calculate_score(self, window, piece):
        score = 0
        opponent = 1
        if piece == 1:
            opponent = 2
        if window.count(piece) == 4:
            score += math.inf
        elif window.count(piece) == 3 and window.count(0) == 1:
            score += math.inf / 100000
        elif window.count(piece) == 2 and window.count(0) == 2:
            score += math.inf / 1000000000000

        if window.count(opponent) == 3 and window.count(0) == 1:
            score += -math.inf / 1000000

        return score

    def position(self, board, piece):

        score = 0

        # for the horizontal locations
        for r in range(ROW_COUNT):
            row_array = [int(i) for i in list(board[r, :])]
            # -3 so you dont start with the last 3 spaces from the left
            for c in range(COLUMN_COUNT - 3):
                window = row_array[c:c + 4]
                score += self.calculate_score(window, piece)

        # for the vertical locations
        for c in range(COLUMN_COUNT):
            column_array = [int(i) for i in list(board[:, c])]
            for r in range(ROW_COUNT - 3):
                window = column_array[r:r + 4]
                score += self.calculate_score(window, piece)

        # for the positive diagonal locations
        for r in range(ROW_COUNT - 3):
            for c in range(COLUMN_COUNT - 3):
                window = [board[r + i][c + i] for i in range(4)]
                score += self.calculate_score(window, piece)

        # for the negative diagonal locations
        for r in range(ROW_COUNT - 3):
            for c in range(COLUMN_COUNT - 3):
                window = [board[r + 3 - i][c + i] for i in range(4)]
                score += self.calculate_score(window, piece)

        return score

    def winning_move(self, board, piece):
        # Check horizontal locations for win
        for c in range(COLUMN_COUNT - 3):
            for r in range(ROW_COUNT):
                if board[r][c] == piece and board[r][c + 1] == piece and board[r][c + 2] == piece and board[r][
                    c + 3] == piece:
                    return True
        # Check vertical locations for win
        for c in range(COLUMN_COUNT):
            for r in range(ROW_COUNT - 3):
                if board[r][c] == piece and board[r + 1][c] == piece and board[r + 2][c] == piece and board[r + 3][
                    c] == piece:
                    return True

        # Check positively sloped diaganols
        for c in range(COLUMN_COUNT - 3):
            for r in range(ROW_COUNT - 3):
                if board[r][c] == piece and board[r + 1][c + 1] == piece and board[r + 2][c + 2] == piece and \
                        board[r + 3][c + 3] == piece:
                    return True

        # Check negatively sloped diaganols
        for c in range(COLUMN_COUNT - 3):
            for r in range(3, ROW_COUNT):
                if board[r][c] == piece and board[r - 1][c + 1] == piece and board[r - 2][c + 2] == piece and \
                        board[r - 3][c + 3] == piece:
                    return True

    def is_valid_location(self, board, col):
        return board[ROW_COUNT - 1][col] == 0

    def get_valid_locations(self, board):
        valid_locations = []
        for col in range(COLUMN_COUNT):
            if self.is_valid_location(board, col):
                valid_locations.append(col)
        return valid_locations

    def is_terminal_node(self, board):
        return self.winning_move(board, 1) or self.winning_move(board, 2) or len(self.get_valid_locations(board)) == 0

    def get_next_open_row(self, board, col):
        for r in range(ROW_COUNT):
            if board[r][col] == 0:
                return r

    def drop_piece(self, board, row, col, piece):
        board[row][col] = piece

    def minimax(self, board, depth, maximizingPlayer):
        valid_locations = self.get_valid_locations(board)
        terminal_node = self.is_terminal_node(board)
        if depth == 0 or terminal_node:
            if terminal_node:
                if self.winning_move(board, 2):
                    return (None, 10000000000000000)
                elif self.winning_move(board, 1):
                    return (None, -100000000000000)
                else:
                    return (None, 0)
            else:
                return (None, self.position(board, 2))
        if maximizingPlayer:
            value = -math.inf
            column = random.choice(valid_locations)
            for col in valid_locations:
                row = self.get_next_open_row(board, col)
                temporary_board = board.copy()
                self.drop_piece(temporary_board, row, col, 2)
                new_score = self.minimax(temporary_board, depth - 1, False)[1]
                if new_score > value:
                    value = new_score
                    column = col
            return column, value
        else:
            value = math.inf
            column = random.choice(valid_locations)
            for col in valid_locations:
                row = self.get_next_open_row(board, col)
                temporary_board = board.copy()
                self.drop_piece(temporary_board, row, col, 1)
                new_score = self.minimax(temporary_board, depth - 1, True)[1]
                if new_score < value:
                    value = new_score
                    column = col
            return column, value

    def pick_move(self, board):
        col, minimax_score = self.minimax(board, 4, True)
        return col