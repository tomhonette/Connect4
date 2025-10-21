import numpy
import random


FINAL_JITTER = 0.25
ROWS = 6
COLS = 7
ORDER = [3, 2, 4, 1, 5, 0, 6]
COL_WEIGHTS = [0, 1, 2, 3, 2, 1, 0]


#In the entire program, the complexity of each method are written before their declaration
#Here are what the parameters used to express the complexity mean:s
#R: rows
#C: columns
#d: depth of the minimax tree
#b: the medium number of legal moves per position
#m: the number of legal moves in this position (so m<=b)
#T_minimax(d): the cost of the minimax with a depth of d

#Here is the complexity of the entire algorithm:
#O(R·C + T_minimax(d)) = O(T_minimax(d))

def aiplayer1(board):
    #Verifying inputs in case of a wrong board provided
    #first checking if it's the right type of array
    if not isinstance(board, numpy.ndarray):
        raise TypeError("Board must be a numpy array")
    #then checking size, otherwise the algorithm could throw errors
    if board.shape != (ROWS, COLS):
        raise ValueError("The board must be a connect 4 standard size")
    #checking if the board only contains 0, 1 and 2 otherwise
    if not numpy.isin(board, [0, 1, 2]).all():
        raise ValueError("The board must only contain 0, 1 and 2")

    #checking which columns are available
    available_columns = valid_columns(board)
    #getting the current player
    me = get_current_player(board)
    #getting the opponent
    opponent = 3 - me
    #checking first if we can win in 1 move
    column = check_if_winner_soon(board, me, opponent)
    #if we can't we call by minimax method
    if column is None:
        #getting the best column to play via the minimax method
        #Note that we set the alpha to a really low value and the beta to a really high value so that we can then maximize and minimize these values later
        #Maximizing is set to true because it's the current AI's turn
        #the depth of the tree is set to 4 by default to plan on the next few moves (depth=4)
        #This value can be changed, but it is important to know that the deeper it is, the longer it will take for the algorithm to compute but the more "good" the AI will be at playing the game
        best_col, _ = minimax(board, 4, -10**9, 10**9, True, me, opponent)
        #if the best_col is not found, we get the first available and if we don't have any available we set it to 0 (game ended)
        if best_col is None:
            best_col = available_columns[0] if available_columns else 0
        column = best_col
    return column

#Complexity: O(b^(d+1).R.C)
def minimax(board, depth, alpha, beta, maximizing, me, opponent):
    """
    Main recursive logic of the minimax algorithm
    :param board: the current board we're playing on
    :param depth: the depth of the tree of the minimax
    :param alpha: best value found on the maximum
    :param beta: best value found on the minimum
    :param maximizing: True if it's the turn of the current AI playing, False otherwise
    :param me: the current AI playing next turn
    :param opponent: the opponent
    :return: None if the game is in a terminal state or best_column if it's found + the heuristic score of the AI
    """
    #checking if the depth is 0 or the game is done
    if depth == 0 or is_terminal(board):
        #if the current AI won we return None and a huge value (it won)
        if has_winner(board, me):
            return None, 10**8
        #if the opponent won, we return None and a tiny value (it lost)
        if has_winner(board, opponent):
            return None, -10**8
        #otherwise we return None, and we evaluate the heuristic value of the position on the board
        return None, evaluate(board, me, opponent)
    #possible moves for the AI
    moves = valid_columns(board)
    #if no valid moves, we return None, and we return the heuristic value of the position on the board (Tie)
    if not moves:
        return None, evaluate(board, me, opponent)

    if maximizing:
        # custom ordering using Jitter's effect
        scored = jitter_ordering(moves, board, me, me, opponent)
        #Sorting Tuples from best to worst
        scored.sort(reverse=True)
        #we extract the column only from the ordered Tuples
        moves = [col for _, __, col in scored]

        #initialazing the best score to a tiny value
        best_score = -10**9
        #initializing the best column by default being the first legal column
        best_col = moves[0]
        #This array will contain pairs of columns to check later
        scored_moves = []
        # looping in legal columns
        for col in moves:
            #simulating a move played by the current AI
            child = simulate(board, col, me)
            #exploring deeper by recursion in the tree to simulate the game for all branches, keeping track of the score
            #Note that we set maximizing to false because it's the opponent's turn after
            _, s = minimax(child, depth - 1, alpha, beta, False, me, opponent)
            #we add the column and score to the scored_moves
            scored_moves.append((col, s))
            #if we found the best score possible for the next move, we set best_score and best_col to the best values found
            if s > best_score:
                best_score, best_col = s, col
            #we set the alpha score to the biggest between the current alpha and best_score
            alpha = max(alpha, best_score)
            #if alpha bigger than beta, it means that the parent node will never "change its mind" regarding what the children's score is, so we exit that subtree to gain time (cutoff)
            if alpha >= beta:
                break

        #This part of the code helps the AI to be less predictable and adds a little bit more bias in the final output of its moves
        #creating our numeric tolerance value
        eps = 1e-6
        #this array will contain candidates between different options available for the AI
        candidates = []
        #we loop between all couples of columns
        for col_i, score_i in scored_moves:
            #we check if they have a really close score. If the difference between their score is smaller than the tolerance, we add them to the candidates
            if abs(score_i - best_score) <= eps:
                candidates.append(col_i)

        #if we have more than 1 candidate and the best_score is not too big
        if len(candidates) > 1 and abs(best_score) < 10 ** 7:
            #we create weights
            weights = []
            #we loop in all candidates
            for col_i in candidates:
                #we add the weights of all columns that are candidates
                weights.append(COL_WEIGHTS[col_i] + 1)
            #then we choose which column to use from the candidates based on the weights we created
            best_col = random.choices(candidates, weights=weights, k=1)[0]
        #returning the best_column to play on and the best_score possible
        return best_col, best_score
    #Minimizing part (opponent)
    else:
        #custom ordering using Jitter's effect
        scored = jitter_ordering(moves, board, opponent, me, opponent)
        #We sort from worse to best Tuples
        scored.sort()
        #we get the columns out of the Tuples
        moves = [col for _, __, col in scored]
        #setting the best score by default to a huge value
        best_score = 10**9
        #the rest is the exact same principle than for the maximizing part, with some twists
        best_col = moves[0]
        scored_moves = []
        for col in moves:
            child = simulate(board, col, opponent)
            _, s = minimax(child, depth - 1, alpha, beta, True, me, opponent)
            scored_moves.append((col, s))
            #checking if the score is lower than the best_score instead of bigger
            if s < best_score:
                best_score, best_col = s, col
            #setting the beta score to the minimum between the current beta and the best_score (opposite that what we've done with alpha score)
            beta = min(beta, best_score)
            #same as for the maximizing part here
            if alpha >= beta:
                break

        #here it's same as maximizing part
        eps = 1e-6
        candidates = []
        for col_i, score_i in scored_moves:
            if abs(score_i - best_score) <= eps:
                candidates.append(col_i)

        if len(candidates) > 1 and abs(best_score) < 10 ** 7:
            weights = []
            for col_i in candidates:
                weights.append(COL_WEIGHTS[col_i] + 1)
            best_col = random.choices(candidates, weights=weights, k=1)[0]
        return best_col, best_score


#Complexity: O(R.C)
def simulate(board, col, piece):
    """
    simulates a turn without affecting the current board
    :param board: the current board we're playing on
    :param col: the column to play on
    :param piece: the piece we put (1 or 2)
    :return: the copied board we added the piece in
    """
    #first copying the board to not affect anything in the main board
    b2 = board.copy()
    #we loop in the column to check which row we have to put the coin (gravity)
    for row in range(ROWS):
        if b2[row][col] == 0:
            b2[row][col] = piece
            break
    return b2

#Complexity: O(R.C)
def evaluate(board, me, opponent):
    """
    Calculates the heuristic score of the current position
    :param board: the current board we're playing on
    :param me: the AI playing next turn
    :param opponent: the opponent
    :return: the heuristic score
    """
    score = 0
    bonus = 0
    # loop in every column
    for col in range(COLS):
        # counts how many pieces the current AI has in that column
        count_in_col = int(numpy.count_nonzero(board[:, col] == me))
        # binding to avoid giving too many points for one column
        bindings = min(count_in_col, 3)
        # getting the weights defined for that column
        weight = COL_WEIGHTS[col]
        # calculating the bonus based on the weight and bindings
        bonus += weight * bindings

    # we add the bonus calculated in the score
    score += bonus

    # we get all windows
    for vals, coords in get_windows(board):
        #checking if the current AI has 3 coins + 1 empty case in a single window/line
        if vals.count(me) == 3 and vals.count(0) == 1:
            #we get the index of the 0 in the window
            k = vals.index(0)
            #we get the coordinates of the 0 on the board
            row, col = coords[k]
            #we check if the move is playable by the AI, if so we add 50 points to the score (good position)
            if playable(board, row, col):
                score += 50

        #same as above for the opponent but we remove 40 points to the AI score (bad position)
        if vals.count(opponent) == 3 and vals.count(0) == 1:
            k = vals.index(0)
            row, col = coords[k]
            if playable(board, row, col):
                score -= 40

        #We check if we have 2 coins + two 0 in one window, if so we add 5 to the score
        if vals.count(me) == 2 and vals.count(0) == 2:
            score += 5

        #same for opponent and we remove 5 if so
        if vals.count(opponent) == 2 and vals.count(0) == 2:
            score -= 5

    return score

#Complexity: O(m.R.C)
def jitter_ordering(moves, board, piece_to_play, me, opponent):
    """
    The jitter ordering here is made to add a little bias to the AI's plays when the AI has to decide
    between different plays that have a close score.
    :param moves: the legal moves playable
    :param board: the current board we're playing on
    :param piece_to_play: the piece to simulate in each move (me in MAX, opponent in MIN)
    :param me: the current AI playing next turn
    :param opponent: the opponent
    :return: a list of tuples (order_score, column_weight, column)
    """
    scored = []
    for col in moves:
        # simulate the right side to play at this node
        s0 = evaluate(simulate(board, col, piece_to_play), me, opponent)
        # tiny randomness only for ordering (does not change minimax values)
        s_ord = s0 + (random.uniform(-FINAL_JITTER, FINAL_JITTER) if FINAL_JITTER else 0.0)
        #we append the score, the weight of the column and the column to the tuple
        scored.append((s_ord, COL_WEIGHTS[col], col))
    return scored

#Complexity: O(R.C)
def check_if_winner_soon(board, me, opponent):
    """
    method to check if there is a win in one move or if there is no winning move we check if we need to block the opponent from winning in one move next turn
    :param board: the current board we're playing on
    :param me: the AI playing this turn
    :param opponent: the opponent
    :return: None if there is no win/block in next move or the column to play/block the next winning move
    """
    #we check first if we can win in one move by splitting the board in arrays of 4 (windows/lines)
    for values, coo in get_windows(board):
        #counting the number of coins the current AI has
        number_of_1 = values.count(me)
        #counting the number of 0
        zero = values.count(0)
        #if we potentially have a winning move
        if number_of_1 == 3 and zero == 1:
            #we check where the 0 is in the line
            k = values.index(0)
            #we get the coordinates of the empty case
            row, col = coo[k]
            #we check if we can legally play that move, then we return the column to play
            if playable(board, row, col):
                return col
    #if no winning move we check if we can block the opponent from winning (we loop in get_windows as above)
    for values, coo in get_windows(board):
        #counting the number of coins the opponent has
        number_of_2 = values.count(opponent)
        zero = values.count(0)
        #checking if there is a potentially have a winning move for the opponent
        if number_of_2 == 3 and zero == 1:
            k = values.index(0)
            row, col = coo[k]
            #if the move is playable to block the opponent we return the column
            if playable(board, row, col):
                return col
    return None


#--------------
#Utils methods
#--------------

#Complexity: O(1)
def check_if_column_full(board, col):
    """
    method to check if the column is full to avoid errors or pointless moves by AI
    :param board: the current board we're playing on
    :param col: the column to check
    :return: true if column is full, false otherwise
    """
    #
    return board[ROWS - 1][col] != 0



#Complexity: O(1)
def playable(board, row, col):
    """
    checking if a move is legally playable
    :param board: current board we're playing on
    :param row: the row we want to check
    :param col: the column we want to check
    :return: true if a move is legally playable, false otherwise
    """
    #first checking if the case on the coordinates on the board is actually empty
    if board[row][col] != 0:
        return False
    #checking if we're not on the floor of the board
    if row == 0:
        return True
    #finally checking if there is a coin under the place we want to put our coin (gravity check)
    if board[row - 1][col] != 0:
        return True
    else:
        return False

#Complexity: O(R.C)
def get_current_player(board):
    """
    getting the current player playing next turn
    :param board: the current board we're playing on
    :return: the current player playing next turn
    """
    #we get how many of 1 and 2 are on the board
    count1 = numpy.count_nonzero(board == 1)
    count2 = numpy.count_nonzero(board == 2)
    #if the numbers are the same then it's player one's turn
    if count1 == count2:
        return 1
    #if there is more 1 than 2 then it's player two's turn
    elif count1 == count2 + 1:
        return 2
    else:
        #handling error in case of a bug
        raise ValueError("error")

#Complexity: O(R.C)
def get_windows(board):
    """
    getting all windows (or lines) of 4 coins
    :param board: the current board we're playing on
    :return: yields all horizontal, vertical and diagonal windows/lines of 4 coins
    """
    #first checking for rows
    for row in range(ROWS):
        #-3 because we are checking for 4 coins in a row
        for col in range(COLS - 3):
            #first calculating the values (0, 1, 2) in each case of the window
            vals = [board[row][col + i] for i in range(4)]
            #then getting coordinates in the same order for these cases
            coords = [(row, col + i) for i in range(4)]
            yield vals, coords
    #same principle here but for columns
    for col in range(COLS):
        for row in range(ROWS - 3):
            vals = [board[row + i][col] for i in range(4)]
            coords = [(row + i, col) for i in range(4)]
            yield vals, coords
    #here we're doing first for diagonal from bottom left to top right
    #the only twist here compared to vertical/horizontal is that we have to bound the area to not have errors by calculation windows of 1, 2 or 3 cases (we want windows of 4 cases)
    #this is why we bound the rows/columns with an offset of 3
    for row in range(ROWS - 3):
        for col in range(COLS - 3):
            vals = [board[row + i][col + i] for i in range(4)]
            coords = [(row + i, col + i) for i in range(4)]
            yield vals, coords
    #here same but going from bottom right to top left
    for row in range(3, ROWS):
        for col in range(COLS - 3):
            vals = [board[row - i][col + i] for i in range(4)]
            coords = [(row - i, col + i) for i in range(4)]
            yield vals, coords

#Complexity: O(C)
def valid_columns(board):
    """
    calculates valid columns the AI can play on based on an order
    :param board: the current board we're playing on
    :return: valid columns ordered by priority (center → edges)
    """
    cols = []
    #looping based on the order
    for col in ORDER:
        #first checking if column is not full, then if not we add it to the list of columns the AI can play on
        if not check_if_column_full(board, col):
            cols.append(col)
    return cols

#Complexity: O(R.C)
def has_winner(board, piece):
    """
    checking if a player has won
    :param board: the current board we're playing on
    :param piece: the player we want to check
    :return: True if the player has won, False otherwise
    """
    #we get all the windows
    for vals, _ in get_windows(board):
        #we check if there are 4 pieces of the same type in it
        if vals.count(piece) == 4:
            return True
    return False

#Complexity: O(R.C)
def is_terminal(board):
    """
    This method checks for the minimax if the current position is a terminal position
    :param board: the current board we're playing on
    :return: True if the current position is terminal or there are no valid columns to play on, False otherwise
    """
    #checking first if a player has won
    if has_winner(board, 1) or has_winner(board, 2):
        return True
    #if not checking if there are valid columns the AI can play on
    return len(valid_columns(board)) == 0
