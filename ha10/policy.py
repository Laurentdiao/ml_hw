# -*- coding: utf-8 -*-
import numpy as np
import random
import pickle as pkl
import os.path

class Policy:
    __doc__ = r"""
    Base Policy class, define the interface.

    * :attr:`marker` keeps the role of Policy, can either be 1 (for the first Policy),
      or -1 (for the second).
    * :attr:`board_size` keeps the size of the board.
    
    Args:
        marker (int): either 1 or -1, indicating first Policy or second Policy respectively.
        board_size (int, int): the size of the board.
    """

    def __init__(self, marker, board_size=(3, 3)):
        assert marker in [1, -1], "A Policy's marker must be 1 (first) or -1 (second)"
        self.marker = marker
        self.board_size = board_size

    def decide(self, state):
        """
        Make move according to the state. It will called by game to get an action.

        Args:
            state (ndarray): a 2D numpy array, the game board, filled with 1, -1, 0,
                indicating marker of first Policy, second Policy, and place not yet occupied.

        Returns:
            action (tuple of (int, int)): indicates the coordinates to place marker,
                starting from 0, i.e. (0, 0) is the top left corner.
        """
        raise NotImplementedError

    def get_available_actions(self, state):
        """
        Make move according to the state. It will called by game to get an action.

        Args:
            state (ndarray): a 2D numpy array, the game board, filled with 1, -1, 0,
                indicating marker of first Policy, second Policy, and place not yet occupied.

        Returns:
            action ([tuple of (int, int)]): a list of available coordinates to place marker.
        """
        assert state.shape == self.board_size, "Board size mismatch!"
        empty_x, empty_y = np.where(state == 0) #array of x, and y where the cell at (x_i, y_i) is empty.
        available_actions = zip(empty_x, empty_y)
        available_actions = list(available_actions)
        return available_actions


class RandomPolicy(Policy):
    __doc__ = r"""
    Example random Policy, which just randomly chooses a position to place marker.

    * :attr:`marker` keeps the role of Policy, can either be 1 (for the first Policy),
      or -1 (for the second).
    * :attr:`board_size` keeps the size of the board.
    
    Args:
        marker (int): either 1 or -1, indicating first Policy or second Policy respectively.
        board_size (int, int): the size of the board.
    """
    def __init__(self, marker, board_size=(3, 3)):
        super(RandomPolicy, self).__init__(marker, board_size)

    def decide(self, state):
        """
        Make a random move according to the state. It will called by game to get an action.

        Args:
            state (ndarray): a 2D numpy array, the game board, filled with 1, -1, 0,
                indicating marker of first Policy, second Policy, and place not yet occupied.

        Returns:
            action (tuple of (int, int)): indicates the coordinates to place marker,
                starting from 0, i.e. (0, 0) is the top left corner.
        """
        available_actions = self.get_available_actions(state)
        assert len(available_actions) > 0, "No empty place!"
        idx = np.random.randint(len(available_actions))
        return available_actions[idx]

class QTable:
    __doc__ = r"""
    A Q table class, to simplify looking up and setting value of a q table. 

    * :attr:`default_val` keeps the default value.
    * :attr:`_encode_state` helper function to convert state to an unique code of state.

    * :attr:`table` keeps the actual table.
    
    Args:
        default_val (float): the default value for an unseen pair of state and action.
    """
    def __init__(self, default_val):
        #using a two layer dictionary with with a default value as the q table
        self.table = {}
        self._encode_state = lambda x: x.astype("b").tobytes()
        self.default_val = default_val

    def __getitem__(self, key):
        """
        Look up Q(state, action).
        Usage: q_table[state, action] gives the value of Q(state, action).

        Args:
            key (tuple of state and action): packed state and action, where state should be a ndarray.
        """
        assert len(key) == 2, "It's not a valid key!"
        state, action = key
        state_code = self._encode_state(state)
        if state_code not in self.table:
            self.table[state_code] = {}
        if action not in self.table[state_code]:
            self.table[state_code][action] = self.default_val
        return self.table[state_code][action]

    def __setitem__(self, key, val):
        """
        Set Q(state, action) to val.
        Usage: q_table[state, action] = val sets the value of Q(state, action) to val.

        Args:
            key (tuple of state and action): packed state and action, where state should be a ndarray.
            val (value): the final value of Q(state, action)
        """
        assert len(key) == 2, "It's not a valid key!"
        state, action = key
        state_code = self._encode_state(state)
        self.table[state_code][action] = val

    def load(self, name):
        """
        Load saved table.

        Args:
            name (str): name of the pickle file saves the table,
        """
        with open(name, "rb") as f:
            self.table = pkl.load(f)

    def save(self, name=None):
        """
        Save table to pickle file.

        Args:
            name (str): name of the pickle file saves the table,
        """
        with open(name, "wb") as f:
            pkl.dump(self.table, f)

class QLearningPolicy(Policy):
    __doc__ = r"""
    A Q-learning Policy class, YOU NEED TO FINISH THIS TO PERFORM Q-LEARNING.

    * :attr:`marker` keeps the role of Policy, can either be 1 (for the first Policy),
      or -1 (for the second).

    * :attr:`board_size` keeps the size of the board.

    * :attr:`q_table` a instance of the QTable class.
        q_table[state, action] gives the value of Q(state, action),
        and it supports assignment.
        Here, the state is the ndarray of the game board, and action is tuple of int
        indicating the coordinates of the cell to place the marker.
    
    * :attr:`mode` either "train" or "eval", indicates the mode of the agent.
        In train mode, apply the epsilon-greedy strategy.
        In eval mode, exploit the learned q_table.
    
    * :attr:`...` You can add additional attributes for this class.
    
    Args:
        marker (int): either 1 or -1, indicating first Policy or second Policy respectively.
        board_size (int, int): the size of the board.

        ... (any, optional): You can add additional optional arguments,
            but ensure it works in main\*.py without changing them
    """

    def __init__(self, marker, board_size=(3, 3), learning_rate=0.05, discount_factor=0.9):
        super(QLearningPolicy, self).__init__(marker, board_size)

        initial_val = 0 #The default value for every unseen state, action pair.
        ########################################## YOUR CODE HERE ##########################################
        # TODO
        # Set hyperparameters
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        epsilon=0.2
        epsilon_min = 0.001
        self.decay_rate = 0.9999
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        ########################################## END OF YOUR CODE ########################################
        
        self.q_table = QTable(initial_val)
        self.mode = "eval"

        #load if any save exists
        name = "q_table_player" + ("1" if marker == 1 else "2")
        if os.path.exists(name):
            self.load()

    def load(self, name=None):
        """
        load saved table.

        Args:
            name (str, optional): name of the pickle file saves the table,
                default to be q_table_player1 (marker == 1) or q_table_player2 (marker == -1)
        """
        if not name:
            name = "q_table_player" + ("1" if self.marker == 1 else "2")
        self.q_table.load(name)


    def save(self, name=None):
        """
        save table to pickle file.

        Args:
            name (str, optional): name of the pickle file saves the table,
                default to be q_table_player1 (marker == 1) or q_table_player2 (marker == -1)
        """
        if not name:
            name = "q_table_player" + ("1" if self.marker == 1 else "2")
        self.q_table.save(name)

    def set_mode(self, mode):
        assert mode in ["train", "eval"]
        self.mode = mode

    def decide(self, state):
        """
        decide action according to the state and current mode.
        YOU NEED TO FINISH IT

        Args:
            state (ndarray): a 2D numpy array, the game board, filled with 1, -1, 0,
                indicating marker of first Policy, second Policy, and place not yet occupied.

        Returns:
            action (tuple of (int, int)): indicates the coordinates to place marker,
                starting from 0, i.e. (0, 0) is the top left corner.        
        """
        available_actions = self.get_available_actions(state)
        move = available_actions[0]
        ########################################## YOUR CODE HERE ##########################################
        # TODO
        # Decide what action to take given current state and mode using the q table.
        # Note: You can directly loop up the value of Q(state, action) with self.q_table[state, action].
        if self.mode == "train":
            if np.random.rand() < self.epsilon:
                move = random.choice(available_actions)
            else:  
                q_values = [self.q_table[state, action] for action in available_actions]
                max_q = max(q_values)
                best_actions = [action for action, q in zip(available_actions, q_values) if q == max_q]
                move = random.choice(best_actions)
                self.epsilon = max(self.epsilon_min, self.epsilon * self.decay_rate)
        else:
            q_values = [self.q_table[state, action] for action in available_actions]
            max_q = max(q_values)
            best_actions = [action for action, q in zip(available_actions, q_values) if q == max_q]
            move = random.choice(best_actions)
        ########################################## END OF YOUR CODE ########################################
        return move

    def update_q_table(self, state, action, reward, next_state):
        """
        Update the q_table give a transaction tuple of (state, action, reward, next_state).
        YOU NEED TO FINISH IT

        Args:
            state (ndarray): a 2D numpy array, the game board before the action was taken, filled with 1, -1, 0,
                indicating marker of first Policy, second Policy, and place not yet occupied.

            action (tuple of (int, int)): indicates the coordinates to place marker,
                starting from 0, i.e. (0, 0) is the top left corner.        

            reward (float): 1 means it wins, 0 means a draw, and -1 means a lose.

            next_state (ndarray or NoneType): a 2D numpy array means, the game board after the action and opponent's
                response, filled with 1, -1, 0, indicating marker of first Policy, second Policy,
                and place not yet occupied, and a None means the state is a terminal state.
        """
        assert self.mode == "train", "Q table should not be updated during evaluation!"
        ########################################## YOUR CODE HERE ##########################################
        # TODO
        # Update the table given a transaction tuple.
        # Note: You can directly set the value Q(state, action) to val with self.q_table[state, action] = val.
        current_qvalues = self.q_table[state, action]
        if next_state is None:
            self.q_table[state, action] = current_qvalues + self.learning_rate * (reward - current_qvalues)
        else: 
            available_actions = self.get_available_actions(next_state)
            next_qvalues = [self.q_table[next_state, next_action] for next_action in available_actions]
            max_qvalues = max(next_qvalues)
            self.q_table[state, action] = current_qvalues + self.learning_rate * (reward + self.discount_factor * max_qvalues - current_qvalues)
        ########################################## END OF YOUR CODE ########################################
