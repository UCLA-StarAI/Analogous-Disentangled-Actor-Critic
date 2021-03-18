import struct
import numpy as np
import os


class LevelFileExporter():
    def __init__(self, level_idx, max_episode_length = 1000):
        self.buffer = b""

        self.level_idx = level_idx
        self.total_move = 0
        self.height = 9
        self.width = 9

        self.cells_type = np.zeros([max_episode_length, self.height, self.width], dtype = np.int16)
        self.cells_layer = np.zeros([max_episode_length, self.height, self.width], dtype = np.int16)
        self.cells_color = np.zeros([max_episode_length, self.height, self.width], dtype = np.int16)
        self.cells_info = np.zeros([max_episode_length, self.height, self.width], dtype = np.int16)

        self.cells_mcts_q = np.zeros([max_episode_length, self.height, self.width], dtype = np.float32)
        self.cells_mcts_n = np.zeros([max_episode_length, self.height, self.width], dtype = np.int16)

        self.goals = []

        self.actions = np.zeros([max_episode_length, 2], dtype = np.int16)

        self.curr_step_count = 0

    def reset_record(self, tapLogicViewParser):
        self.curr_step_count = 0

        self.total_move = tapLogicViewParser.moveLeft

        self.height = tapLogicViewParser.boardSize[0]
        self.width = tapLogicViewParser.boardSize[1]

        self.cells_type *= 0
        self.cells_layer *= 0
        self.cells_color *= 0
        self.cells_info *= 0
        self.cells_mcts_q *= 0.0
        self.cells_mcts_n *= 0

        for h in range(self.height):
            for w in range(self.width):
                self.cells_type[0, self.height - h - 1, w] = tapLogicViewParser.viewBoard[h, w]
                self.cells_layer[0, self.height - h - 1, w] = tapLogicViewParser.itemCountBoard[h, w]
                self.cells_color[0, self.height - h - 1, w] = tapLogicViewParser.itemColorBoard[h, w]
                self.cells_info[0, self.height - h - 1, w] = tapLogicViewParser.itemInfoBoard[h, w]

        gdict = []
        for itemType in tapLogicViewParser.goals_dict:
            gdict.append(itemType)
            gdict.append(tapLogicViewParser.goals_dict[itemType])
        self.goals.append(gdict)

    def record_next(self, tapLogicViewParser, action, mcts_result = None):
        self.curr_step_count += 1

        if not isinstance(action, list):
            action = [action // self.width, action % self.width]

        for h in range(self.height):
            for w in range(self.width):
                self.cells_type[self.curr_step_count, self.height - h - 1, w] = tapLogicViewParser.viewBoard[h, w]
                self.cells_layer[self.curr_step_count, self.height - h - 1, w] = tapLogicViewParser.itemCountBoard[h, w]
                self.cells_color[self.curr_step_count, self.height - h - 1, w] = tapLogicViewParser.itemColorBoard[h, w]
                self.cells_info[self.curr_step_count, self.height - h - 1, w] = tapLogicViewParser.itemInfoBoard[h, w]

        if mcts_result is not None:
            mcts_actions, mcts_utilities, mcts_counts = mcts_result

            for mcts_action, mcts_utility, mcts_count in zip(mcts_actions, mcts_utilities, mcts_counts):
                h = mcts_action // self.width
                w = mcts_action % self.width

                self.cells_mcts_q[self.curr_step_count - 1, self.height - h - 1, w] = mcts_utility
                self.cells_mcts_n[self.curr_step_count - 1, self.height - h - 1, w] = mcts_count

        self.actions[self.curr_step_count - 1, 0] = action[0]
        self.actions[self.curr_step_count - 1, 1] = action[1]

        gdict = []
        for itemType in tapLogicViewParser.goals_dict:
            gdict.append(itemType)
            gdict.append(tapLogicViewParser.goals_dict[itemType])
        self.goals.append(gdict)

    def store_file(self, extra_name = ""):
        buffer = b""

        buffer += struct.pack("<i", self.level_idx)
        buffer += struct.pack("<i", self.total_move)
        buffer += struct.pack("<i", len(self.goals[0]) // 2)

        for i in range(len(self.goals[0]) // 2):
            buffer += struct.pack("<i", self.goals[0][i * 2])
            buffer += struct.pack("<i", self.goals[0][i * 2 + 1])

        buffer += struct.pack("<i", self.width)
        buffer += struct.pack("<i", self.height)
        for move in range(self.curr_step_count + 1):
            for h in range(self.height):
                for w in range(self.width):
                    buffer += struct.pack("<i", self.cells_type[move, h, w])
                    buffer += struct.pack("<i", self.cells_layer[move, h, w])
                    buffer += struct.pack("<i", self.cells_color[move, h, w])
                    buffer += struct.pack("<i", self.cells_info[move, h, w])

                    buffer += struct.pack("<f", self.cells_mcts_q[move, h, w])
                    buffer += struct.pack("<i", self.cells_mcts_n[move, h, w])

            buffer += struct.pack("<i", self.actions[move, 0])
            buffer += struct.pack("<i", self.actions[move, 1])

            for i in range(len(self.goals[0]) // 2):
                buffer += struct.pack("<i", self.goals[move][i * 2 + 1])
                '''if move == 0:
                    buffer += struct.pack("<i", 0)
                else:
                    buffer += struct.pack("<i", self.goals[move][i * 2 + 1])'''

        if not os.path.exists('./logs'):
            os.mkdir("./logs")

        if extra_name != "":
            extra_name = "_" + extra_name

        with open('logs/level_log' + str(self.level_idx) + extra_name + '.bin', 'wb') as f:
            f.write(buffer)

        return

    def store_test_file(self):
        level_idx = 20
        height = 9
        width = 9
        moves = 32
        cells_type = np.random.randint(12, 18, [moves, height, width])
        cells_layer = np.zeros([moves, height, width], dtype = np.int16)
        cells_color = np.zeros([moves, height, width], dtype = np.int16)

        actions = np.random.randint(0, 8, [moves, 2], dtype = np.int16)

        buffer = b""

        buffer += struct.pack("<i", level_idx)
        buffer += struct.pack("<i", height)
        buffer += struct.pack("<i", width)
        for move in range(moves):
            for h in range(height):
                for w in range(width):
                    buffer += struct.pack("<i", cells_type[move, h, w])
                    buffer += struct.pack("<i", cells_layer[move, h, w])
                    buffer += struct.pack("<i", cells_color[move, h, w])
            buffer += struct.pack("<i", actions[move, 0])
            buffer += struct.pack("<i", actions[move, 1])

        with open('logs/level_log' + str(self.level_idx) + '.bin', 'wb') as f:
            f.write(buffer)


if __name__ == "__main__":
    levelFileExporter = LevelFileExporter(10)

    levelFileExporter.store_test_file()
