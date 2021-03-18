from envs.tapLogicEnv.tapLogicEnv import tapLogicEnv


class RepresentationManager():
    def __init__(self, args):
        self.args = args

        level_idx = args.level_idx
        if args.train_multiple_levels:
            self.itemsInState = [3, 4, 5, 6, 7, 8, 9, 10, -10, 11, 12, 13, 14, 15, 16, 17]
            self.statusInState = []
            self.itemsInGoal = []
            self.goalRatio = 2
            self.actionIdxs = [0, 7, 8, 9]
            for i in range(args.multiple_level_start, args.multiple_level_end):
                self.representation_auto_generator(i)
        elif 1 <= level_idx <= 3000:
            self.itemsInState = [3, 4, 5, 6, 7, 8, 9, 10, -10, 11, 12, 13, 14, 15, 16, 17]
            self.statusInState = []
            self.itemsInGoal = []
            self.goalRatio = 2
            self.actionIdxs = [0, 7, 8, 9]
            self.representation_auto_generator(level_idx)
        else:
            raise NotImplementedError()

        self.level_idx = level_idx
        self.max_episode_length = args.max_episode_length

    def representation_auto_generator(self, level_idx):
        env = tapLogicEnv("envs/tapLogicEnv/levels/" + self.args.level_version + "/" + str(level_idx) + ".txt",
                          extra_info = {"max_episode_length": 400})

        viewBoard = env.reset()
        goalsDict = env.viewParser.goals_dict

        for goal in goalsDict:
            if goal not in self.itemsInGoal:
                self.itemsInGoal.append(goal)

                if goal == 24:
                    # Bubble
                    self.statusInState.append(1)
                    self.actionIdxs.append(12)
                elif goal == 18 or goal == 25:
                    # Duck or giant duck
                    self.itemsInState.append(goal)
                    self.actionIdxs.append(11)
                    self.actionIdxs.append(13)
                elif goal == 19:
                    # Balloon
                    self.itemsInState.append(goal)
                    self.actionIdxs.append(10)
                elif 12 <= goal <= 17:
                    # Color
                    self.actionIdxs.append(goal - 11)
                elif goal == 22:
                    # Crate
                    self.itemsInState.append(22)
                    self.actionIdxs.append(14)
                    self.actionIdxs.append(15)
                elif goal == 28:
                    # Magic hat and carrot
                    self.itemsInState.append(27)
                    self.actionIdxs.append(16)
                elif goal == 2:
                    # Colored balloon
                    self.itemsInState.append(2)
                    self.actionIdxs.append(17)
                    self.actionIdxs.append(18)
                elif goal == 21:
                    # Lightblub
                    self.itemsInState.append(21)
                    self.statusInState.append(2)
                    self.actionIdxs.append(19)
                    self.actionIdxs.append(20)
                    self.actionIdxs.append(21)
                elif goal == 29:
                    # Jelly
                    self.itemsInState.append(29)
                    self.actionIdxs.append(22)
                    self.actionIdxs.append(23)
                elif goal == 23:
                    # Colored crate
                    self.itemsInState.append(23)
                    self.actionIdxs.append(24)
                    self.actionIdxs.append(25)
                elif goal == 20:
                    # Pinata
                    self.itemsInState.append(20)

                elif goal == 22:
                    # Crate
                    self.itemsInState.append(22)
                    self.statusInState.append(2)
                    self.actionIdxs.append(26)
                    self.actionIdxs.append(27)
                    self.actionIdxs.append(28)
                elif goal == 30:
                    # CanToss
                    self.itemsInState.append(30)
                    self.actionIdxs.append(29)
                elif goal == 26:
                    # Doughnut
                    self.itemsInState.append(26)
                    self.actionIdxs.append(30)
                    self.actionIdxs.append(31)
                    self.actionIdxs.append(32)
                elif goal == 32:
                    # Giant pinata
                    self.itemsInState.append(32)
                elif goal == 33:
                    # Clown
                    self.itemsInState.append(33)
                    self.statusInState.append(2)
                    self.actionIdxs.append(33)
                    self.actionIdxs.append(34)
                    self.actionIdxs.append(35)
                elif goal == 34:
                    # FourColorBottle
                    self.itemsInState.append(34)
                    self.statusInState.append(2)
                    self.statusInState.append(3)
                    self.statusInState.append(4)
                    self.actionIdxs.append(36)
                    self.actionIdxs.append(37)
                    self.actionIdxs.append(38)
                elif goal == 35:
                    # Egg
                    self.itemsInState.append(35)
                    self.statusInState.append(2)
                    self.statusInState.append(3)
                elif goal == 36:
                    # Iron box
                    self.itemsInState.append(36)
                elif goal == 43:
                    # Changing color item
                    self.itemsInState.append(43)
                elif goal == 31:
                    # Medal case
                    self.itemsInState.append(31)
                elif goal == 46:
                    # Rabbit
                    self.itemsInState.append(45)
                    self.statusInState.append(1)
                elif goal == -1:
                    pass
                else:
                    raise NotImplementedError()

        for x in range(viewBoard.shape[0]):
            for y in range(viewBoard.shape[1]):
                if viewBoard[x, y] == 0:
                    continue
                if viewBoard[x, y] not in self.itemsInState:
                    self.itemsInState.append(viewBoard[x, y])

        all_items = env.GetCurLevelAllItemType().split("，")
        all_items = [int(item) for item in all_items]
        for item in all_items:
            if not item in self.itemsInState:
                self.itemsInState.append(item)

        # Boss level
        # if env.CheckWhetherBossLevel():
        #     self.itemsInGoal.append(-1)

        return

    def get_extra_info_dict(self):
        print("level", self.level_idx)
        print("itemsInState", self.itemsInState)
        print("statusInState", self.statusInState)
        print("itemsInGoal", self.itemsInGoal)
        print("actionIdxs", self.actionIdxs)

        extra_info = dict()
        extra_info["itemsInState"] = self.itemsInState
        extra_info["statusInState"] = self.statusInState
        extra_info["itemsInGoal"] = self.itemsInGoal
        extra_info["goalRatio"] = self.goalRatio
        extra_info["actionIdxs"] = self.actionIdxs
        extra_info["max_episode_length"] = self.max_episode_length

        return extra_info
