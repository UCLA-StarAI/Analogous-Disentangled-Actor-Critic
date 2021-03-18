import pandas as pd
import numpy as np


def collect_train_data(chosen_AIs, chosen_levels, features, y_feature):
    filenames = ['TapLogicData.xlsx', 'PassrateData.xlsx', 'PlayerData.xlsx', 'PlayerItemDetailsData.xls']

    return collect_data(chosen_AIs, chosen_levels, features, y_feature, filenames)


def collect_test_data(chosen_AIs, chosen_levels, features):
    filenames = ['TapLogicData2.xlsx']

    return collect_data2(chosen_AIs, chosen_levels, features, filenames)


def collect_user_classified_data(chosen_AIs, chosen_levels, features, y_feature, date = "7_4"):
    filenames = ['TapLogicData_' + date + '.xlsx', 'PassrateData.xlsx', 'PlayerData.xlsx',
                 'PlayerItemDetailsData.xls', 'pass_rate_' + date + '.xls']

    return collect_data_with_player_classification(chosen_AIs, chosen_levels, features, y_feature, filenames)


def collect_data(chosen_AIs, chosen_levels, features, y_feature, filenames):
    data = pd.read_excel(filenames[0])
    userPassrateData = pd.read_excel(filenames[1])
    userData = pd.read_excel(filenames[2])
    userPropDetailedData = pd.read_excel(filenames[3], sheet_name = 1)

    level_num = len(chosen_levels)
    total_feature_num = len(chosen_AIs) * len(features)

    parsed_data = dict()
    for AI in chosen_AIs:
        for level in chosen_levels:
            parsed_data[(AI, level)] = []

    provided_step = dict()
    for level in chosen_levels:
        provided_step[level] = 0

    passrate_data = dict()
    for level in chosen_levels:
        passrate_data[level] = 0.0

    for row in range(data.iloc[:, 0].size):
        level = data.iloc[row, 0]
        AI = data.iloc[row, 2]

        if (AI, level) in parsed_data:
            parsed_data[(AI, level)].append(data.iloc[row, 1])

        if AI == "通关值":
            provided_step[level] = data.iloc[row, 1]

    for row in range(userPassrateData.iloc[:, 0].size):
        level = userPassrateData.iloc[row, 0]
        AI = userPassrateData.iloc[row, 2]

        if AI == "PLAYER":
            passrate_data[level] = userPassrateData.iloc[row, 1] / 100.0

    used_extra_steps_rate = dict()
    total_try_times = dict()
    for level in chosen_levels:
        used_extra_steps_rate[level] = 0.0
        total_try_times[level] = 0

    for row in range(userData.iloc[:, 0].size):
        level = userData.iloc[row, 0]

        try:
            used_extra_steps_rate[level] = (int(userData.iloc[row, 1]) + int(userData.iloc[row, 2])) / \
                                           (int(userData.iloc[row, 6]) + 1e-4)
        except ValueError:
            used_extra_steps_rate[level] = 0.0

        try:
            total_try_times[level] = int(userData.iloc[row, 6])
        except ValueError:
            total_try_times[level] = 0.0

    used_prop_rate = dict()
    for level in chosen_levels:
        used_prop_rate[level] = 0.0

    for row in range(userData.iloc[:, 0].size):
        level = userData.iloc[row, 0]

        try:
            used_prop_rate[level] = int(userData.iloc[row, 3]) / \
                                    (int(userData.iloc[row, 6]) + 1e-4)
        except ValueError:
            used_prop_rate[level] = 0.0

    user_stay_count = dict()
    for level in chosen_levels:
        user_stay_count[level] = 0

    max_level = 0
    for row in range(userData.iloc[:, 0].size):
        level = userData.iloc[row, 0]

        try:
            user_stay_count[level] = int(userData.iloc[row, 8])
        except ValueError:
            user_stay_count[level] = 0

        if level > max_level:
            max_level = level

    for level in reversed(range(2, max_level + 1)):
        user_stay_count[level] = int(user_stay_count[level]) - int(user_stay_count[level - 1])

    user_stay_rate = dict()
    level_group_start = 1
    level_group_end = 10
    while level_group_end < max_level:
        accumulate_stay_count = 0
        for level in range(level_group_start, level_group_end + 1):
            accumulate_stay_count += user_stay_count[level]

        for level in range(level_group_start, level_group_end + 1):
            user_stay_rate[level] = user_stay_count[level] / accumulate_stay_count

        level_group_start += 10
        level_group_end += 10

    use_specific_prop_rate = dict()
    for level in chosen_levels:
        use_specific_prop_rate[level] = dict()

    for row in range(userPropDetailedData.iloc[:, 0].size):
        level = userPropDetailedData.iloc[row, 0]
        try:
            prop_idx = int(userPropDetailedData.iloc[row, 3])
        except ValueError:
            continue

        if level in chosen_levels:
            use_specific_prop_rate[level][prop_idx] = int(userPropDetailedData.iloc[row, 4])

    all_props = [2, 3, 4, 5, 6, 7]
    for level in chosen_levels:
        for prop in all_props:
            if prop in use_specific_prop_rate[level]:
                use_specific_prop_rate[level][prop] = use_specific_prop_rate[level][prop] / (total_try_times[level] + 1e-4)
            else:
                use_specific_prop_rate[level][prop] = 0.0

    data_x = np.zeros([level_num, total_feature_num])
    data_y = np.zeros([level_num, 1])

    level_idx = []

    curr_feature_idx = 0
    for AI in chosen_AIs:
        for feature in features:
            for idx, level in enumerate(chosen_levels):
                if feature == "Passrate":
                    data_x[idx, curr_feature_idx] = (np.array(parsed_data[(AI, level)]) <=
                                                            provided_step[level]).astype(np.float32).mean()
                elif feature == "AvegUsedStepPrecentile":
                    data_x[idx, curr_feature_idx] = \
                        1.0 * np.array(parsed_data[(AI, level)]).mean() / provided_step[level]
                    if data_x[idx, curr_feature_idx] >= 4.0:
                        data_x[idx, curr_feature_idx] = 4.0
                elif feature == "NormalizedStd":
                    data_x[idx, curr_feature_idx] = \
                        1.0 * np.array(parsed_data[(AI, level)]).std() / provided_step[level]
                elif feature == "Emplitude":
                    steps = np.array(parsed_data[(AI, level)])
                    percentiles = np.percentile(steps, [25, 75])
                    data_x[idx, curr_feature_idx] = \
                        (percentiles[1] - percentiles[0]) / provided_step[level]
                else:
                    raise NotImplementedError()

                if curr_feature_idx == 0:
                    if y_feature == "Passrate":
                        data_y[idx, 0] = passrate_data[level]
                    elif y_feature == "UseExtraStepRate":
                        data_y[idx, 0] = used_extra_steps_rate[level]
                    elif y_feature == "UsePropRate":
                        data_y[idx, 0] = used_prop_rate[level]
                    elif y_feature == "UserStayRate":
                        data_y[idx, 0] = user_stay_rate[level]
                    elif y_feature[:5] == "Prop_":
                        data_y[idx, 0] = use_specific_prop_rate[level][int(y_feature[5:])]
                    else:
                        raise NotImplementedError()

                    level_idx.append(level)

            curr_feature_idx += 1

    return level_idx, data_x, data_y


def collect_data2(chosen_AIs, chosen_levels, features, filenames):
    data = pd.read_excel(filenames[0])

    level_num = len(chosen_levels)
    total_feature_num = len(chosen_AIs) * len(features)

    parsed_data = dict()
    for AI in chosen_AIs:
        for level in chosen_levels:
            parsed_data[(AI, level)] = []

    provided_step = dict()
    for level in chosen_levels:
        provided_step[level] = 0

    for row in range(data.iloc[:, 0].size):
        level = data.iloc[row, 0]
        AI = data.iloc[row, 2]

        if (AI, level) in parsed_data:
            parsed_data[(AI, level)].append(data.iloc[row, 1])

        if AI == "通关值":
            provided_step[level] = data.iloc[row, 1]

    data_x = np.zeros([level_num, total_feature_num])

    level_idx = []

    curr_feature_idx = 0
    for AI in chosen_AIs:
        for feature in features:
            for idx, level in enumerate(chosen_levels):
                if feature == "Passrate":
                    data_x[idx, curr_feature_idx] = (np.array(parsed_data[(AI, level)]) <=
                                                            provided_step[level]).astype(np.float32).mean()
                elif feature == "AvegUsedStepPrecentile":
                    if len(parsed_data[(AI, level)]) == 0:
                        print(level, AI)
                    data_x[idx, curr_feature_idx] = \
                        1.0 * np.array(parsed_data[(AI, level)]).mean() / provided_step[level]
                    if data_x[idx, curr_feature_idx] >= 4.0:
                        data_x[idx, curr_feature_idx] = 4.0
                elif feature == "NormalizedStd":
                    data_x[idx, curr_feature_idx] = \
                        1.0 * np.array(parsed_data[(AI, level)]).std() / provided_step[level]
                elif feature == "Emplitude":
                    steps = np.array(parsed_data[(AI, level)])
                    percentiles = np.percentile(steps, [25, 75])
                    data_x[idx, curr_feature_idx] = \
                        (percentiles[1] - percentiles[0]) / provided_step[level]
                else:
                    raise NotImplementedError()

                if curr_feature_idx == 0:
                    level_idx.append(level)

            curr_feature_idx += 1

    return level_idx, data_x


def collect_data_with_player_classification(chosen_AIs, chosen_levels, features, y_feature, filenames):
    level_idx, data_x, data_y = collect_data(chosen_AIs, chosen_levels, features, y_feature, filenames)

    new_level_idx = []
    new_data_x = np.zeros([0, data_x.shape[1] + 1], dtype = np.float32)
    new_data_y = np.zeros([0, 1], dtype = np.float32)

    flag = True
    sheet_idx = 0
    while flag:
        try:
            playerData = pd.read_excel(filenames[4], sheet_name = sheet_idx)
            sheet_idx += 1
        except:
            flag = False
            break

        temp_data_x = np.zeros([data_x.shape[0], data_x.shape[1] + 1], dtype = np.float32)
        temp_data_x[:, :-1] = data_x.copy()

        temp_data_y = np.zeros_like(data_y)

        baseline_people = 0

        for row in range(playerData.iloc[:, 0].size):
            level = playerData.iloc[row, 0]

            if level == 20:
                baseline_people = int(playerData.iloc[row, 5])

            elif level in level_idx:
                idx = level_idx.index(level)

                temp_data_x[idx, -1] = 1.0 * int(playerData.iloc[row, 5]) / baseline_people
                temp_data_y[idx, 0] = float(playerData.iloc[row, 4]) / 100.0

        new_level_idx = new_level_idx + level_idx
        new_data_x = np.concatenate((new_data_x, temp_data_x), axis = 0)
        new_data_y = np.concatenate((new_data_y, temp_data_y), axis = 0)

    return new_level_idx, new_data_x, new_data_y
