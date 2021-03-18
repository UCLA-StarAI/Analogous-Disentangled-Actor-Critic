import numpy as np
import xlwt
import scipy.io as sio
import os


def main():
    model = "A2CSD"
    levels = [1, 2, 3, 4, 5, 6, 7, 9, 10, 13, 14, 15, 16, 19, 22, 27, 41, 42, 43, 52, 55, 58,
              12, 21, 25, 26, 28, 29, 31, 33, 34, 35, 36, 38, 44, 45, 49]
    record_count = 3

    levels = sorted(levels)

    workbook = xlwt.Workbook(encoding = 'ascii')

    worksheet = workbook.add_sheet('Sheet1')

    worksheet.write(0, 0, label = '关卡')
    worksheet.write(0, 1, label = '最多分布次数')
    worksheet.write(0, 2, label = '使用步数')
    worksheet.write(0, 3, label = '对应步数得到的分数结果')
    worksheet.write(0, 4, label = '本关卡AI逻辑')

    row_idx = 1

    for level in levels:
        data = sio.loadmat(os.path.join("logs", model, "mats", str(level) + ".mat"))

        scores = np.reshape(data["scores"], (-1,))
        steps = np.reshape(data["results"], (-1,))

        moves_idxs = np.argsort(data["results"]).reshape(-1)

        worksheet.write(row_idx, 0, level)
        for i in range(-1, -record_count - 1, -1):
            if moves_idxs[i] == 0 or steps[moves_idxs[i]] == 0:
                worksheet.write(row_idx - i - 1, 1, 0)
                worksheet.write(row_idx - i - 1, 2, 0)
                worksheet.write(row_idx - i - 1, 3, 0)
            else:
                worksheet.write(row_idx - i - 1, 1, int(steps[moves_idxs[i]]))
                worksheet.write(row_idx - i - 1, 2, int(moves_idxs[i]))
                worksheet.write(row_idx - i - 1, 3, int(scores[moves_idxs[i]]))

        row_idx += record_count

    workbook.save('Excel_Workbook.xls')


if __name__ == "__main__":
    main()
