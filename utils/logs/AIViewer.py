import tkinter as tk
import scipy.io as sio
import numpy as np


class Interface():
    def __init__(self):
        # Interface part
        self.window = tk.Tk()
        tk.Label(self.window, text = "Level file name:").grid(row = 0)
        self.level_file_name = tk.StringVar()
        tk.Entry(self.window, textvariable = self.level_file_name, width = 45).grid(row = 0, column = 1)
        tk.Button(self.window, text = "Import", command = self.import_file).grid(row = 0, column = 2)

        self.text = tk.Text(self.window, width = 20, height = 20)
        self.text.grid(row = 1)

        self.text2 = tk.Text(self.window, width = 20, height = 20)
        self.text2.grid(row = 1, column = 1)

        self.window.mainloop()

    def import_file(self):
        file_name = self.level_file_name.get()

        try:
            data = sio.loadmat(file_name)
        except:
            return

        data["scores"] = np.reshape(data["scores"], (-1,))
        data["results"] = np.reshape(data["results"], (-1,))

        total_run_time = data["scores"].size

        moves_idxs = np.argsort(data["results"])

        self.text.delete(1.0, tk.END)

        self.text2.delete(1.0, tk.END)

        self.text.insert(tk.END, "  ---Moves---\n")

        self.text2.insert(tk.END, "---Scores---\n")

        for i in range(-1, -16, -1):
            if moves_idxs[i] == 0 or data["results"][moves_idxs[i]] == 0:
                break
            self.text.insert(tk.END, "{0:3d}: {1:5d} times\n".format(moves_idxs[i], data["results"][moves_idxs[i]]))

            self.text2.insert(tk.END, "{}\n".format(data["scores"][moves_idxs[i]] / data["results"][moves_idxs[i]]))




def main():
    interface = Interface()


if __name__ == '__main__':
    main()
