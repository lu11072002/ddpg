from matplotlib import pyplot as plt

class DrawLog:
    def __init__(self):
        self.data = {}
        
    def draw_data_OneForEach(self, file_name):
        f = open(f"data/{file_name}/logger.log")
        self.data[file_name] = {}
        self.data[file_name]["Episode"] = []
        self.data[file_name]["evaluating"] = []
        
        for line in f.readlines():
            d = line.split(" ")
            if d[0] == "WARNING:root:Episode":
                self.data[file_name]["Episode"].append(float(d[-1].replace(".\n", "")))
            elif d[0] == "WARNING:root:evaluating":
                self.data[file_name]["evaluating"].append(float(d[-1].replace(".\n", "")))
        
    def draw_data_OneForAll(self, file_name):
        f = open(f"data/{file_name}/logger.log")
        name = ""
        
        for line in f.readlines():
            d = line.split(" ")
            if d[0] == "WARNING:root:Episode":
                name = d[-1].replace(".\n", "")
                if name not in self.data:
                    self.data[name] = {}
                    self.data[name]["Episode"] = []
                    self.data[name]["evaluating"] = []
                self.data[name]["Episode"].append(float(d[-3]))
            elif d[0] == "WARNING:root:evaluating":
                self.data[name]["evaluating"].append(float(d[-1].replace(".\n", "")))
        
    def plot(self):
        index = 1
        nbr = len(self.data)
        for name in self.data:
            evalutations = self.data[name]["evaluating"]
            plt.subplot(nbr, 2, index)
            plt.title(f"{name} : Evaluation during trainning")
            plt.plot([i*100 for i in range(len(evalutations))], evalutations)
            plt.xlabel("Epoch")
            plt.ylabel("Reward")
            
            episodes = self.data[name]["Episode"]
            plt.subplot(nbr, 2, index + 1)
            plt.title(f"{name} : Trainning")
            plt.plot([i*10 for i in range(len(episodes))], episodes)
            plt.xlabel("Epoch")
            plt.ylabel("Reward")
            
            index += 2
        
    def display(self):
        self.plot()
        plt.tight_layout()
        plt.show()
        self.data = {}
    
    def save(self, file_name, sizex, sizey):
        self.plot()
        plt.tight_layout()
        plt.gcf().set_size_inches(sizex/100, sizey/100)
        plt.savefig(file_name, dpi=100)
        self.data = {}        

if __name__ == "__main__":
    drawer = DrawLog()
    #drawer.draw_data_OneForEach("reach-v2-goal-observable")
    #drawer.draw_data_OneForEach("drawer-close-v2-goal-observable")
    #drawer.draw_data_OneForEach("window-open-v2-goal-observable")
    drawer.draw_data_OneForAll("OneForAll")
    drawer.display()
    #drawer.save("images/graph_OneForAll.jpg", 1270, 720)