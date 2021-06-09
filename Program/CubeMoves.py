import numpy as np


class CubeMove:

    def __init__(self, state, solution):
        self.state = state.copy()
        self.steps_to_solution = solution
        self.next_move = self.steps_to_solution.pop(0)
        self.next_state = self.evaluate_next_state(self.next_move)
        self.shortform = {"White": "W", "Yellow": "Y", "Red": "R", "Orange": "O", "Green": "G", "Blue": "B"}

    def update_states(self):
        try:
            self.state = self.next_state
            self.next_move = self.steps_to_solution.pop(0)
            self.next_state = self.evaluate_next_state(self.next_move)
            return False
        except IndexError:
            self.next_move = ""
            return True

    def get_state(self):
        return self.state

    def L(self):
        temp_state = {}
        for key, val in self.state.items():
            temp_state[key] = val[:]
        for i in range(3):
            temp_state["Left"] = list(np.rot90(np.array(temp_state["Left"]).reshape((3, 3))).reshape(-1))
            temp = [temp_state["Front"][0], temp_state["Front"][3], temp_state["Front"][6]]
            [temp_state["Front"][0], temp_state["Front"][3], temp_state["Front"][6]] = [temp_state["Bottom"][0],
                                                                                        temp_state["Bottom"][3],
                                                                                        temp_state["Bottom"][6]]
            [temp_state["Bottom"][0], temp_state["Bottom"][3], temp_state["Bottom"][6]] = [temp_state["Back"][8],
                                                                                           temp_state["Back"][5],
                                                                                           temp_state["Back"][2]]
            [temp_state["Back"][8], temp_state["Back"][5], temp_state["Back"][2]] = [temp_state["Top"][0],
                                                                                     temp_state["Top"][3],
                                                                                     temp_state["Top"][6]]
            [temp_state["Top"][0], temp_state["Top"][3], temp_state["Top"][6]] = temp
        return temp_state

    def L_prime(self):
        temp_state = {}
        for key, val in self.state.items():
            temp_state[key] = val[:]
        temp_state["Left"] = list(np.rot90(np.array(temp_state["Left"]).reshape((3, 3))).reshape(-1))
        temp = [temp_state["Front"][0], temp_state["Front"][3], temp_state["Front"][6]]
        [temp_state["Front"][0], temp_state["Front"][3], temp_state["Front"][6]] = [temp_state["Bottom"][0], temp_state["Bottom"][3], temp_state["Bottom"][6]]
        [temp_state["Bottom"][0], temp_state["Bottom"][3], temp_state["Bottom"][6]] = [temp_state["Back"][8], temp_state["Back"][5], temp_state["Back"][2]]
        [temp_state["Back"][8], temp_state["Back"][5], temp_state["Back"][2]] = [temp_state["Top"][0], temp_state["Top"][3], temp_state["Top"][6]]
        [temp_state["Top"][0], temp_state["Top"][3], temp_state["Top"][6]] = temp
        return temp_state

    def R(self):
        temp_state = {}
        for key, val in self.state.items():
            temp_state[key] = val[:]
        for i in range(3):
            temp_state["Right"] = list(np.rot90(np.array(temp_state["Right"]).reshape((3, 3))).reshape(-1))
            temp = [temp_state["Front"][2], temp_state["Front"][5], temp_state["Front"][8]]
            [temp_state["Front"][2], temp_state["Front"][5], temp_state["Front"][8]] = [temp_state["Top"][2],
                                                                                        temp_state["Top"][5],
                                                                                        temp_state["Top"][8]]
            [temp_state["Top"][2], temp_state["Top"][5], temp_state["Top"][8]] = [temp_state["Back"][6],
                                                                                  temp_state["Back"][3],
                                                                                  temp_state["Back"][0]]
            [temp_state["Back"][0], temp_state["Back"][3], temp_state["Back"][6]] = [temp_state["Bottom"][8],
                                                                                     temp_state["Bottom"][5],
                                                                                     temp_state["Bottom"][2]]
            [temp_state["Bottom"][2], temp_state["Bottom"][5], temp_state["Bottom"][8]] = temp
        return temp_state

    def R_prime(self):
        temp_state = {}
        for key, val in self.state.items():
            temp_state[key] = val[:]
        temp_state["Right"] = list(np.rot90(np.array(temp_state["Right"]).reshape((3, 3))).reshape(-1))
        temp = [temp_state["Front"][2], temp_state["Front"][5], temp_state["Front"][8]]
        [temp_state["Front"][2], temp_state["Front"][5], temp_state["Front"][8]] = [temp_state["Top"][2],
                                                                                    temp_state["Top"][5],
                                                                                    temp_state["Top"][8]]
        [temp_state["Top"][2], temp_state["Top"][5], temp_state["Top"][8]] = [temp_state["Back"][6],
                                                                                       temp_state["Back"][3],
                                                                                       temp_state["Back"][0]]
        [temp_state["Back"][0], temp_state["Back"][3], temp_state["Back"][6]] = [temp_state["Bottom"][8],
                                                                                 temp_state["Bottom"][5],
                                                                                 temp_state["Bottom"][2]]
        [temp_state["Bottom"][2], temp_state["Bottom"][5], temp_state["Bottom"][8]] = temp
        return temp_state

    def U(self):
        temp_state = {}
        for key, val in self.state.items():
            temp_state[key] = val[:]
        for i in range(3):
            temp_state["Top"] = list(np.rot90(np.array(temp_state["Top"].copy()).reshape((3, 3))).reshape(-1))
            temp = temp_state["Back"][0:3]
            temp_state["Back"][0:3] = temp_state["Right"][0:3]
            temp_state["Right"][0:3] = temp_state["Front"][0:3]
            temp_state["Front"][0:3] = temp_state["Left"][0:3]
            temp_state["Left"][0:3] = temp
        return temp_state

    def U_prime(self):
        temp_state = {}
        for key, val in self.state.items():
            temp_state[key] = val[:]
        temp_state["Top"] = list(np.rot90(np.array(temp_state["Top"]).reshape((3, 3))).reshape(-1))
        temp = temp_state["Back"][0:3]
        temp_state["Back"][0:3] = temp_state["Right"][0:3]
        temp_state["Right"][0:3] = temp_state["Front"][0:3]
        temp_state["Front"][0:3] = temp_state["Left"][0:3]
        temp_state["Left"][0:3] = temp
        return temp_state

    def D(self):
        temp_state = {}
        for key, val in self.state.items():
            temp_state[key] = val[:]
        for i in range(3):
            temp_state["Bottom"] = list(np.rot90(np.array(temp_state["Bottom"]).reshape((3, 3))).reshape(-1))
            temp = temp_state["Front"][6:9]
            temp_state["Front"][6:9] = temp_state["Right"][6:9]
            temp_state["Right"][6:9] = temp_state["Back"][6:9]
            temp_state["Back"][6:9] = temp_state["Left"][6:9]
            temp_state["Left"][6:9] = temp
        return temp_state

    def D_prime(self):
        temp_state = {}
        for key, val in self.state.items():
            temp_state[key] = val[:]
        temp_state["Bottom"] = list(np.rot90(np.array(temp_state["Bottom"]).reshape((3, 3))).reshape(-1))
        temp = temp_state["Front"][6:9]
        temp_state["Front"][6:9] = temp_state["Right"][6:9]
        temp_state["Right"][6:9] = temp_state["Back"][6:9]
        temp_state["Back"][6:9] = temp_state["Left"][6:9]
        temp_state["Left"][6:9] = temp
        return temp_state

    def F(self):
        temp_state = {}
        for key, val in self.state.items():
            temp_state[key] = val[:]
        for i in range(3):
            temp_state["Front"] = list(np.rot90(np.array(temp_state["Front"]).reshape((3, 3))).reshape(-1))
            temp = temp_state["Top"][6:9]
            temp_state["Top"][6:9] = [temp_state["Right"][0], temp_state["Right"][3], temp_state["Right"][6]]
            [temp_state["Right"][6], temp_state["Right"][3], temp_state["Right"][0]] = temp_state["Bottom"][0:3]
            temp_state["Bottom"][0:3] = [temp_state["Left"][2], temp_state["Left"][5], temp_state["Left"][8]]
            [temp_state["Left"][8], temp_state["Left"][5], temp_state["Left"][2]] = temp
        return temp_state

    def F_prime(self):
        temp_state = {}
        for key, val in self.state.items():
            temp_state[key] = val[:]
        temp_state["Front"] = list(np.rot90(np.array(temp_state["Front"]).reshape((3, 3))).reshape(-1))
        temp = temp_state["Top"][6:9]
        temp_state["Top"][6:9] = [temp_state["Right"][0], temp_state["Right"][3], temp_state["Right"][6]]
        [temp_state["Right"][6], temp_state["Right"][3], temp_state["Right"][0]] = temp_state["Bottom"][0:3]
        temp_state["Bottom"][0:3] = [temp_state["Left"][2], temp_state["Left"][5], temp_state["Left"][8]]
        [temp_state["Left"][8], temp_state["Left"][5], temp_state["Left"][2]] = temp
        return temp_state

    def B(self):
        temp_state = {}
        for key, val in self.state.items():
            temp_state[key] = val[:]
        for i in range(3):
            temp_state["Back"] = list(np.rot90(np.array(temp_state["Back"]).reshape((3, 3))).reshape(-1))
            temp = temp_state["Top"][0:3]
            temp_state["Top"][0:3] = [temp_state["Left"][6], temp_state["Left"][3], temp_state["Left"][0]]
            [temp_state["Left"][0], temp_state["Left"][3], temp_state["Left"][6]] = temp_state["Bottom"][6:9]
            temp_state["Bottom"][6:9] = [temp_state["Right"][8], temp_state["Right"][5], temp_state["Right"][2]]
            [temp_state["Right"][2], temp_state["Right"][5], temp_state["Right"][8]] = temp
        return temp_state

    def B_prime(self):
        temp_state = {}
        for key, val in self.state.items():
            temp_state[key] = val[:]
        temp_state["Back"] = list(np.rot90(np.array(temp_state["Back"]).reshape((3, 3))).reshape(-1))
        temp = temp_state["Top"][0:3]
        temp_state["Top"][0:3] = [temp_state["Left"][6], temp_state["Left"][3], temp_state["Left"][0]]
        [temp_state["Left"][0], temp_state["Left"][3], temp_state["Left"][6]] = temp_state["Bottom"][6:9]
        temp_state["Bottom"][6:9] = [temp_state["Right"][8], temp_state["Right"][5], temp_state["Right"][2]]
        [temp_state["Right"][2], temp_state["Right"][5], temp_state["Right"][8]] = temp
        return temp_state

    def print_current_cube(self):
        print("        -------")
        print("        |" + self.shortform[self.state["Top"][0]] + " " + self.shortform[self.state["Top"][1]] + " " + self.shortform[self.state["Top"][2]] + "|")
        print("        |" + self.shortform[self.state["Top"][3]] + " " + self.shortform[self.state["Top"][4]] + " " + self.shortform[self.state["Top"][5]] + "|")
        print("        |" + self.shortform[self.state["Top"][6]] + " " + self.shortform[self.state["Top"][7]] + " " + self.shortform[self.state["Top"][8]] + "|")

        print("-------------------------------")

        print("|" + self.shortform[self.state["Left"][0]] + " " + self.shortform[self.state["Left"][1]] + " " + self.shortform[self.state["Left"][2]] + "|", end="")
        print(" |" + self.shortform[self.state["Front"][0]] + " " + self.shortform[self.state["Front"][1]] + " " + self.shortform[self.state["Front"][2]] + "|", end="")
        print(" |" + self.shortform[self.state["Right"][0]] + " " + self.shortform[self.state["Right"][1]] + " " + self.shortform[self.state["Right"][2]] + "|", end="")
        print(" |" + self.shortform[self.state["Back"][0]] + " " + self.shortform[self.state["Back"][1]] + " " + self.shortform[self.state["Back"][2]] + "|")

        print("|" + self.shortform[self.state["Left"][3]] + " " + self.shortform[self.state["Left"][4]] + " " +
              self.shortform[self.state["Left"][5]] + "|", end="")
        print(" |" + self.shortform[self.state["Front"][3]] + " " + self.shortform[self.state["Front"][4]] + " " +
              self.shortform[self.state["Front"][5]] + "|", end="")
        print(" |" + self.shortform[self.state["Right"][3]] + " " + self.shortform[self.state["Right"][4]] + " " +
              self.shortform[self.state["Right"][5]] + "|", end="")
        print(" |" + self.shortform[self.state["Back"][3]] + " " + self.shortform[self.state["Back"][4]] + " " +
              self.shortform[self.state["Back"][5]] + "|")

        print("|" + self.shortform[self.state["Left"][6]] + " " + self.shortform[self.state["Left"][7]] + " " +
              self.shortform[self.state["Left"][8]] + "|", end="")
        print(" |" + self.shortform[self.state["Front"][6]] + " " + self.shortform[self.state["Front"][7]] + " " +
              self.shortform[self.state["Front"][8]] + "|", end="")
        print(" |" + self.shortform[self.state["Right"][6]] + " " + self.shortform[self.state["Right"][7]] + " " +
              self.shortform[self.state["Right"][8]] + "|", end="")
        print(" |" + self.shortform[self.state["Back"][6]] + " " + self.shortform[self.state["Back"][7]] + " " +
              self.shortform[self.state["Back"][8]] + "|")
        print("-------------------------------")

        print("        |" + self.shortform[self.state["Bottom"][0]] + " " + self.shortform[self.state["Bottom"][1]] + " " +
              self.shortform[self.state["Bottom"][2]] + "|")
        print("        |" + self.shortform[self.state["Bottom"][3]] + " " + self.shortform[self.state["Bottom"][4]] + " " +
              self.shortform[self.state["Bottom"][5]] + "|")
        print("        |" + self.shortform[self.state["Bottom"][6]] + " " + self.shortform[self.state["Bottom"][7]] + " " +
              self.shortform[self.state["Bottom"][8]] + "|")
        print("        -------")

    def evaluate_next_state(self, next_move):
        if next_move == "R":
            return self.R()
        elif next_move == "R'":
            return self.R_prime()
        elif next_move == "L":
            return self.L()
        elif next_move == "L'":
            return self.L_prime()
        elif next_move == "U":
            return self.U()
        elif next_move == "U'":
            return self.U_prime()
        elif next_move == "D":
            return self.D()
        elif next_move == "D'":
            return self.D_prime()
        elif next_move == "F":
            return self.F()
        elif next_move == "F'":
            return self.F_prime()
        elif next_move == "B":
            return self.B()
        elif next_move == "B'":
            return self.B_prime()
        else:
            return ""
