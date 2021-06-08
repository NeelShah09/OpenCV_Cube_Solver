from CubeSolver import CubeSolver
from CubeMoves import CubeMove

if __name__ == "__main__":
    CubeSolver().run()

if __name__ == "__main2_":
    scanned_faces = {"Front": ["Orange", "White", "Blue", "Green", "Green", "Green", "Green", "Yellow", "Red"],
                     "Left": ["Orange", "Blue", "Yellow", "White", "Red", "White", "White", "Yellow", "Red"],
                     "Back": ["Orange", "White", "Green", "Orange", "Blue", "Orange", "Blue", "Yellow", "Red"],
                     "Right": ["White", "Yellow", "Blue", "Red", "Orange", "Blue", "Blue", "Orange", "Yellow"],
                     "Top": ["White", "Red", "Yellow", "Red", "White", "Green", "Green", "Blue", "Orange"],
                     "Bottom": ["Yellow", "Blue", "White", "Orange", "Yellow", "Green", "Green", "Red", "Red"]}

    if True:
        c = CubeMove(scanned_faces)
        c.print_current_cube()
        c.D()
        c.L2()
        c.F_prime()
        c.R()
        c.L_prime()
        c.D()
        c.F()
        c.B()
        c.U()
        c.L_prime()
        c.B2()
        c.D()
        c.R2()
        c.D2()
        c.L2()
        c.U_prime()
        c.R2()
        c.L2()
        c.F2()
        c.B2()
        c.U_prime()
        c.print_current_cube()
