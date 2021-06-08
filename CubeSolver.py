# from utility.Stream import Stream
import numpy as np
import cv2
from constants import *
from twophase import solve, solve_best
from threading import Thread
from CubeMoves import CubeMove


class CubeSolver:
    def __init__(self):
        self.calibrating_mode = False
        self.flipped_image = False
        # self.cam = Stream().start()
        self.cam = cv2.VideoCapture(0)
        self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        self.cube = None

        self.contour_copy = None

        self.current_calibrating_colour = 0
        self.colours_list = ["Red", "Green", "White", "Yellow", "Orange", "Blue"]
        self.standard_colours = {"Green": (0, 255, 0), "Blue": (255, 0, 0), "Red": (0, 0, 255),
                                 "Yellow": (0, 255, 255), "Orange": (0, 165, 255), "White": (255, 255, 255)}
        self.calibrated_colours = {"Green": (), "Blue": (), "Red": (), "Yellow": (), "Orange": (), "White": ()}
        self.done_calibration = False

        self.face_list = ["Top", "Right", "Front", "Bottom", "Left", "Back"]
        self.colour_to_face = {"White": "Top", "Yellow": "Bottom", "Red": "Left", "Orange": "Right", "Green": "Front", "Blue": "Back"}
        self.translate = {"White": "U", "Yellow": "D", "Red": "L", "Orange": "R", "Green": "F", "Blue": "B"}
        self.scanned_faces = {"Front": None, "Left": None, "Back": None, "Right": None, "Top": None, "Bottom": None}

        # self.scanned_faces = {"Front": ["Green", "Yellow", "Orange", "Red", "Green", "White", "Red", "Orange", "Blue"],
        #                       "Left": ["Orange", "Yellow", "Yellow", "Green", "Red", "Blue", "White", "Yellow", "Green"],
        #                       "Back": ["Green", "Orange", "White", "Red", "Blue", "Orange", "Red", "Yellow", "Blue"],
        #                       "Right": ["White", "Red", "Orange", "Green", "Orange", "White", "Yellow", "White", "Blue"],
        #                       "Top": ["Blue", "Blue", "Yellow", "Blue", "White", "Green", "Red", "Red", "Green"],
        #                       "Bottom": ["White", "White", "Orange", "Orange", "Yellow", "Blue", "Red", "Green", "Yellow"]}

        self.done_scanning = False
        self.done_validation = False
        self.solution = None

    def run(self):
        while True:
            key = cv2.waitKey(1)
            # frame = cv2.flip(self.cam.read()[1], 1)
            frame = self.cam.read()[1]
            # frame = cv2.flip(self.cam.read(), 1)
            # frame = self.cam.read()
            self.contour_copy = frame.copy()
            if key == 27:
                break
            if key == 32:
                self.calibrating_mode = not self.calibrating_mode
                self.reset_scanning()
                if self.calibrating_mode:
                    self.reset_calibration()

            if key == ord("r"):
                if self.calibrating_mode:
                    self.reset_calibration()
                else:
                    self.reset_scanning()

            grayImg = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blurImg = cv2.blur(grayImg, (3, 3))
            cannyImg = cv2.Canny(blurImg, 40, 100)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
            dilatedFrame = cv2.dilate(cannyImg, kernel)
            qb_boxes = self.reorder(self.find_contours(dilatedFrame))

            if self.calibrating_mode:
                if key == ord("c") and qb_boxes != [] and not self.done_calibration:
                    image = self.contour_copy[qb_boxes[4][1]:qb_boxes[4][1]+qb_boxes[4][3], qb_boxes[4][0]:qb_boxes[4][0]+qb_boxes[4][2]]
                    self.calibrate_colour(image)
                self.update_preview_tab()
                self.update_text_print_calib()
                self.draw_boxes(qb_boxes, centre=True)
            else:
                self.draw_single_white_box(qb_boxes)
                detected_colours, detected_colours_name = self.scan_colour(qb_boxes)
                self.update_single_face_preview(detected_colours)
                if key == ord("c") and detected_colours != []:
                    self.record_face(detected_colours_name)
                self.update_text_print_scan()

                if self.solution is not None and self.solution != "Already Solved" and qb_boxes != []:
                    current_face = self.colour_to_face[detected_colours_name[4]]
                    if detected_colours_name == self.cube.get_state()[current_face]:
                        self.draw_next_step_arrows(qb_boxes, current_face=current_face, next_move=self.cube.next_move)
                    elif detected_colours_name == self.cube.next_state[current_face]:
                        if self.cube.update_states():
                            self.solution = "Solved"
                        self.scanned_faces = self.cube.get_state()

            # cv2.imshow("CubeSolver", frame)
            cv2.imshow("Canny", dilatedFrame)
            cv2.imshow("Cube Solver", self.contour_copy)
        cv2.destroyAllWindows()
        self.cam.release()

    def find_contours(self, frame_for_process):
        qb_list = []
        contours, _ = cv2.findContours(frame_for_process, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            per = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.12 * per, True)
            area = cv2.contourArea(contour)
            if len(approx) == 4 and area > AREA_THRESH_MIN and area < AREA_THRESH_MAX:
                (x, y, w, h) = cv2.boundingRect(contour)
                qb_list.append((x, y, w, h))
        qb_list = np.array(qb_list)
        grid_coordinates = []
        # print("Before :", qb_list)
        for index, pt in enumerate(qb_list):
            temp_list = np.abs(qb_list[:, :2] - qb_list[:, :2][index])
            temp_list = np.sum(temp_list, axis=1, keepdims=True)

            if len(temp_list[temp_list < 50]) >= 3:
                grid_coordinates.append((pt[0], pt[1], pt[2], pt[3]))
                #print("Centre index :", index)
        if len(grid_coordinates) >= 9:
            return grid_coordinates
        else:
            return []

    def reorder(self, grid):
        y_grid = sorted(grid, key = lambda x : x[1])
        upp = sorted(y_grid[0:3], key = lambda x : x[0])
        mid = sorted(y_grid[3:6], key=lambda x: x[0])
        low = sorted(y_grid[6:9], key=lambda x: x[0])
        return upp + mid + low if len(upp) or len(mid) or len(low) else []

    def draw_single_white_box(self, box_list):
        if box_list != [] and not self.done_scanning:
            x, y, w, h = box_list[0][0]-10, box_list[0][1]-10, box_list[8][0] - box_list[0][0] + box_list[8][2] + 18, box_list[8][1] - box_list[0][1] + box_list[8][3] + 18
            cv2.rectangle(self.contour_copy, (x, y), (x + w, y + h), (255, 255, 255), 2)
            cv2.line(self.contour_copy, (x + w // 3, y), (x + w // 3, y + h), (255, 255, 255), 2)
            cv2.line(self.contour_copy, (x + 2 * w // 3, y), (x + 2 * w // 3, y + h), (255, 255, 255), 2)
            cv2.line(self.contour_copy, (x, y + h // 3), (x + w, y + h // 3), (255, 255, 255), 2)
            cv2.line(self.contour_copy, (x, y + 2 * h // 3), (x + w, y + 2 * h // 3), (255, 255, 255), 2)

    def draw_boxes(self, box_list, colour=(0, 255, 0), centre=False):
        if centre and box_list != []:
            self.contour_copy = cv2.rectangle(self.contour_copy, (box_list[4][0], box_list[4][1]), (box_list[4][0] + box_list[4][2], box_list[4][1] + box_list[4][3]), colour, 2)
        else:
            for box in box_list:
                self.contour_copy = cv2.rectangle(self.contour_copy, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), colour, 2)

    def update_text_print_calib(self):
        self.contour_copy = cv2.putText(self.contour_copy, "Calibrating Mode",
                                        (int(self.cam.get(cv2.CAP_PROP_FRAME_WIDTH)//2)-130, 30), cv2.FONT_HERSHEY_TRIPLEX,
                                        1, (255, 255, 255), 2)
        if not self.done_calibration:
            colour_to_scan = "Colour to scan : {}".format(self.colours_list[self.current_calibrating_colour])
            self.contour_copy = cv2.putText(self.contour_copy, colour_to_scan,
                                            (10, int(self.cam.get(cv2.CAP_PROP_FRAME_HEIGHT))-20),
                                            cv2.FONT_HERSHEY_SIMPLEX,
                                            0.7, self.standard_colours[self.colours_list[self.current_calibrating_colour]], 2)
        else:
            if self.solution is None:
                pass
            self.contour_copy = cv2.putText(self.contour_copy, "Done Calibration",
                                            (10, int(self.cam.get(cv2.CAP_PROP_FRAME_HEIGHT)) - 20),
                                            cv2.FONT_HERSHEY_SIMPLEX,
                                            0.7, (255, 255, 255), 2)

    def update_text_print_scan(self):
        self.draw_2d_cube()
        self.contour_copy = cv2.putText(self.contour_copy, "Scanning Mode",
                                        (int(self.cam.get(cv2.CAP_PROP_FRAME_WIDTH) // 2) - 120, 30),
                                        cv2.FONT_HERSHEY_TRIPLEX,
                                        1, (255, 255, 255), 2)
        if not self.done_scanning:
            self.contour_copy = cv2.putText(self.contour_copy, "Scan Faces",
                                            (10, int(self.cam.get(cv2.CAP_PROP_FRAME_HEIGHT))-20),
                                            cv2.FONT_HERSHEY_SIMPLEX,
                                            0.7, (255, 255, 255), 2)
        elif self.done_scanning and not self.done_validation:
            self.contour_copy = cv2.putText(self.contour_copy, "Invalid Cube : Press R to re-scan",
                                            (150, int(self.cam.get(cv2.CAP_PROP_FRAME_HEIGHT)) // 2),
                                            cv2.FONT_HERSHEY_SIMPLEX,
                                            0.7, (0, 0, 255), 2)
        elif self.solution is None:
            self.contour_copy = cv2.putText(self.contour_copy, "Solving Cube...",
                                            (10, int(self.cam.get(cv2.CAP_PROP_FRAME_HEIGHT)) - 20),
                                            cv2.FONT_HERSHEY_SIMPLEX,
                                            0.5, (255, 255, 255), 2)
        else:
            self.contour_copy = cv2.putText(self.contour_copy, "Solution : " + self.solution,
                                            (10, int(self.cam.get(cv2.CAP_PROP_FRAME_HEIGHT)) - 20),
                                            cv2.FONT_HERSHEY_SIMPLEX,
                                            0.5, (255, 255, 255), 2)

    def update_preview_tab(self):
        for i in range(self.current_calibrating_colour):
            self.contour_copy = cv2.rectangle(self.contour_copy, (20, 50 + i * PREVIEW_HEIGHT),
                                              (20 + PREVIEW_WIDTH, 50 + (i + 1) * PREVIEW_HEIGHT), (255, 255, 255), 1)
            pts_list = np.array([[22, 52 + i * PREVIEW_HEIGHT], [18 + PREVIEW_WIDTH, 52 + i * PREVIEW_HEIGHT],
                                 [18 + PREVIEW_WIDTH, 48 + (i + 1) * PREVIEW_HEIGHT], [22, 48 + (i + 1) * PREVIEW_HEIGHT]])
            cv2.fillPoly(self.contour_copy, [pts_list], self.calibrated_colours[self.colours_list[i]])

    def draw_2d_cube(self):
        for face in self.face_list:
            for i in range(3):
                for j in range(3):
                    cv2.rectangle(self.contour_copy, (FACE_POS_START[face][0] + i * QB_SIDE_LEN, FACE_POS_START[face][1] + j * QB_SIDE_LEN),
                                  (FACE_POS_START[face][0] + (i + 1) * QB_SIDE_LEN, FACE_POS_START[face][1] + (j + 1) * QB_SIDE_LEN),
                                  (0, 0, 0), 2)
                    pts = np.array([(FACE_POS_START[face][0] + i * QB_SIDE_LEN, FACE_POS_START[face][1] + j * QB_SIDE_LEN),
                           (FACE_POS_START[face][0] + (i + 1) * QB_SIDE_LEN, FACE_POS_START[face][1] + j * QB_SIDE_LEN),
                           (FACE_POS_START[face][0] + (i + 1) * QB_SIDE_LEN, FACE_POS_START[face][1] + (j + 1) * QB_SIDE_LEN),
                           (FACE_POS_START[face][0] + i * QB_SIDE_LEN, FACE_POS_START[face][1] + (j + 1) * QB_SIDE_LEN)])
                    colour = self.standard_colours[self.scanned_faces[face][3 * j + i]] if self.scanned_faces[face] is not None else (100, 100, 100)
                    # print(temp)
                    # colour = (100, 100, 100)
                    cv2.fillPoly(self.contour_copy, [pts], colour)

    def draw_next_step_arrows(self, boxes, current_face, next_move):
        if current_face == "Front":
            if next_move == "R":
                arrow_pts = [(6, 0)] if self.flipped_image else [(8, 2)]
            elif next_move == "R'":
                arrow_pts = [(0, 6)] if self.flipped_image else [(2, 8)]
            elif next_move == "L":
                arrow_pts = [(2, 8)] if self.flipped_image else [(0, 6)]
            elif next_move == "L'":
                arrow_pts = [(8, 2)] if self.flipped_image else [(0, 6)]
            elif next_move == "U":
                arrow_pts = [(0, 2)] if self.flipped_image else [(2, 0)]
            elif next_move == "U'":
                arrow_pts = [(2, 0)] if self.flipped_image else [(0, 2)]
            elif next_move == "D":
                arrow_pts = [(8, 6)] if self.flipped_image else [(6, 8)]
            elif next_move == "D'":
                arrow_pts = [(6, 8)] if self.flipped_image else [(8, 6)]
            elif next_move == "F":
                arrow_pts = [(2, 0), (0, 6), (6, 8), (8, 2)] if self.flipped_image else [(0, 2), (2, 8), (8, 6), (6, 0)]
            elif next_move == "F'":
                arrow_pts = [(0, 2), (2, 8), (8, 6), (6, 0)] if self.flipped_image else [(2, 0), (0, 6), (6, 8), (8, 2)]
            elif next_move == "B" or next_move == "B'":
                arrow_pts = [(0, 2), (3, 5), (6, 8)] if self.flipped_image else [(2, 0), (5, 3), (8, 6)]
            else:
                arrow_pts = []
            flag = (len(arrow_pts) == 4)
            for i, (start_box, end_box) in enumerate(arrow_pts):
                cv2.arrowedLine(self.contour_copy,
                                (boxes[start_box][0] + boxes[start_box][2] // 2, boxes[start_box][1] + boxes[start_box][3] // 2),
                                (boxes[end_box][0] + boxes[end_box][2] // 2, boxes[end_box][1] + boxes[end_box][3] // 2),
                                (255, 255, 0), 2, cv2.LINE_AA)
        elif current_face == "Right":
            if next_move == "B":
                arrow_pts = [(6, 0)] if self.flipped_image else [(8, 2)]
            elif next_move == "B'":
                arrow_pts = [(0, 6)] if self.flipped_image else [(2, 8)]
            elif (next_move == "F" or next_move == "F'" or next_move == "U" or next_move == "U'" or next_move == "L" or next_move == "L'"
                    or next_move == "R" or next_move == "R'" or next_move == "D" or next_move == "D'"):
                arrow_pts = [(2, 0), (5, 3), (8, 6)] if self.flipped_image else [(0, 2), (3, 5), (6, 8)]
            else:
                arrow_pts = []
            for start_point, end_point in arrow_pts:
                cv2.arrowedLine(self.contour_copy,
                                (boxes[start_point][0] + boxes[start_point][2] // 2, boxes[start_point][1] + boxes[start_point][3] // 2),
                                (boxes[end_point][0] + boxes[end_point][2] // 2, boxes[end_point][1] + boxes[end_point][3] // 2),
                                (255, 255, 0), 2, cv2.LINE_AA)

    def find_colour(self, img):
        return np.round(img.mean(axis=0).mean(axis=0))

    def reset_calibration(self):
        self.calibrated_colours = {"Green": (), "Blue": (), "Red": (), "Yellow": (), "Orange": (), "White": ()}
        self.current_calibrating_colour = 0
        self.done_calibration = False

    def reset_scanning(self):
        self.done_scanning = False
        self.done_validation = False
        self.solution = None
        self.scanned_faces = {"Front": None, "Left": None, "Back": None, "Right": None, "Top": None, "Bottom": None}

    def calibrate_colour(self, box_image):
        colour = self.find_colour(box_image)
        self.calibrated_colours[self.colours_list[self.current_calibrating_colour]] = (colour)
        self.current_calibrating_colour += 1
        self.done_calibration = self.current_calibrating_colour == 6
        # print(self.calibrated_colours)

    def find_nearest_colour(self, colour):
        nearest_dist = None
        nearest_colour = None
        nearest_colour_name = None
        for i in range(6):
            ref = self.calibrated_colours[self.colours_list[i]] if self.done_calibration else self.standard_colours[self.colours_list[i]]
            b1, g1, r1 = ref
            b2, g2, r2 = colour
            dist = (b2 - b1)**2 + (g2 - g1)**2 + (r2 - r1)**2
            if nearest_dist is None or dist < nearest_dist:
                nearest_colour = ref
                nearest_dist = dist
                nearest_colour_name = self.colours_list[i]
        return nearest_colour, nearest_colour_name

    def scan_colour(self, boxes):
        col_val_list = []
        col_name_list = []
        for i in range(len(boxes)):
            image = self.contour_copy[boxes[i][1]:boxes[i][1] + boxes[i][3], boxes[i][0]:boxes[i][0] + boxes[i][2]]
            col, col_name = self.find_nearest_colour(self.find_colour(image))
            col_val_list.append(col)
            col_name_list.append(col_name)
        if self.flipped_image:
            col_val_list = list(reversed(col_val_list[0:3])) + list(reversed(col_val_list[3:6])) + list(reversed(col_val_list[6:9]))
            col_name_list = list(reversed(col_name_list[0:3])) + list(reversed(col_name_list[3:6])) + list(reversed(col_name_list[6:9]))
        return col_val_list, col_name_list


    def update_single_face_preview(self, colour):
        if self.flipped_image:
            colour = list(reversed(colour[0:3])) + list(reversed(colour[3:6])) + list(reversed(colour[6:9]))
        for i in range(len(colour)//3):
            for j in range(3):
                cv2.rectangle(self.contour_copy,
                              (CUBE_FACE_X + j * QB_LEN, CUBE_FACE_Y + i * QB_LEN),
                              (CUBE_FACE_X + (j + 1) * QB_LEN, CUBE_FACE_Y + (i + 1) * QB_LEN),
                              (255, 255, 255), 1)
                poly_pts = np.array([[CUBE_FACE_X + j * QB_LEN + 2, CUBE_FACE_Y + i * QB_LEN + 2],
                            [CUBE_FACE_X + j * QB_LEN + 2, CUBE_FACE_Y + (i + 1) * QB_LEN - 2],
                            [CUBE_FACE_X + (j + 1) * QB_LEN - 2, CUBE_FACE_Y + (i + 1) * QB_LEN - 2],
                            [CUBE_FACE_X + (j + 1) * QB_LEN - 2, CUBE_FACE_Y + i * QB_LEN + 2]])
                cv2.fillPoly(self.contour_copy, [poly_pts], colour[3 * i + j])

    def record_face(self, colours):
        self.scanned_faces[self.colour_to_face[colours[4]]] = colours
        self.done_scanning = all(self.scanned_faces.values())
        self.done_validation = False
        if self.done_scanning:
            self.validate_scanned_faces()
            if self.done_validation:
                t = Thread(target=self.get_solution_from_faces, args=(), daemon=True)
                t.start()

    def validate_scanned_faces(self):
        all_faces = []
        for i in range(6):
            all_faces += self.scanned_faces[self.face_list[i]]
        self.done_validation = (all_faces.count("White") == 9 and all_faces.count("Green") == 9 and all_faces.count("Yellow") and all_faces.count("Orange") and all_faces.count("Blue") == 9 and all_faces.count("Red") == 9)

    def get_solution_from_faces(self):
        try:
            all_faces = []
            for i in range(6):
                all_faces += self.scanned_faces[self.face_list[i]]
            for i, elem in enumerate(all_faces):
                all_faces[i] = self.translate[elem]
            soltn = solve_best("".join(all_faces))
            # t1 = Thread(target=lambda a, seq, maxlen, tyme : a.append(solve(seq, maxlen, tyme)), args=(soltn, "".join(all_faces), 15, 20,), daemon=True)
            # t2 = Thread(target=lambda a, seq, maxlen, tyme : a.append(solve(seq, maxlen, tyme)), args=(soltn, "".join(all_faces), 25, 10,), daemon=True)
            # t1.start()
            # t2.start()
            # t1.join()
            # t2.join()
            for sol in soltn:
                if self.solution is None or len(sol) < len(self.solution):
                    self.solution = sol
            parsed_solution = []
            for step in self.solution.split():
                if step == "U2":
                    parsed_solution += ["U", "U"]
                elif step == "D2":
                    parsed_solution += ["D", "D"]
                elif step == "L2":
                    parsed_solution += ["L", "L"]
                elif step == "R2":
                    parsed_solution += ["R", "R"]
                elif step == "F2":
                    parsed_solution += ["F", "F"]
                elif step == "B2":
                    parsed_solution += ["B", "B"]
                else:
                    parsed_solution.append(step)
            self.cube = CubeMove(self.scanned_faces, parsed_solution)
        except IndexError:
            self.solution = "Already Solved"
        # else:
        #     self.done_validation = False
