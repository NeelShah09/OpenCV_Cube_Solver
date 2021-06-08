PREVIEW_WIDTH = 60
PREVIEW_HEIGHT = 30

AREA_THRESH_MIN = 50
AREA_THRESH_MAX = 10000

QB_LEN = 51
CUBE_FACE_X = 480
CUBE_FACE_Y = 320

# 2D CUBE CONSTANTS
FACE_GAP = 6
FACE_SIDE_LEN = 45
QB_SIDE_LEN = FACE_SIDE_LEN // 3
PREVIEW_START_X = 20
PREVIEW_START_Y = 20
FACE_POS_START = {"Top": (PREVIEW_START_X + 1 * (FACE_SIDE_LEN + FACE_GAP), PREVIEW_START_Y + 0 * (FACE_SIDE_LEN + FACE_GAP)),
                  "Front": (PREVIEW_START_X + 1 * (FACE_SIDE_LEN + FACE_GAP), PREVIEW_START_Y + 1 * (FACE_SIDE_LEN + FACE_GAP)),
                  "Bottom": (PREVIEW_START_X + 1 * (FACE_SIDE_LEN + FACE_GAP), PREVIEW_START_Y + 2 * (FACE_SIDE_LEN + FACE_GAP)),
                  "Left": (PREVIEW_START_X + 0 * (FACE_SIDE_LEN + FACE_GAP), PREVIEW_START_Y + 1 * (FACE_SIDE_LEN + FACE_GAP)),
                  "Right": (PREVIEW_START_X + 2 * (FACE_SIDE_LEN + FACE_GAP), PREVIEW_START_Y + 1 * (FACE_SIDE_LEN + FACE_GAP)),
                  "Back": (PREVIEW_START_X + 3 * (FACE_SIDE_LEN + FACE_GAP), PREVIEW_START_Y + 1 * (FACE_SIDE_LEN + FACE_GAP))}
