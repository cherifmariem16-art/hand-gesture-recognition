import cv2
import mediapipe as mp
import math
import numpy as np

# -------------------------
# Settings
# -------------------------
ANGLE_STRAIGHT_DEG = 165
LOVE_DIST_FACTOR   = 0.23
STABLE_FRAMES      = 6
CONF_DET = 0.85
CONF_TRK = 0.85

# -------------------------
# Mediapipe initialization
# -------------------------
mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils
mp_style = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=CONF_DET,
    min_tracking_confidence=CONF_TRK
)

# -------------------------
# Geometry helpers
# -------------------------
def vec(a, b):
    return np.array([b.x - a.x, b.y - a.y, b.z - a.z], dtype=np.float32)

def angle_deg(a, b, c):
    v1, v2 = vec(b, a), vec(b, c)
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return 0.0
    cosang = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return math.degrees(math.acos(cosang))

def dist_norm(a, b):
    return math.sqrt((a.x - b.x)**2 + (a.y - b.y)**2 + (a.z - b.z)**2)

def palm_width(lm):
    return dist_norm(lm[5], lm[17]) + 1e-6

# -------------------------
# Finger detection
# -------------------------
FINGERS = {
    "thumb":  [1, 2, 3, 4],
    "index":  [5, 6, 7, 8],
    "middle": [9, 10, 11, 12],
    "ring":   [13, 14, 15, 16],
    "pinky":  [17, 18, 19, 20]
}
ORDER = ["thumb", "index", "middle", "ring", "pinky"]

def is_finger_straight(lm, chain):
    if chain is FINGERS["thumb"]:
        ang_ip = angle_deg(lm[2], lm[3], lm[4])
        return ang_ip >= ANGLE_STRAIGHT_DEG
    else:
        mcp, pip, dip, tip = chain
        ang_pip = angle_deg(lm[mcp], lm[pip], lm[dip])
        ang_dip = angle_deg(lm[pip], lm[dip], lm[tip])
        return (ang_pip >= ANGLE_STRAIGHT_DEG) and (ang_dip >= ANGLE_STRAIGHT_DEG)

def finger_states(lm):
    return [1 if is_finger_straight(lm, FINGERS[name]) else 0 for name in ORDER]

def thumb_direction_up_down(lm):
    tip, mcp = lm[4], lm[2]
    dy = tip.y - mcp.y
    pw = palm_width(lm)
    return dy < -0.15 * pw, dy > 0.15 * pw


def classify_single_hand(lm):
    states = finger_states(lm)
    s = sum(states)
    thumb_up, thumb_down = thumb_direction_up_down(lm)

    if states == [1,0,0,0,0]:
        if thumb_up: return "Good", states
        if thumb_down: return "Bad", states
    if states == [0,1,0,0,0]: return "Point", states
    if states == [0,1,1,0,0]: return "Peace", states

    if s == 0: return "Run", states
    if s == 5: return "Stop", states
    return "Unknown", states

def classify_two_hands(per_hand):
    labels = [lab for lab,_ in per_hand]
    sums   = [sum(st) for _,st in per_hand]

    if len(per_hand)==2 and labels[0]=="Stop" and labels[1]=="Stop" and sums[0]==5 and sums[1]==5:
        return "Stop", 10

    priority = ["Good", "Bad", "Point", "Peace", "Run", "Stop"]
    for p in priority:
        if p in labels: return p, sum(sums)

    return "Unknown", sum(sums) if sums else 0

# -------------------------
# Stable gesture tracking
# -------------------------
prev_label = "Unknown"
stable_label = "Unknown"
count_same = 0

def update_stable(label_now):
    global prev_label, stable_label, count_same
    if label_now == prev_label:
        count_same += 1
    else:
        prev_label = label_now
        count_same = 1
    if count_same >= STABLE_FRAMES:
        stable_label = label_now
    return stable_label

# -------------------------
# Main loop
# -------------------------
cap = cv2.VideoCapture(0)

while True:
    ok, frame = cap.read()
    if not ok: break

    frame = cv2.flip(frame, 1)                  # Mirror
    frame = cv2.resize(frame, (640,480))       # Resize for speed
    h, w, _ = frame.shape

    try:
        res = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    except Exception as e:
        print("Mediapipe error:", e)
        continue

    per_hand = []
    total_fingers = 0

    if res.multi_hand_landmarks:
        for i, lmset in enumerate(res.multi_hand_landmarks):
            mp_draw.draw_landmarks(frame, lmset, mp_hands.HAND_CONNECTIONS,
                                   mp_style.get_default_hand_landmarks_style(),
                                   mp_style.get_default_hand_connections_style())
            lbl, states = classify_single_hand(lmset.landmark)
            per_hand.append((lbl, states))
            total_fingers += sum(states)

    final_label, tf = classify_two_hands(per_hand)
    final_label = update_stable(final_label)

    cv2.putText(frame, f"Gesture: {final_label}", (40,90),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 4)
    cv2.putText(frame, f"Fingers: {tf}", (40,170),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,0), 4)
    cv2.putText(frame, "ESC to quit", (40,h-40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200,200,200), 2)

    cv2.imshow("Gesture Recognition", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
