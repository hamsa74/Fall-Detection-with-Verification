import math

def evaluate_posture(box_coords, landmarks):
    if box_coords is None or landmarks is None:
        return False

    _, _, width, height = box_coords
    
    # 1. Ratio Check (Width vs Height)
    posture_ratio = width / float(height)
    
    # 2. Angle Check (Torso Angle using Landmarks 11 and 23)
    shoulder = landmarks[11]
    hip = landmarks[23]
    
    dy = hip.y - shoulder.y
    dx = hip.x - shoulder.x
    
    # Calculate angle relative to vertical axis
    angle_deg = abs(math.degrees(math.atan2(dy, dx)))
    
    # Logic: Fall is confirmed if Ratio is wide AND body is inclined
    is_inclined = angle_deg < 45 or angle_deg > 135

    if posture_ratio > 1.2 and is_inclined:
        return True
    return False