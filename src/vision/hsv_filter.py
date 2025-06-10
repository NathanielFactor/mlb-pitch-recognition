import cv2
import numpy as np

MIN_AREA = 100  # Minimum contour area to consider
MAX_AREA = 10000  # Maximum contour area to consider

class HSV_Filter:
    def __init__(self, h_min=0, h_max=179, s_min=0, s_max=255, v_min=0, v_max=255):
        self.h_min = h_min
        self.h_max = h_max
        self.s_min = s_min
        self.s_max = s_max
        self.v_min = v_min
        self.v_max = v_max

    def apply(self, frame):
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_bound = (self.h_min, self.s_min, self.v_min)
        upper_bound = (self.h_max, self.s_max, self.v_max)
        mask = cv2.inRange(hsv_frame, lower_bound, upper_bound)
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)  # Remove noise
        masked_frame = cv2.bitwise_and(frame, frame, mask=mask)
        return masked_frame, mask

def find_baseball(frame):
    # HSV values for white color (tunable)
    hsv_filter = HSV_Filter(h_min=0, h_max=179, s_min=0, s_max=50, v_min=200, v_max=255)

    filtered_frame, mask = hsv_filter.apply(frame)

    # Find contours on the mask (white areas)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_AREA or area > MAX_AREA:
            continue
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue  # avoid division by zero
        circularity = 4 * np.pi * (area / (perimeter * perimeter))
        if circularity > 0.7:  # roughly circular shape
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(filtered_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Show the results
    cv2.imshow("Original with Detected Baseball", frame)
    cv2.imshow("Filtered Mask", mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    image_path = "pitch_frame.jpg"  # replace with your image path
    frame = cv2.imread(image_path)

    if frame is None:
        print(f"Error: Could not read image from {image_path}")
    else:
        find_baseball(frame)
