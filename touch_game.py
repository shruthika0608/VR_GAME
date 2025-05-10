import pygame
import random
import cv2
import mediapipe as mp
import numpy as np

# Initialize Pygame
pygame.init()
WIDTH, HEIGHT = 640, 480
win = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Physio Fish Game")

# Load fish image
try:
    fish_img = pygame.image.load("fish1.png").convert_alpha()
    fish_img = pygame.transform.scale(fish_img, (90, 60))  # bigger fish
except Exception as e:
    print("Error loading fish image:", e)
    pygame.quit()
    exit()

# Load fish tank image
try:
    tank_img = pygame.image.load("fish_tank.png").convert_alpha()
    tank_img = pygame.transform.scale(tank_img, (140, 100))
except Exception as e:
    print("Error loading fish tank image:", e)
    pygame.quit()
    exit()

# Load water background video
water_cap = cv2.VideoCapture("water.mp4")
if not water_cap.isOpened():
    print("Error: Could not open water video.")
    pygame.quit()
    exit()

# Game variables
fish_size = (90, 60)
fish_radius = 45
fish_pos = [random.randint(50, WIDTH - 200), random.randint(50, HEIGHT - 150)]
tank_pos = (WIDTH - 160, HEIGHT - 110)
tank_size = (140, 100)
caught_fish_positions = []
score = 0
clock = pygame.time.Clock()

# Mediapipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# OpenCV webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not access webcam.")
    pygame.quit()
    exit()

def convert_coords(x, y):
    return int(WIDTH - x * WIDTH), int(y * HEIGHT)

def rect_collision(rect1, rect2):
    return rect1.colliderect(rect2)

# Draw everything
def draw_game(fish_pos, caught_fish_positions, score, bg_frame):
    bg_frame_rgb = cv2.cvtColor(bg_frame, cv2.COLOR_BGR2RGB)
    bg_frame_rgb = np.rot90(bg_frame_rgb)
    bg_surface = pygame.surfarray.make_surface(bg_frame_rgb)
    win.blit(bg_surface, (0, 0))

    # Draw tank
    win.blit(tank_img, tank_pos)

    # Draw caught fish
    for pos in caught_fish_positions:
        win.blit(fish_img, (pos[0], pos[1]))

    # Draw current fish
    win.blit(fish_img, (fish_pos[0], fish_pos[1]))

    # Draw score
    font = pygame.font.SysFont(None, 40)
    text = font.render(f"Score: {score}", True, (0, 0, 0))
    win.blit(text, (10, 10))

    pygame.display.update()

# Main game loop
running = True
while running:
    success, img = cap.read()
    if not success:
        continue

    ret, water_frame = water_cap.read()
    if not ret:
        water_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, water_frame = water_cap.read()
    if not ret:
        print("Error reading water video frame.")
        break

    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    hand_x, hand_y = None, None
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            lm = handLms.landmark[8]
            hand_x, hand_y = convert_coords(lm.x, lm.y)
            mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Move fish with hand if near
    if hand_x and hand_y:
        dist = ((hand_x - (fish_pos[0] + fish_size[0] // 2)) ** 2 + (hand_y - (fish_pos[1] + fish_size[1] // 2)) ** 2) ** 0.5
        if dist < fish_radius + 10:
            fish_pos = [hand_x - fish_size[0] // 2, hand_y - fish_size[1] // 2]

    # Collision check with tank
    fish_rect = pygame.Rect(fish_pos[0], fish_pos[1], *fish_size)
    tank_rect = pygame.Rect(tank_pos[0], tank_pos[1], *tank_size)

    if rect_collision(fish_rect, tank_rect):
        caught_fish_positions.append(tank_pos)
        score += 1
        fish_pos = [random.randint(50, WIDTH - 200), random.randint(50, HEIGHT - 150)]

    draw_game(fish_pos, caught_fish_positions, score, water_frame)

    cv2.imshow("Webcam View", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        running = False

    clock.tick(60)

# Cleanup
cap.release()
water_cap.release()
cv2.destroyAllWindows()
pygame.quit()
