from keras import Sequential
from keras.layers import Dense
from tensorflow import keras
from train_model import trained_model
import numpy as np
import pygame
import os


WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
D_BLUE = (0, 32, 96)

WIDTH, HEIGHT = 500, 500


def load_model() -> keras.Sequential:
    model = None
    # Try loading model structure and weights from folder
    try:
        model = keras.models.load_model('model')
    except (OSError, AttributeError):
        print(f"Encountered ERROR while loading model.\n"
              f"Building model from saved weights")
    if model is not None:
        print("Loaded model from directory '/model'.")
        return model
    # If above fails try building model layers then loading ONLY weights
    loading_weights_error = False
    model = Sequential(layers=[Dense(units=784, activation='sigmoid', input_dim=784),
                               Dense(units=500, activation='sigmoid'),
                               Dense(units=10, activation='sigmoid')])
    try:
        model.load_weights("model_weights/cp.cpkt")
    except (ImportError, ValueError):
        loading_weights_error = True
        print("Encountered ERROR while loading weights.\n"
              "Ensure module h5py is installed and directory to weights is correct.")
    if loading_weights_error is False:
        print("Created model layers and loaded weights.")
        return model
    # If all above fails then train a model, store it, and return it
    print("Loading model and loading weights failed. Proceeding to\n"
          "build a model, store it and it's weights in /model and /model_weights.")
    if len(os.listdir('/mnist_data')) != 2:
        raise TypeError("Cannot execute above. mnist_data folder does not contain training and test data.")
    return trained_model()


class Window:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.run = True
        self.clock = pygame.time.Clock()
        # non pygame attrs
        self.pixels = [[255 for _ in range(28)] for _ in range(28)]
        self.model = load_model()
        # continuous loop
        self.render()

    def preprocess_input(self):
        rv = []
        for vec in self.pixels:
            for n in vec:
                rv.append(255 - n)
        rv = np.array([np.array(rv, dtype="int16")])
        return rv

    def clear_screen(self):
        self.pixels = [[255 for _ in range(28)] for _ in range(28)]

    def render(self):
        while self.run:
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        self.run = False
                    if event.key == pygame.K_t:
                        print(np.argmax(self.model(self.preprocess_input())))
                    if event.key == pygame.K_c:
                        self.clear_screen()
            self.screen.fill(WHITE)
            # --[render start]-- It might be slightly slower to put
            # the below code blocks i.e render pixels, render border, handle input
            # into functions or methods of class Window --[render start]--

            # Render pixels
            for y,vec in enumerate(self.pixels):
                for x,p in enumerate(vec):
                    pygame.draw.rect(self.screen, (p, p, p),
                                     [x * 10,y * 10, 10, 10])
            # Render border
            pygame.draw.line(self.screen, D_BLUE, (40, 37), (283, 37), 5) # top
            pygame.draw.line(self.screen, D_BLUE, (37, 35), (37, 284), 5) # left
            pygame.draw.line(self.screen, D_BLUE, (40, 282), (282, 282), 5) # bottom
            pygame.draw.line(self.screen, D_BLUE, (282, 35), (282, 284), 5) # right
            # Handle mouse input (for drawing)
            if pygame.mouse.get_pressed(3)[0]:
                x, y = pygame.mouse.get_pos()
                # Checks if mouse is in drawing square
                if not(x >= 320 or x < 40 or y >= 320 or y <= 40):
                    if (o := x % 10) < 5: x -= o
                    else: x += (10 - o)

                    if (o := y % 10) < 5: y -= o
                    else: y += (10 - o)
                    try:
                        if (self.pixels[y//10][x//10]) != 0:
                            self.pixels[y//10][x//10] -= 51
                    except IndexError:
                        pass

            # --[render end]--
            pygame.display.flip()
            self.clock.tick(144)
        pygame.quit()


if __name__ == '__main__':
    Window()