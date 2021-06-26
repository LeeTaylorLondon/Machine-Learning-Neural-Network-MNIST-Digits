import pygame
from tensorflow import keras
from keras import optimizers
from keras import models
import numpy as np


WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
D_BLUE = (0, 32, 96)

WIDTH, HEIGHT = 500, 500


class Window:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.run = True
        self.clock = pygame.time.Clock()
        # non pygame attrs
        self.pixels = [[255 for _ in range(28)] for _ in range(28)]
        self.model = keras.models.load_model('model')
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
            # --[render start]--

            # render pixels
            for y,vec in enumerate(self.pixels):
                for x,p in enumerate(vec):
                    pygame.draw.rect(self.screen, (p, p, p),
                                     [x * 10,y * 10, 10, 10])
            # render border
            pygame.draw.line(self.screen, D_BLUE, (40, 37), (283, 37), 5) # top
            pygame.draw.line(self.screen, D_BLUE, (37, 35), (37, 284), 5) # left
            pygame.draw.line(self.screen, D_BLUE, (40, 282), (282, 282), 5) # bottom
            pygame.draw.line(self.screen, D_BLUE, (282, 35), (282, 284), 5) # right
            # handle mouse input for drawing
            if pygame.mouse.get_pressed()[0]:
                x, y = pygame.mouse.get_pos()
                # if mouse not in top left
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