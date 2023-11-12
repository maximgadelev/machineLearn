import pygame
import random
import numpy as np
from sklearn.svm import SVC
import sklearn.svm as svm


class Point:
    x = 0
    y = 0
    class_index = 0

    def __init__(self, x, y, class_index):
        self.x = x
        self.y = y
        self.class_index = class_index


def make_line(points, screen):
    X = []
    Y = []

    for point in points:
        X.append([point.x, point.y])
        Y.append(point.class_index)

    svclassifier = SVC(kernel='linear')
    clf = svclassifier.fit(X, Y)

    w = clf.coef_[0]
    b = clf.intercept_[0]
    x_points = np.linspace(0, 1000)
    y_points = -(w[0] / w[1]) * x_points - b / w[1]

    pygame.draw.aaline(screen, (64, 128, 255), [x_points[0], y_points[0]], [x_points[len(x_points) - 1], y_points[len(y_points) - 1]])
    pygame.display.update()

    return svclassifier


if __name__ == "__main__":
    pygame.init()
    screen = pygame.display.set_mode((600, 400), pygame.RESIZABLE)
    screen.fill(color='#FFFFFF')
    pygame.display.update()

    is_active = True
    mousemotion = None
    classifier = None
    points = []
    while is_active:
        for event in pygame.event.get():
            print(event)
            if event.type == pygame.QUIT:
                is_active = False
            if event.type == pygame.MOUSEMOTION:
                mousemotion = event
            if event.type == pygame.KEYUP:
                if event.key == 13:
                    points = []
                    screen.fill(color='#FFFFFF')
                    pygame.display.update()
                if event.key == 32:
                    classifier = make_line(points, screen)
                if event.key == 103:
                    center_coordinates = mousemotion.pos
                    point = Point(center_coordinates[0], center_coordinates[1], 1)
                    points.append(point)
                    pygame.draw.circle(screen, color='green', center=center_coordinates, radius=5)
                    pygame.display.update()
                if event.key == 114:
                    center_coordinates = mousemotion.pos
                    point = Point(center_coordinates[0], center_coordinates[1], -1)
                    points.append(point)
                    pygame.draw.circle(screen, color='red', center=center_coordinates, radius=5)
                    pygame.display.update()
                if event.key == 98:
                    center_coordinates = mousemotion.pos
                    result = classifier.predict([[center_coordinates[0], center_coordinates[1]]])
                    point = Point(center_coordinates[0], center_coordinates[1], result[0])
                    points.append(point)
                    if result[0] == 1:
                        pygame.draw.circle(screen, color='green', center=center_coordinates, radius=5)
                    else:
                        pygame.draw.circle(screen, color='red', center=center_coordinates, radius=5)
                    pygame.display.update()