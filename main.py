import cv2
import pygame
from pygame.locals import *
import numpy as np

from OpenGL.GL import *
from OpenGL.GLU import *

verticies = (
    (1, -1, -1),
    (1, 1, -1),
    (-1, 1, -1),
    (-1, -1, -1),
    (1, -1, 1),
    (1, 1, 1),
    (-1, -1, 1),
    (-1, 1, 1)
    )

edges = (
    (0,1),
    (0,3),
    (0,4),
    (2,1),
    (2,3),
    (2,7),
    (6,3),
    (6,4),
    (6,7),
    (5,1),
    (5,4),
    (5,7)
    )


def Cube():
    glBegin(GL_LINES)
    for edge in edges:
        for vertex in edge:
            glVertex3fv(verticies[vertex])
    glEnd()


def main():
    face_cascade = cv2.CascadeClassifier('cv_test/haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)

    pygame.init()
    WIDTH = 1920
    HEIGHT = 1080
    display = (WIDTH,HEIGHT)
    pygame.display.set_mode(display, DOUBLEBUF|OPENGL)

    FOV = 45
    gluPerspective(FOV, (display[0]/display[1]), 0.1, 50.0)

    glRotatef(4,0,1,0)
    glTranslatef(0.0,0.0, -5)

    while True:
        # Find the face in the screen
        ret, img = cap.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in faces:
            cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)

            face_center = (x+w/2, y+h/2)
            face_r = (w+h)/2

        print(face_center, face_r)

        cv2.imshow('img', img)
        k = cv2.waitKey(30) & 0xff
        if k == 27: # Esc key
            break

        # Update cube on screen
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        rot_degrees_y = -(face_center[0] - img.shape[0]/2) / (img.shape[0] / FOV)
        rot_degrees_x = -(face_center[1] - img.shape[1]/2) / (img.shape[1] / FOV)
        view_dist = 2 + (img.shape[0] / face_r)

        glRotatef(rot_degrees_y,0,1,0)
        glRotatef(rot_degrees_x,1,0,0)
        glTranslatef(0.0,0.0, -view_dist)
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
        Cube()
        pygame.display.flip()
        pygame.time.wait(10)
        # Hacky attempt to undo rotation lol
        glTranslatef(0.0,0.0,view_dist)
        glRotatef(-rot_degrees_x,1,0,0)
        glRotatef(-rot_degrees_y,0,1,0)

    # free OpenCV vars
    cap.release()
    cv2.destroyAllWindows()

main()
