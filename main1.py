import cv2
from scene import Scene
import taichi as ti
from taichi.math import *

scene = Scene(voxel_edges=0)
scene.set_floor(-0.77, (1.0, 1.0, 1.0))

image = cv2.imread("pic/1.png")
image = cv2.resize(image, (100, 100))
pixel = ti.Vector.field(3, ti.f32, shape=(100, 100))
pixel.from_numpy(image)


@ti.kernel
def initialize_voxels():
    for i, j in ti.ndrange((-50, 50), 25):
        if ti.random() < 0.05:
            scene.set_voxel(vec3(i, -50, j), 2, vec3(1, 1, 1))
    for i, j in ti.ndrange(100, 100):
        bgr = pixel[i, j]
        scene.set_voxel(vec3(j - 50, 50 - i, 0), 1, vec3(bgr[2] / 255, bgr[1] / 255, bgr[0] / 255))


if __name__ == "__main__":
    initialize_voxels()
    scene.finish()
