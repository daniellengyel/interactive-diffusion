import matplotlib.pyplot as plt
import numpy as np
import time, os, subprocess


def create_animation_pictures(path, X, Y, Z, graph_type="contour"):
    """
    path: path[:, 0]=path_x, path[:, 1]=path_y, path[:, 2]=path_z"""
    if not os.path.isdir("./tmp"):
        os.mkdir("./tmp")
    ani_path = "./tmp/{}".format(time.time())
    os.mkdir(ani_path)

    for i in range(len(path)):
        fig, ax = plt.subplots()
        ax.contour(X, Y, Z, 40)

        # plot the path
        for j in range(max(0, i - 20), i):
            ax.plot(path[j - 1:j + 1, 0], path[j - 1:j + 1, 1], "--*", color="red", alpha=np.exp(-(i - j - 1) / 5.))

        plt.savefig(ani_path + "/{}.png".format(i))
    return ani_path

# ffmpeg -r 20 -f image2 -s 1920x1080 -i %d.png -vcodec libx264 -crf 25 -pix_fmt yuv420p test.mp4
def create_animation(image_folder, video_name, screen_resolution="1920x1080", framerate=30, qaulity=25,
                     extension=".png"):
    proc = subprocess.Popen(
        [
            "ffmpeg",
            "-r", str(framerate),
            "-f", "image2",
            "-s", screen_resolution,
            "-i", os.path.join(image_folder, "%d" + extension),
            "-vcodec", "libx264",
            "-crf", str(qaulity),
            "-pix_fmt", "yuv420p",
            os.path.join(image_folder, video_name)
        ])
