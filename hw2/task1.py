import numpy as np
import matplotlib.pyplot as plt
import cv2
import moviepy.editor as mpy

planes = []
for plane_idx in range(1,10):
    with open("hw2_material/Plane_{}.txt".format(plane_idx)) as f:
        all_lines = f.read().splitlines() # read line by line - each line 4 corner points per frame
        frame_points = []
        # Parse corner points for each frame
        for i, line in enumerate(all_lines):
            line_points = line.split(')')
            points = []
            # Parse coordinates of points for each corner point
            for i in range(len(line_points)-1):
                coord = []
                crd = line_points[i].split(' ')
                coord.append(float(crd[0][1:]))
                coord.append(float(crd[1]))
                coord.append(float(crd[2]))
                points.append(coord)
            frame_points.append(points)
    planes.append(frame_points)
planes = np.array(planes)

image = plt.imread('hw2_material/myslovitz.jpg')

source_corners = np.array([[0, 0, 1], [image.shape[0]-1, 0, 1],
[0, image.shape[1]-1, 1],
[image.shape[0]-1, image.shape[1]-1, 1]])

def normalise2Dpts(points):
    saved = points
    points = points.astype(float)
    points[:, 2] = 1
    t_x = points[:, 0].mean()
    t_y = points[:, 1].mean()
    points[:, 0] -= t_x
    points[:, 1] -= t_y
    norm_mean = np.sqrt(points[:, 0] ** 2 + points[:, 1] ** 2).mean()
    scale = np.sqrt(2) / norm_mean
    points[:, 0] *= scale
    points[:, 1] *= scale
    T1 = np.array([[1, 0, -t_x],[0, 1, -t_y],[0, 0, 1]])
    T2 = np.array([[scale, 0, 0],[0, scale, 0],[0, 0, 1]])
    T3 = T2 @ T1
    return points, T3


def to_A2n(sources, dests):
    base_A = np.tile(sources.repeat(2, axis=0), (1, 3))
    to_A = np.array([[0, -dests[0][2], dests[0][1]],
                     [dests[0][2], 0, -dests[0][0]],
                     [0, -dests[1][2], dests[1][1]],
                     [dests[1][2], 0, -dests[1][0]],
                     [0, -dests[2][2], dests[2][1]],
                     [dests[2][2], 0, -dests[2][0]],
                     [0, -dests[3][2], dests[3][1]],
                     [dests[3][2], 0, -dests[3][0]]]).repeat(3, axis=-1)
    return to_A * base_A

def get_H(sources, dests):
    ndests = np.copy(dests)
    ndests[:, 2] = 1
    s_pts, s_T3 = normalise2Dpts(sources)
    t_pts, t_T3 = normalise2Dpts(ndests)
    A = to_A2n(s_pts, t_pts)
    u, s, v = np.linalg.svd(A)
    vec = v[-1].reshape(9, 1)
    H = vec.reshape((3, 3))
    H = np.linalg.inv(t_T3) @ (H @ s_T3)
    return H

t1_list = np.array([get_H(source_corners, p_f) for p_f in planes[0]])
t2_list = np.array([get_H(source_corners, p_f) for p_f in planes[1]])
t3_list = np.array([get_H(source_corners, p_f) for p_f in planes[2]])
t4_list = np.array([get_H(source_corners, p_f) for p_f in planes[3]])
t5_list = np.array([get_H(source_corners, p_f) for p_f in planes[4]])
t6_list = np.array([get_H(source_corners, p_f) for p_f in planes[5]])
t7_list = np.array([get_H(source_corners, p_f) for p_f in planes[6]])
t8_list = np.array([get_H(source_corners, p_f) for p_f in planes[7]])
t9_list = np.array([get_H(source_corners, p_f) for p_f in planes[8]])

source_coords = []
for i in np.linspace(0,image.shape[0]-1,image.shape[0]):
    l = []
    for j in np.linspace(0,image.shape[1]-1,image.shape[1]):
        l.append([j, i, 1])
    source_coords.append(l)
source_coords = np.array(source_coords)
source_coords.shape

tp1_list = np.array([(source_coords @ t1.T) for t1 in t1_list])
tp1_list = np.array([tp1 / tp1[:, :, 2].reshape(-1, image.shape[1], 1) for tp1 in tp1_list])

tp2_list = np.array([(source_coords @ t2.T) for t2 in t2_list])
tp2_list = np.array([tp2 / tp2[:, :, 2].reshape(-1, image.shape[1], 1) for tp2 in tp2_list])

tp3_list = np.array([(source_coords @ t3.T) for t3 in t3_list])
tp3_list = np.array([tp3 / tp3[:, :, 2].reshape(-1, image.shape[1], 1) for tp3 in tp3_list])

tp4_list = np.array([(source_coords @ t4.T) for t4 in t4_list])
tp4_list = np.array([tp4 / tp4[:, :, 2].reshape(-1, image.shape[1], 1) for tp4 in tp4_list])

tp5_list = np.array([(source_coords @ t5.T) for t5 in t5_list])
tp5_list = np.array([tp5 / tp5[:, :, 2].reshape(-1, image.shape[1], 1) for tp5 in tp5_list])

tp6_list = np.array([(source_coords @ t6.T) for t6 in t6_list])
tp6_list = np.array([tp6 / tp6[:, :, 2].reshape(-1, image.shape[1], 1) for tp6 in tp6_list])

tp7_list = np.array([(source_coords @ t7.T) for t7 in t7_list])
tp7_list = np.array([tp7 / tp7[:, :, 2].reshape(-1, image.shape[1], 1) for tp7 in tp7_list])

tp8_list = np.array([(source_coords @ t8.T) for t8 in t8_list])
tp8_list = np.array([tp8 / tp8[:, :, 2].reshape(-1, image.shape[1], 1) for tp8 in tp8_list])

tp9_list = np.array([(source_coords @ t9.T) for t9 in t9_list])
tp9_list = np.array([tp9 / tp9[:, :, 2].reshape(-1, image.shape[1], 1) for tp9 in tp9_list])

kitty = plt.imread('hw2_material/cat-headphones.png')

orig_bg_h, orig_bg_w, ch = (322, 572, 4)
kitty_h, kitty_w, ch = kitty.shape
ratio = orig_bg_h / kitty.shape[-3]
small_kitty = cv2.resize(kitty, (int(kitty_w * ratio), orig_bg_h))
small_kitty_rgb = cv2.cvtColor(small_kitty, cv2.COLOR_RGBA2RGB)
small_kitty.shape

frame_list = []

sort_by_area = True

whole_video = True
max_frame = 5

for idx, (tp1, tp2, tp3, tp4, tp5, tp6, tp7, tp8, tp9) in enumerate(zip(tp1_list, tp2_list,
                                                       tp3_list, tp4_list,
                                                       tp5_list, tp6_list,
                                                       tp7_list, tp8_list,
                                                       tp9_list
                                                      )):
    print("At frame {}".format(idx, end="\r"))
    if sort_by_area:
        p_area = np.array([np.abs(tp1[source_corners[0][1]][source_corners[0][0]][0] - tp1[source_corners[1][1]][source_corners[1][0]][0]) * np.abs(tp1[source_corners[0][1]][source_corners[0][0]][1] - tp1[source_corners[2][1]][source_corners[2][0]][1]),
        np.abs(tp2[source_corners[0][1]][source_corners[0][0]][0] - tp2[source_corners[1][1]][source_corners[1][0]][0]) * np.abs(tp2[source_corners[0][1]][source_corners[0][0]][1] - tp2[source_corners[2][1]][source_corners[2][0]][1]),
        np.abs(tp3[source_corners[0][1]][source_corners[0][0]][0] - tp3[source_corners[1][1]][source_corners[1][0]][0]) * np.abs(tp3[source_corners[0][1]][source_corners[0][0]][1] - tp3[source_corners[2][1]][source_corners[2][0]][1]),
        np.abs(tp4[source_corners[0][1]][source_corners[0][0]][0] - tp4[source_corners[1][1]][source_corners[1][0]][0]) * np.abs(tp4[source_corners[0][1]][source_corners[0][0]][1] - tp4[source_corners[2][1]][source_corners[2][0]][1]),
        np.abs(tp5[source_corners[0][1]][source_corners[0][0]][0] - tp5[source_corners[1][1]][source_corners[1][0]][0]) * np.abs(tp5[source_corners[0][1]][source_corners[0][0]][1] - tp5[source_corners[2][1]][source_corners[2][0]][1]),
        np.abs(tp6[source_corners[0][1]][source_corners[0][0]][0] - tp6[source_corners[1][1]][source_corners[1][0]][0]) * np.abs(tp6[source_corners[0][1]][source_corners[0][0]][1] - tp6[source_corners[2][1]][source_corners[2][0]][1]),
        np.abs(tp7[source_corners[0][1]][source_corners[0][0]][0] - tp7[source_corners[1][1]][source_corners[1][0]][0]) * np.abs(tp7[source_corners[0][1]][source_corners[0][0]][1] - tp7[source_corners[2][1]][source_corners[2][0]][1]),
        np.abs(tp8[source_corners[0][1]][source_corners[0][0]][0] - tp8[source_corners[1][1]][source_corners[1][0]][0]) * np.abs(tp8[source_corners[0][1]][source_corners[0][0]][1] - tp8[source_corners[2][1]][source_corners[2][0]][1]),
        np.abs(tp9[source_corners[0][1]][source_corners[0][0]][0] - tp9[source_corners[1][1]][source_corners[1][0]][0]) * np.abs(tp9[source_corners[0][1]][source_corners[0][0]][1] - tp9[source_corners[2][1]][source_corners[2][0]][1])])
        p_area = np.append(p_area, p_area.mean())
        sorted_planes = np.argsort(p_area)
    else:
        p_z = np.array([planes[0][idx][:, 2].mean(),
                       planes[1][idx][:, 2].mean(),
                       planes[2][idx][:, 2].mean(),
                       planes[3][idx][:, 2].mean(),
                       planes[4][idx][:, 2].mean(),
                       planes[5][idx][:, 2].mean(),
                       planes[6][idx][:, 2].mean(),
                       planes[7][idx][:, 2].mean(),
                       planes[8][idx][:, 2].mean()])
        p_z = np.append(p_z, p_z.mean())
        sorted_planes = np.argsort(p_z)[::-1]

    frame = np.ones((322, 572, 3), dtype=np.uint8) * 255
    
    for p_idx in sorted_planes:
        if p_idx == 0:
            for i in range(image.shape[0]):
                for j in range(image.shape[1]):
                    frame[int(np.around(tp1[i][j][1]))][int(np.around(tp1[i][j][0]))][0] = image[i][j][0]
                    frame[int(np.around(tp1[i][j][1]))][int(np.around(tp1[i][j][0]))][1] = image[i][j][1]
                    frame[int(np.around(tp1[i][j][1]))][int(np.around(tp1[i][j][0]))][2] = image[i][j][2]
        elif p_idx == 1:
            for i in range(image.shape[0]):
                for j in range(image.shape[1]):
                    frame[int(np.around(tp2[i][j][1]))][int(np.around(tp2[i][j][0]))][0] = image[i][j][0]
                    frame[int(np.around(tp2[i][j][1]))][int(np.around(tp2[i][j][0]))][1] = image[i][j][1]
                    frame[int(np.around(tp2[i][j][1]))][int(np.around(tp2[i][j][0]))][2] = image[i][j][2]
        elif p_idx == 2:
            for i in range(image.shape[0]):
                for j in range(image.shape[1]):
                    frame[int(np.around(tp3[i][j][1]))][int(np.around(tp3[i][j][0]))][0] = image[i][j][0]
                    frame[int(np.around(tp3[i][j][1]))][int(np.around(tp3[i][j][0]))][1] = image[i][j][1]
                    frame[int(np.around(tp3[i][j][1]))][int(np.around(tp3[i][j][0]))][2] = image[i][j][2]
        elif p_idx == 3:
            for i in range(image.shape[0]):
                for j in range(image.shape[1]):
                    frame[int(np.around(tp4[i][j][1]))][int(np.around(tp4[i][j][0]))][0] = image[i][j][0]
                    frame[int(np.around(tp4[i][j][1]))][int(np.around(tp4[i][j][0]))][1] = image[i][j][1]
                    frame[int(np.around(tp4[i][j][1]))][int(np.around(tp4[i][j][0]))][2] = image[i][j][2]
        elif p_idx == 4:
            for i in range(image.shape[0]):
                for j in range(image.shape[1]):
                    frame[int(np.around(tp5[i][j][1]))][int(np.around(tp5[i][j][0]))][0] = image[i][j][0]
                    frame[int(np.around(tp5[i][j][1]))][int(np.around(tp5[i][j][0]))][1] = image[i][j][1]
                    frame[int(np.around(tp5[i][j][1]))][int(np.around(tp5[i][j][0]))][2] = image[i][j][2]
        elif p_idx == 5:
            for i in range(image.shape[0]):
                for j in range(image.shape[1]):
                    frame[int(np.around(tp6[i][j][1]))][int(np.around(tp6[i][j][0]))][0] = image[i][j][0]
                    frame[int(np.around(tp6[i][j][1]))][int(np.around(tp6[i][j][0]))][1] = image[i][j][1]
                    frame[int(np.around(tp6[i][j][1]))][int(np.around(tp6[i][j][0]))][2] = image[i][j][2]
        elif p_idx == 6:
            for i in range(image.shape[0]):
                for j in range(image.shape[1]):
                    frame[int(np.around(tp7[i][j][1]))][int(np.around(tp7[i][j][0]))][0] = image[i][j][0]
                    frame[int(np.around(tp7[i][j][1]))][int(np.around(tp7[i][j][0]))][1] = image[i][j][1]
                    frame[int(np.around(tp7[i][j][1]))][int(np.around(tp7[i][j][0]))][2] = image[i][j][2]
        elif p_idx == 7:
            for i in range(image.shape[0]):
                for j in range(image.shape[1]):
                    frame[int(np.around(tp8[i][j][1]))][int(np.around(tp8[i][j][0]))][0] = image[i][j][0]
                    frame[int(np.around(tp8[i][j][1]))][int(np.around(tp8[i][j][0]))][1] = image[i][j][1]
                    frame[int(np.around(tp8[i][j][1]))][int(np.around(tp8[i][j][0]))][2] = image[i][j][2]
        elif p_idx == 8:
            for i in range(image.shape[0]):
                for j in range(image.shape[1]):
                    frame[int(np.around(tp9[i][j][1]))][int(np.around(tp9[i][j][0]))][0] = image[i][j][0]
                    frame[int(np.around(tp9[i][j][1]))][int(np.around(tp9[i][j][0]))][1] = image[i][j][1]
                    frame[int(np.around(tp9[i][j][1]))][int(np.around(tp9[i][j][0]))][2] = image[i][j][2]
        elif p_idx == 9:
            move_x = (572 - 326) // 2
            for i in range(322):
                for j in range(326):
                    if small_kitty[i][j][3] > 0:
                        frame[i][j + move_x][0] = int(small_kitty_rgb[i][j][0] * 255)
                        frame[i][j + move_x][1] = int(small_kitty_rgb[i][j][1] * 255)
                        frame[i][j + move_x][2] = int(small_kitty_rgb[i][j][2] * 255)
    opened_frame = cv2.morphologyEx(frame, cv2.MORPH_OPEN, np.ones((2,2),np.uint8))
    frame_list.append(opened_frame)
    if whole_video is False and idx == max_frame - 1:
        break
frame_list = np.array(frame_list)
frame_list.shape

clip = mpy.ImageSequenceClip(list(frame_list), fps = 25)
clip.write_videofile("last_video.mp4", codec="libx264")
