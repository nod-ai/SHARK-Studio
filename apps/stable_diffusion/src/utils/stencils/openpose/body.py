import cv2
import numpy as np
import math
from scipy.ndimage.filters import gaussian_filter
import torch
import torch.nn as nn
from collections import OrderedDict
from apps.stable_diffusion.src.utils.stencils.openpose.openpose_util import (
    make_layers,
    transfer,
    padRightDownCorner,
)


class BodyPoseModel(nn.Module):
    def __init__(self):
        super(BodyPoseModel, self).__init__()

        # these layers have no relu layer
        no_relu_layers = [
            "conv5_5_CPM_L1",
            "conv5_5_CPM_L2",
            "Mconv7_stage2_L1",
            "Mconv7_stage2_L2",
            "Mconv7_stage3_L1",
            "Mconv7_stage3_L2",
            "Mconv7_stage4_L1",
            "Mconv7_stage4_L2",
            "Mconv7_stage5_L1",
            "Mconv7_stage5_L2",
            "Mconv7_stage6_L1",
            "Mconv7_stage6_L1",
        ]
        blocks = {}
        block0 = OrderedDict(
            [
                ("conv1_1", [3, 64, 3, 1, 1]),
                ("conv1_2", [64, 64, 3, 1, 1]),
                ("pool1_stage1", [2, 2, 0]),
                ("conv2_1", [64, 128, 3, 1, 1]),
                ("conv2_2", [128, 128, 3, 1, 1]),
                ("pool2_stage1", [2, 2, 0]),
                ("conv3_1", [128, 256, 3, 1, 1]),
                ("conv3_2", [256, 256, 3, 1, 1]),
                ("conv3_3", [256, 256, 3, 1, 1]),
                ("conv3_4", [256, 256, 3, 1, 1]),
                ("pool3_stage1", [2, 2, 0]),
                ("conv4_1", [256, 512, 3, 1, 1]),
                ("conv4_2", [512, 512, 3, 1, 1]),
                ("conv4_3_CPM", [512, 256, 3, 1, 1]),
                ("conv4_4_CPM", [256, 128, 3, 1, 1]),
            ]
        )

        # Stage 1
        block1_1 = OrderedDict(
            [
                ("conv5_1_CPM_L1", [128, 128, 3, 1, 1]),
                ("conv5_2_CPM_L1", [128, 128, 3, 1, 1]),
                ("conv5_3_CPM_L1", [128, 128, 3, 1, 1]),
                ("conv5_4_CPM_L1", [128, 512, 1, 1, 0]),
                ("conv5_5_CPM_L1", [512, 38, 1, 1, 0]),
            ]
        )

        block1_2 = OrderedDict(
            [
                ("conv5_1_CPM_L2", [128, 128, 3, 1, 1]),
                ("conv5_2_CPM_L2", [128, 128, 3, 1, 1]),
                ("conv5_3_CPM_L2", [128, 128, 3, 1, 1]),
                ("conv5_4_CPM_L2", [128, 512, 1, 1, 0]),
                ("conv5_5_CPM_L2", [512, 19, 1, 1, 0]),
            ]
        )
        blocks["block1_1"] = block1_1
        blocks["block1_2"] = block1_2

        self.model0 = make_layers(block0, no_relu_layers)

        # Stages 2 - 6
        for i in range(2, 7):
            blocks["block%d_1" % i] = OrderedDict(
                [
                    ("Mconv1_stage%d_L1" % i, [185, 128, 7, 1, 3]),
                    ("Mconv2_stage%d_L1" % i, [128, 128, 7, 1, 3]),
                    ("Mconv3_stage%d_L1" % i, [128, 128, 7, 1, 3]),
                    ("Mconv4_stage%d_L1" % i, [128, 128, 7, 1, 3]),
                    ("Mconv5_stage%d_L1" % i, [128, 128, 7, 1, 3]),
                    ("Mconv6_stage%d_L1" % i, [128, 128, 1, 1, 0]),
                    ("Mconv7_stage%d_L1" % i, [128, 38, 1, 1, 0]),
                ]
            )

            blocks["block%d_2" % i] = OrderedDict(
                [
                    ("Mconv1_stage%d_L2" % i, [185, 128, 7, 1, 3]),
                    ("Mconv2_stage%d_L2" % i, [128, 128, 7, 1, 3]),
                    ("Mconv3_stage%d_L2" % i, [128, 128, 7, 1, 3]),
                    ("Mconv4_stage%d_L2" % i, [128, 128, 7, 1, 3]),
                    ("Mconv5_stage%d_L2" % i, [128, 128, 7, 1, 3]),
                    ("Mconv6_stage%d_L2" % i, [128, 128, 1, 1, 0]),
                    ("Mconv7_stage%d_L2" % i, [128, 19, 1, 1, 0]),
                ]
            )

        for k in blocks.keys():
            blocks[k] = make_layers(blocks[k], no_relu_layers)

        self.model1_1 = blocks["block1_1"]
        self.model2_1 = blocks["block2_1"]
        self.model3_1 = blocks["block3_1"]
        self.model4_1 = blocks["block4_1"]
        self.model5_1 = blocks["block5_1"]
        self.model6_1 = blocks["block6_1"]

        self.model1_2 = blocks["block1_2"]
        self.model2_2 = blocks["block2_2"]
        self.model3_2 = blocks["block3_2"]
        self.model4_2 = blocks["block4_2"]
        self.model5_2 = blocks["block5_2"]
        self.model6_2 = blocks["block6_2"]

    def forward(self, x):
        out1 = self.model0(x)

        out1_1 = self.model1_1(out1)
        out1_2 = self.model1_2(out1)
        out2 = torch.cat([out1_1, out1_2, out1], 1)

        out2_1 = self.model2_1(out2)
        out2_2 = self.model2_2(out2)
        out3 = torch.cat([out2_1, out2_2, out1], 1)

        out3_1 = self.model3_1(out3)
        out3_2 = self.model3_2(out3)
        out4 = torch.cat([out3_1, out3_2, out1], 1)

        out4_1 = self.model4_1(out4)
        out4_2 = self.model4_2(out4)
        out5 = torch.cat([out4_1, out4_2, out1], 1)

        out5_1 = self.model5_1(out5)
        out5_2 = self.model5_2(out5)
        out6 = torch.cat([out5_1, out5_2, out1], 1)

        out6_1 = self.model6_1(out6)
        out6_2 = self.model6_2(out6)

        return out6_1, out6_2


class Body(object):
    def __init__(self, model_path):
        self.model = BodyPoseModel()
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        model_dict = transfer(self.model, torch.load(model_path))
        self.model.load_state_dict(model_dict)
        self.model.eval()

    def __call__(self, oriImg):
        scale_search = [0.5]
        boxsize = 368
        stride = 8
        padValue = 128
        thre1 = 0.1
        thre2 = 0.05
        multiplier = [x * boxsize / oriImg.shape[0] for x in scale_search]
        heatmap_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], 19))
        paf_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], 38))

        for m in range(len(multiplier)):
            scale = multiplier[m]
            imageToTest = cv2.resize(
                oriImg,
                (0, 0),
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_CUBIC,
            )
            imageToTest_padded, pad = padRightDownCorner(
                imageToTest, stride, padValue
            )
            im = (
                np.transpose(
                    np.float32(imageToTest_padded[:, :, :, np.newaxis]),
                    (3, 2, 0, 1),
                )
                / 256
                - 0.5
            )
            im = np.ascontiguousarray(im)

            data = torch.from_numpy(im).float()
            if torch.cuda.is_available():
                data = data.cuda()
            with torch.no_grad():
                Mconv7_stage6_L1, Mconv7_stage6_L2 = self.model(data)
            Mconv7_stage6_L1 = Mconv7_stage6_L1.cpu().numpy()
            Mconv7_stage6_L2 = Mconv7_stage6_L2.cpu().numpy()

            # extract outputs, resize, and remove padding
            heatmap = np.transpose(
                np.squeeze(Mconv7_stage6_L2), (1, 2, 0)
            )  # output 1 is heatmaps
            heatmap = cv2.resize(
                heatmap,
                (0, 0),
                fx=stride,
                fy=stride,
                interpolation=cv2.INTER_CUBIC,
            )
            heatmap = heatmap[
                : imageToTest_padded.shape[0] - pad[2],
                : imageToTest_padded.shape[1] - pad[3],
                :,
            ]
            heatmap = cv2.resize(
                heatmap,
                (oriImg.shape[1], oriImg.shape[0]),
                interpolation=cv2.INTER_CUBIC,
            )

            # paf = np.transpose(np.squeeze(net.blobs[output_blobs.keys()[0]].data), (1, 2, 0))  # output 0 is PAFs
            paf = np.transpose(
                np.squeeze(Mconv7_stage6_L1), (1, 2, 0)
            )  # output 0 is PAFs
            paf = cv2.resize(
                paf,
                (0, 0),
                fx=stride,
                fy=stride,
                interpolation=cv2.INTER_CUBIC,
            )
            paf = paf[
                : imageToTest_padded.shape[0] - pad[2],
                : imageToTest_padded.shape[1] - pad[3],
                :,
            ]
            paf = cv2.resize(
                paf,
                (oriImg.shape[1], oriImg.shape[0]),
                interpolation=cv2.INTER_CUBIC,
            )

            heatmap_avg += heatmap_avg + heatmap / len(multiplier)
            paf_avg += +paf / len(multiplier)

        all_peaks = []
        peak_counter = 0

        for part in range(18):
            map_ori = heatmap_avg[:, :, part]
            one_heatmap = gaussian_filter(map_ori, sigma=3)

            map_left = np.zeros(one_heatmap.shape)
            map_left[1:, :] = one_heatmap[:-1, :]
            map_right = np.zeros(one_heatmap.shape)
            map_right[:-1, :] = one_heatmap[1:, :]
            map_up = np.zeros(one_heatmap.shape)
            map_up[:, 1:] = one_heatmap[:, :-1]
            map_down = np.zeros(one_heatmap.shape)
            map_down[:, :-1] = one_heatmap[:, 1:]

            peaks_binary = np.logical_and.reduce(
                (
                    one_heatmap >= map_left,
                    one_heatmap >= map_right,
                    one_heatmap >= map_up,
                    one_heatmap >= map_down,
                    one_heatmap > thre1,
                )
            )
            peaks = list(
                zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0])
            )  # note reverse
            peaks_with_score = [x + (map_ori[x[1], x[0]],) for x in peaks]
            peak_id = range(peak_counter, peak_counter + len(peaks))
            peaks_with_score_and_id = [
                peaks_with_score[i] + (peak_id[i],)
                for i in range(len(peak_id))
            ]

            all_peaks.append(peaks_with_score_and_id)
            peak_counter += len(peaks)

        # find connection in the specified sequence, center 29 is in the position 15
        limbSeq = [
            [2, 3],
            [2, 6],
            [3, 4],
            [4, 5],
            [6, 7],
            [7, 8],
            [2, 9],
            [9, 10],
            [10, 11],
            [2, 12],
            [12, 13],
            [13, 14],
            [2, 1],
            [1, 15],
            [15, 17],
            [1, 16],
            [16, 18],
            [3, 17],
            [6, 18],
        ]
        # the middle joints heatmap correpondence
        mapIdx = [
            [31, 32],
            [39, 40],
            [33, 34],
            [35, 36],
            [41, 42],
            [43, 44],
            [19, 20],
            [21, 22],
            [23, 24],
            [25, 26],
            [27, 28],
            [29, 30],
            [47, 48],
            [49, 50],
            [53, 54],
            [51, 52],
            [55, 56],
            [37, 38],
            [45, 46],
        ]

        connection_all = []
        special_k = []
        mid_num = 10

        for k in range(len(mapIdx)):
            score_mid = paf_avg[:, :, [x - 19 for x in mapIdx[k]]]
            candA = all_peaks[limbSeq[k][0] - 1]
            candB = all_peaks[limbSeq[k][1] - 1]
            nA = len(candA)
            nB = len(candB)
            indexA, indexB = limbSeq[k]
            if nA != 0 and nB != 0:
                connection_candidate = []
                for i in range(nA):
                    for j in range(nB):
                        vec = np.subtract(candB[j][:2], candA[i][:2])
                        norm = math.sqrt(vec[0] * vec[0] + vec[1] * vec[1])
                        norm = max(0.001, norm)
                        vec = np.divide(vec, norm)

                        startend = list(
                            zip(
                                np.linspace(
                                    candA[i][0], candB[j][0], num=mid_num
                                ),
                                np.linspace(
                                    candA[i][1], candB[j][1], num=mid_num
                                ),
                            )
                        )

                        vec_x = np.array(
                            [
                                score_mid[
                                    int(round(startend[I][1])),
                                    int(round(startend[I][0])),
                                    0,
                                ]
                                for I in range(len(startend))
                            ]
                        )
                        vec_y = np.array(
                            [
                                score_mid[
                                    int(round(startend[I][1])),
                                    int(round(startend[I][0])),
                                    1,
                                ]
                                for I in range(len(startend))
                            ]
                        )

                        score_midpts = np.multiply(
                            vec_x, vec[0]
                        ) + np.multiply(vec_y, vec[1])
                        score_with_dist_prior = sum(score_midpts) / len(
                            score_midpts
                        ) + min(0.5 * oriImg.shape[0] / norm - 1, 0)
                        criterion1 = len(
                            np.nonzero(score_midpts > thre2)[0]
                        ) > 0.8 * len(score_midpts)
                        criterion2 = score_with_dist_prior > 0
                        if criterion1 and criterion2:
                            connection_candidate.append(
                                [
                                    i,
                                    j,
                                    score_with_dist_prior,
                                    score_with_dist_prior
                                    + candA[i][2]
                                    + candB[j][2],
                                ]
                            )

                connection_candidate = sorted(
                    connection_candidate, key=lambda x: x[2], reverse=True
                )
                connection = np.zeros((0, 5))
                for c in range(len(connection_candidate)):
                    i, j, s = connection_candidate[c][0:3]
                    if i not in connection[:, 3] and j not in connection[:, 4]:
                        connection = np.vstack(
                            [connection, [candA[i][3], candB[j][3], s, i, j]]
                        )
                        if len(connection) >= min(nA, nB):
                            break

                connection_all.append(connection)
            else:
                special_k.append(k)
                connection_all.append([])

        # last number in each row is the total parts number of that person
        # the second last number in each row is the score of the overall configuration
        subset = -1 * np.ones((0, 20))
        candidate = np.array(
            [item for sublist in all_peaks for item in sublist]
        )

        for k in range(len(mapIdx)):
            if k not in special_k:
                partAs = connection_all[k][:, 0]
                partBs = connection_all[k][:, 1]
                indexA, indexB = np.array(limbSeq[k]) - 1

                for i in range(len(connection_all[k])):  # = 1:size(temp,1)
                    found = 0
                    subset_idx = [-1, -1]
                    for j in range(len(subset)):  # 1:size(subset,1):
                        if (
                            subset[j][indexA] == partAs[i]
                            or subset[j][indexB] == partBs[i]
                        ):
                            subset_idx[found] = j
                            found += 1

                    if found == 1:
                        j = subset_idx[0]
                        if subset[j][indexB] != partBs[i]:
                            subset[j][indexB] = partBs[i]
                            subset[j][-1] += 1
                            subset[j][-2] += (
                                candidate[partBs[i].astype(int), 2]
                                + connection_all[k][i][2]
                            )
                    elif found == 2:  # if found 2 and disjoint, merge them
                        j1, j2 = subset_idx
                        membership = (
                            (subset[j1] >= 0).astype(int)
                            + (subset[j2] >= 0).astype(int)
                        )[:-2]
                        if len(np.nonzero(membership == 2)[0]) == 0:  # merge
                            subset[j1][:-2] += subset[j2][:-2] + 1
                            subset[j1][-2:] += subset[j2][-2:]
                            subset[j1][-2] += connection_all[k][i][2]
                            subset = np.delete(subset, j2, 0)
                        else:  # as like found == 1
                            subset[j1][indexB] = partBs[i]
                            subset[j1][-1] += 1
                            subset[j1][-2] += (
                                candidate[partBs[i].astype(int), 2]
                                + connection_all[k][i][2]
                            )

                    # if find no partA in the subset, create a new subset
                    elif not found and k < 17:
                        row = -1 * np.ones(20)
                        row[indexA] = partAs[i]
                        row[indexB] = partBs[i]
                        row[-1] = 2
                        row[-2] = (
                            sum(
                                candidate[
                                    connection_all[k][i, :2].astype(int), 2
                                ]
                            )
                            + connection_all[k][i][2]
                        )
                        subset = np.vstack([subset, row])
        # delete some rows of subset which has few parts occur
        deleteIdx = []
        for i in range(len(subset)):
            if subset[i][-1] < 4 or subset[i][-2] / subset[i][-1] < 0.4:
                deleteIdx.append(i)
        subset = np.delete(subset, deleteIdx, axis=0)

        # candidate: x, y, score, id
        return candidate, subset
