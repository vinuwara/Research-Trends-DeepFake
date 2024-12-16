import cv2
import mediapipe as mp
import numpy as np
import sys

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)

def get_landmarks(image):
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if not results.multi_face_landmarks:
        raise Exception("No faces detected")
    landmarks = results.multi_face_landmarks[0]
    return np.array([(lm.x * image.shape[1], lm.y * image.shape[0]) for lm in landmarks.landmark])

def annotate_landmarks(im, landmarks):
    for idx, point in enumerate(landmarks):
        pos = (int(point[0]), int(point[1]))
        cv2.circle(im, pos, 2, (0, 255, 0), -1)
    return im

def warp_im(im, M, dshape):
    output_im = np.zeros(dshape, dtype=im.dtype)
    cv2.warpAffine(im, M[:2], (dshape[1], dshape[0]), dst=output_im, borderMode=cv2.BORDER_TRANSPARENT, flags=cv2.WARP_INVERSE_MAP)
    return output_im

def transformation_from_points(points1, points2):
    """
    Return an affine transformation [s * R | T] such that:

        sum ||s*R*p1,i + T - p2,i||^2

    is minimized.

    """
    # Solve the procrustes problem by subtracting centroids, scaling by the
    # standard deviation, and then using the SVD to calculate the rotation. See
    # the following for more details:
    #   https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem

    points1 = points1.astype(np.float64)
    points2 = points2.astype(np.float64)

    c1 = np.mean(points1, axis=0)
    c2 = np.mean(points2, axis=0)
    points1 -= c1
    points2 -= c2

    s1 = np.std(points1)
    s2 = np.std(points2)
    points1 /= s1
    points2 /= s2

    U, S, Vt = np.linalg.svd(points1.T @ points2)

    # The R we seek is in fact the transpose of the one given by U @ Vt. This
    # is because the above formulation assumes the matrix goes on the right
    # (with row vectors) whereas our solution requires the matrix to be on the
    # left (with column vectors).
    R = (U @ Vt).T

    T = c2 - (s2 / s1) * R @ c1.T

    # Stack the transformation matrix
    return np.vstack([
        np.hstack([(s2 / s1) * R, T[:, np.newaxis]]),
        np.array([0., 0., 1.])
    ])

def main():
    im1 = cv2.imread(sys.argv[1])
    im2 = cv2.imread(sys.argv[2])
    landmarks1 = get_landmarks(im1)
    landmarks2 = get_landmarks(im2)

    M = transformation_from_points(landmarks1, landmarks2)
    warped_im2 = warp_im(im2, M, im1.shape)

    output_im = cv2.addWeighted(im1, 0.5, warped_im2, 0.5, 0)
    cv2.imwrite('output.jpg', output_im)

if __name__ == "__main__":
    main()
