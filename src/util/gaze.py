"""Utility methods for gaze angle and error calculations."""
import cv2 as cv
import numpy as np
import tensorflow as tf


def pitchyaw_to_vector(pitchyaws):
    r"""Convert given yaw (:math:`\theta`) and pitch (:math:`\phi`) angles to unit gaze vectors.

    Args:
        pitchyaws (:obj:`numpy.array`): yaw and pitch angles :math:`(n\times 2)` in radians.

    Returns:
        :obj:`numpy.array` of shape :math:`(n\times 3)` with 3D vectors per row.
    """
    n = pitchyaws.shape[0]
    sin = np.sin(pitchyaws)
    cos = np.cos(pitchyaws)
    out = np.empty((n, 3))
    out[:, 0] = np.multiply(cos[:, 0], sin[:, 1])
    out[:, 1] = sin[:, 0]
    out[:, 2] = np.multiply(cos[:, 0], cos[:, 1])
    return out


def vector_to_pitchyaw(vectors):
    r"""Convert given gaze vectors to yaw (:math:`\theta`) and pitch (:math:`\phi`) angles.

    Args:
        vectors (:obj:`numpy.array`): gaze vectors in 3D :math:`(n\times 3)`.

    Returns:
        :obj:`numpy.array` of shape :math:`(n\times 2)` with values in radians.
    """
    n = vectors.shape[0]
    out = np.empty((n, 2))
    vectors = np.divide(vectors, np.linalg.norm(vectors, axis=1).reshape(n, 1))
    out[:, 0] = np.arcsin(vectors[:, 1])  # theta
    out[:, 1] = np.arctan2(vectors[:, 0], vectors[:, 2])  # phi
    return out

radians_to_degrees = 180.0 / np.pi


def angular_error(a, b):
    """Calculate angular error (via cosine similarity)."""
    a = pitchyaw_to_vector(a) if a.shape[1] == 2 else a
    b = pitchyaw_to_vector(b) if b.shape[1] == 2 else b

    ab = np.sum(np.multiply(a, b), axis=1)
    a_norm = np.linalg.norm(a, axis=1)
    b_norm = np.linalg.norm(b, axis=1)

    # Avoid zero-values (to avoid NaNs)
    a_norm = np.clip(a_norm, a_min=1e-7, a_max=None)
    b_norm = np.clip(b_norm, a_min=1e-7, a_max=None)

    similarity = np.divide(ab, np.multiply(a_norm, b_norm))

    return np.arccos(similarity) * radians_to_degrees


def mean_angular_error(a, b):
    """Calculate mean angular error (via cosine similarity)."""
    return np.mean(angular_error(a, b))


def tensorflow_angular_error_from_pitchyaw(y_true, y_pred):
    """Tensorflow method to calculate angular loss from head angles."""
    def angles_to_unit_vectors(y):
        sin = tf.sin(y)
        cos = tf.cos(y)
        return tf.stack([
            tf.multiply(cos[:, 0], sin[:, 1]),
            sin[:, 0],
            tf.multiply(cos[:, 0], cos[:, 1]),
        ], axis=1)

    with tf.name_scope('mean_angular_error'):
        v_true = angles_to_unit_vectors(y_true)
        v_pred = angles_to_unit_vectors(y_pred)
        return tensorflow_angular_error_from_vector(v_true, v_pred)


def tensorflow_angular_error_from_vector(v_true, v_pred):
    """Tensorflow method to calculate angular loss from 3D vector."""
    with tf.name_scope('mean_angular_error'):
        v_true_norm = tf.sqrt(tf.reduce_sum(tf.square(v_true), axis=1))
        v_pred_norm = tf.sqrt(tf.reduce_sum(tf.square(v_pred), axis=1))

        sim = tf.div(tf.reduce_sum(tf.multiply(v_true, v_pred), axis=1),
                     tf.multiply(v_true_norm, v_pred_norm))

        # Floating point precision can cause sim values to be slightly outside of
        # [-1, 1] so we clip values
        sim = tf.clip_by_value(sim, -1.0 + 1e-6, 1.0 - 1e-6)

        ang = tf.scalar_mul(radians_to_degrees, tf.acos(sim))
        return tf.reduce_mean(ang)


def draw_gaze(image_in, eye_pos, pitchyaw, thickness=2, color=(0, 0, 255)):
    """Draw gaze angle on given image with a given eye positions."""
    image_out = image_in
    if len(image_out.shape) == 2 or image_out.shape[2] == 1:
        image_out = cv.cvtColor(image_out, cv.COLOR_GRAY2BGR)

    cv.arrowedLine(image_out, tuple(np.round(eye_pos).astype(np.int32)),
                   tuple(np.round([pitchyaw[0], pitchyaw[1]]).astype(np.int32)), color,
                   thickness, cv.LINE_AA, tipLength=0.1)
    return image_out


def draw_gaze_point(image_in, gaze_point_coordinate, thickness=2, color=(0, 0, 255)):
    """Draw gaze angle on given image with a given eye positions."""
    image_out = image_in
    if len(image_out.shape) == 2 or image_out.shape[2] == 1:
        image_out = cv.cvtColor(image_out, cv.COLOR_GRAY2BGR)

    cv.circle(image_out,
              tuple(np.round([gaze_point_coordinate[0], gaze_point_coordinate[1]]).astype(np.int32)),
              30,
              color, thickness=-1)
    return image_out


def get_monitor_resolution(monitor_index = 0):
    import subprocess
    output = subprocess.Popen('xrandr | grep "\*" | cut -d" " -f4',shell=True, stdout=subprocess.PIPE).communicate()[0]

    resolution = output.decode().split('\n')[monitor_index].split('x')
    resolution = list(map(int, resolution))         # List to Integer

    print('resolution - x : ', resolution[0], ' / y : ', resolution[1])
    return resolution


def draw_monitor_grid(img, resolution_x, resolution_y, total_grid_count, is_horizon):

    if is_horizon:
        grid_size_unit = int(resolution_y / total_grid_count)
        for cnt in range(0, total_grid_count):
            start_coordinate = (0, grid_size_unit * (cnt + 1))
            end_coordinate = (resolution_x, grid_size_unit * (cnt + 1))
            cv.line(img, start_coordinate, end_coordinate, color=(100, 100, 100), thickness=1)
    else:
        grid_size_unit = int(resolution_x / total_grid_count)
        for cnt in range(0, total_grid_count):
            start_coordinate = (grid_size_unit * (cnt + 1), 0)
            end_coordinate = (grid_size_unit * (cnt + 1), resolution_y)
            cv.line(img, start_coordinate, end_coordinate, color=(100, 100, 100), thickness=1)





    return img
