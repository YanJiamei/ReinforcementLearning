import numpy as np


def MajorVote(feed_data, Env_1, Env_2, Env_3):
    y_ = feed_data['label'].as_matrix()

    re_1 = Env_1.get_softmax_output(feed_data)
    re_2 = Env_2.get_softmax_output(feed_data)
    re_3 = Env_3.get_softmax_output(feed_data)
    prediction_prob_sum = re_1 + re_2 + re_3
    predict_y = np.argmax(prediction_prob_sum, axis=1)

    equal_result = np.equal(predict_y, y_)
    equal_result = equal_result.astype(np.float32)
    acc = np.mean(equal_result, dtype=np.float32)
    return acc
