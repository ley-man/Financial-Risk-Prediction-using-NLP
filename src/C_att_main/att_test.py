########################### import ###############################

from sklearn.metrics import accuracy_score as ac


def gen_accu(pre_np_, y_test):
    predict_result = pre_np_[:, 0, 1]
    predict_ = []
    for item in predict_result:
        if item > 0.5:
            predict_.append(1)
        else:
            predict_.append(0)
    real_result = y_test.tolist()

    accuracy = ac(real_result, predict_)
    print("the accuracy is:", accuracy)

def single_gen_accu(pre_np_, y_test):
    predict_result = pre_np_[:, 1]
    predict_ = []
    for item in predict_result:
        if item > 0.5:
            predict_.append(1)
        else:
            predict_.append(0)

    real_result = y_test.tolist()

    accuracy = ac(real_result, predict_)
    print("the accuracy is:", accuracy)


def test(model, x_pos_test):
    cur_model = model
    prediction = cur_model(x_pos_test.float()).unsqueeze(1)
    pre_np_ = prediction.data.squeeze(dim=1).numpy()
    return pre_np_
