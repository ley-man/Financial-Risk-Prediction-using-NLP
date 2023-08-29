########################### import ###############################
import att_data_preprocess
import PoS_only_train
import att_train
import att_test
import embed_only_train
import torch
from torch.utils.data import *
from datetime import date


def main():
    batch_size = 50
    pos_EPOCH = 70
    embed_EPOCH = 200
    EPOCH = 100

    len_sentences, x_pos_train, x_embed_train, y_train, x_pos_test, x_embed_test, y_test, x_pos_valid, x_embed_valid, y_valid = att_data_preprocess.get_input_data()

    pos_train_data = []
    for i in range(len(x_pos_train)):
        pos_train_data.append([x_pos_train[i], y_train[i]])
    pos_val_data = []
    for i in range(len(x_pos_valid)):
        pos_val_data.append([x_pos_valid[i], y_valid[i]])

    for i in pos_train_data[0]:
        print(i.data, " i shape:", i.size)
    pos_train_data = DataLoader(pos_train_data, batch_size=batch_size, shuffle=True, num_workers=4)
    pos_val_data = DataLoader(pos_val_data, batch_size=batch_size, shuffle=True, num_workers=4)

    pos_best_model = PoS_only_train.train(pos_EPOCH, pos_train_data, pos_val_data)

    embed_train_data = []
    for i in range(len(x_embed_train)):
        embed_train_data.append([x_embed_train[i], y_train[i]])
    embed_val_data = []
    for i in range(len(x_embed_valid)):
        embed_val_data.append([x_embed_valid[i], y_valid[i]])

    embed_train_data = DataLoader(embed_train_data, batch_size=batch_size, shuffle=True, num_workers=4)
    embed_val_data = DataLoader(embed_val_data, batch_size=batch_size, shuffle=True, num_workers=4)
    embed_best_model = embed_only_train.train(embed_EPOCH, len_sentences,embed_train_data, embed_val_data)

    train_pre_pos = pos_best_model(x_pos_train).unsqueeze(1)
    val_pre_pos = pos_best_model(x_pos_valid).unsqueeze(1)
    test_pre_pos = pos_best_model(x_pos_test).unsqueeze(1)

    train_pre_embed = embed_best_model(x_embed_train).unsqueeze(1)
    val_pre_embed = embed_best_model(x_embed_valid).unsqueeze(1)
    test_pre_embed = embed_best_model(x_embed_test).unsqueeze(1)

    x_train = torch.cat([train_pre_pos, train_pre_embed], dim=1).detach()
    x_val = torch.cat([val_pre_pos, val_pre_embed], dim=1).detach()
    x_test = torch.cat([test_pre_pos, test_pre_embed], dim=1).detach()

    x_train_data = []
    for i in range(len(x_pos_train)):
        x_train_data.append([x_train[i], y_train[i]])
    x_val_data = []
    for i in range(len(x_pos_valid)):
        x_val_data.append([x_val[i], y_valid[i]])
    for i in x_train_data[0]:
        print(i)

    x_train_data = DataLoader(x_train_data, batch_size=batch_size, shuffle=True, num_workers=4)
    x_val_data = DataLoader(x_val_data, batch_size=batch_size, shuffle=True, num_workers=4)

    best_model = att_train.train( EPOCH, x_train_data, x_val_data)

    pre_np = att_test.test(best_model, x_test)
    att_test.gen_accu(pre_np, y_test)


if __name__ == '__main__':
    main()