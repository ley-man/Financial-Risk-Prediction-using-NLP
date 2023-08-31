########################### import ###############################
import att_model
from att_model import *
from EarlyStopping import *

def train_epoch(model,  training_data, optimizer, loss_func):
    model.train()
    loss_train = []
    for step, (batch_x, batch_y) in enumerate(training_data):
        prediction = model(batch_x).squeeze(dim=1)

        loss = loss_func(prediction, batch_y)

        loss_number = loss.data.numpy()
        loss_train.append(loss_number)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print("###### mean loss is:", np.mean(loss_train))
    return model

def valid_epoch(model, validation_data, loss_func):
    model.eval()
    loss_valid = []
    for step, (batch_x, batch_y) in enumerate(validation_data):
        with torch.no_grad():
            predict_val = model(batch_x.float()).squeeze(dim=1)
            loss_v = loss_func(predict_val, batch_y)
            loss_v_number = loss_v.data.numpy()
            loss_valid.append(loss_v_number)

    mean_val_loss = np.mean(loss_valid)
    print("loss_valid is:", mean_val_loss)
    return mean_val_loss

def train( EPOCH, training_data, validation_data):

    model = att_model.make_connect(d_model=2)

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)

    valid_losses = []
    minvalue = 100.0
    best_model = None
    early_stopping = EarlyStopping(patience=50, verbose=True)
    index = 0
    for epoch in range(EPOCH):
        mult_step_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                                   milestones=[EPOCH // 4,EPOCH // 2, EPOCH // 4 * 3],
                                                                   gamma=0.1)

        print("+" * 40, epoch, "*" * 40)
        weights = [2, 1]
        loss_func = nn.CrossEntropyLoss(torch.FloatTensor(weights))
        train_epoch(model,  training_data, optimizer, loss_func)
        valid_loss = valid_epoch(model, validation_data,loss_func)
        valid_losses += [valid_loss]
        if valid_loss <= minvalue:
            best_model = model
        minvalue = min(valid_losses)
        mult_step_scheduler.step()
        early_stopping(valid_loss, model)

        if early_stopping.early_stop:
            print("Early stopping")
            break
    return best_model
