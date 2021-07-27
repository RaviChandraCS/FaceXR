import numpy as np
import torch

def accuracy_of_net(net, loader):
    """
    This function computes the accuracy of
    the network on the loader argument passed

    Parameters
    ----------
    net : Net
        This is the model object.
    loader : DataLoader
        This contains the data in image label format.

    Returns
    -------
    accuracy : float
        Accuracy of the model.

    """
    net.eval()
    num_correct = 0
    num_samples = 0
    with torch.no_grad():
        for data in loader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images).to(device)
            _, predictions = outputs.max(1)
            num_correct += (predictions == labels).sum()
            num_samples += predictions.size(0)
        accuracy = 100 * num_correct / num_samples
        net.train()
    return accuracy

def MAPE(target, prediction):
    """
    This function computes the mean absolute percentage error
    from the arguments passed

    Parameters
    ----------
    target : FloatTensor
        This tensor contains the target values.
    prediction : FloatTensor
        This tensor contains the predicted values.

    Returns
    -------
    float
        mean absolute percentage error.

    """
    n = len(target)
    target = [target[i] + 1 for i in range(n)]
    prediction = [prediction[i] + 1 for i in range(n)]
    s = 0
    for i in range(n):
        s += (abs((target[i] - prediction[i]) / target[i]))
    s /= n
    return s * 100

def mean_absolute_error(target, prediction):
    """
    This function computes the mean absolute error
    from the arguments passed using L1Loss function
    provided by PyTorch

    Parameters
    ----------
    target : FloatTensor
        This tensor contains the target values.
    prediction : FloatTensor
        This tensor contains the predicted values.

    Returns
    -------
    float
        mean absolute error.

    """
    maeloss = torch.nn.L1Loss()
    output = maeloss(prediction, target)
    return output.item()

def mean_squared_error(target, prediction):
    """
    This function computes the mean squared error
    from the arguments passed using MSELoss function
    provided by PyTorch

    Parameters
    ----------
    target : FloatTensor
        This tensor contains the target values.
    prediction : FloatTensor
        This tensor contains the predicted values.

    Returns
    -------
    float
        mean squared error.

    """
    mseloss = torch.nn.MSELoss()
    output = mseloss(target, prediction)
    return output.item()

def root_mean_squared_error(target, prediction):
    """
    This function computes the root mean squared error
    from the arguments passed using MSELoss function
    provided by PyTorch and sqrt function

    Parameters
    ----------
    target : FloatTensor
        This tensor contains the target values.
    prediction : FloatTensor
        This tensor contains the predicted values.

    Returns
    -------
    float
        root mean squared error.

    """
    loss = torch.nn.MSELoss()
    rmseloss = torch.sqrt(loss(target, prediction))
    return rmseloss.item()

def get_metrics(net, loader):
    """
    This function computes the metrics MAPE, MAE, MSE,
    RMSE, and accuracy for the model on the loader argument

    Parameters
    ----------
    net : Net
        This is the model object.
    loader : DataLoader
        It contains images and their corresponding labels.

    Returns
    -------
    dict
        The result is a dictionary that contains the metrics
        with appropriate keys.

    """
    net.eval()
    pred = []
    target = []
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = net(images).to(device)
        _, predictions = outputs.max(1)
        for p in predictions:
            pred.append(p.item())
        for l in labels:
            target.append(l.item())
    
    pred = list_to_tensor(pred, device)
    target = list_to_tensor(target, device)

    mape = MAPE(target, pred)
    mae = mean_absolute_error(target, pred)
    mse = mean_squared_error(target, pred)
    rmse = root_mean_squared_error(target, pred)
    net.train()
    return {'mape' : mape, 'mae' : mae, 'mse' : mse, 'rmse' : rmse}

def print_metrics(metrics):
    """
    This is a helper function for printing the metrics

    Parameters
    ----------
    metrics : dict
        It contains the metrics values stored with appropriate keys.

    Returns
    -------
    None.

    """
    mape = metrics['mape']
    mae = metrics['mae']
    mse = metrics['mse']
    rmse = metrics['rmse']
    print(f'Mean Absolute Percentage Error:\t{mape}')
    print(f'Mean Absolute Error:\t{mae}')
    print(f'Mean Squared Error:\t{mse}')
    print(f'Root Mean Squared Error:\t{rmse}')