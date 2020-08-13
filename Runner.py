import plot_service, data_service
import optimization, opt_utils

plot_service.visualize_dataset()

train_X, train_Y = data_service.load_dataset()

# train 3-layer model
layers_dims = [train_X.shape[0], 5, 2, 1]
learning_rate = 0.0007

optimizers = ['gd', 'momentum', 'adam']

for optimizer in optimizers:

    parameters, costs = optimization.model(train_X, train_Y, layers_dims, optimizer=optimizer)
    plot_service.plot_loss_per_iteration_for_learning_rate(costs, learning_rate)

    predictions = opt_utils.predict(train_X, train_Y, parameters)

    plot_service.plot_decision_boundary(lambda x: opt_utils.predict_dec(parameters, x.T), train_X, train_Y,
                                     "Model with {0} optimization".format(optimizer))


