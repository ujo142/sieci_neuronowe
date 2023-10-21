import toml
import utils


if __name__ == "__main__":
    with open('config/config.toml', 'r') as file:
        config = toml.load(file)

    # training data
    X_train, y_train, X_test, y_test = utils.load_data(**config['data'])
    if config['data']['objective'] == 'classification':
        utils.plot_dataset_classification(X_train, y_train)
    else:
        utils.plot_dataset_regression(X_train, y_train)