import matplotlib.pyplot as plt
from cross_validation import *
from missing_values import *
from utils import parse_file


def test(dataset, k=10, trials=10):
    """ Test accuracy for given dataset using fixed probabilities (0.0, 0.1, 0.2, 0.5). """

    # Probabilities to test
    perc = [0.0, 0.1, 0.2, 0.5]
    accs = []

    for p in perc:

        # Modify dataset
        new_dataset = randomly_remove_values(dataset, p)
        handle_missing_values(new_dataset)

        # Evaluate accuracy and print result to console
        err_train, err_test = cross_validation(new_dataset, k, trials)
        err_train, err_test = round(err_train * 100, 4), round(err_test * 100, 4)
        accuracy = round(100 - err_test, 4)
        accs.append(accuracy)
        format_data_for_console_output(new_dataset.name, k, trials, p, err_test, err_train, accuracy)

    return accs


def format_data_for_console_output(name, k, trials, p, err_test, err_train, accuracy):
    k, trials, p, err_test, err_train, accuracy = str(k), str(trials), str(p), str(err_test), str(err_train), str(accuracy)
    print("")
    print("| --------------------------------------------------------------------------------------------------")
    print("| Testing '" + name + "' dataset using " + k + "-fold cross-validation with " + trials + " trial(s) and p = " + p + ":")
    print("|")
    print("| * SCORES = { Training error: " + err_train + "% , Test error: " + err_test + "% }")
    print("| * ACCURACY -~-> " + accuracy + "%")
    print("| --------------------------------------------------------------------------------------------------")


def plot_accuracies(accs, names):
    """ Plot different accuracies using 'matplotlib'. """
    for i in range(len(accs)):
        plt.plot([0.0, 0.1, 0.2, 0.5], accs[i], label=names[i])
    plt.grid()
    plt.ylabel('Accuracy (%)')
    plt.xlabel('Probability of a removed value (p)')
    plt.title('Decision Tree Learning Accuracy - Various Datasets')
    plt.legend()
    plt.show()


if __name__ == '__main__':

    # CAR DATASET
    car = parse_file("datasets/car.txt")
    car_attrnames = ["buying", "maint", "doors", "persons", "lug_boot", "safety", "class"]
    car_dataset = DataSet(name="car", examples=car, attrnames=car_attrnames, target="class")
    car_accuracy = test(car_dataset)

    # PHISHING DATASET
    phishing = parse_file("datasets/phishing.txt")
    phishing_attrnames = ["SFH", "popUpWindow", "SSLfinal_State", "Request_URL", "URL_of_Anchor", "web_traffic",
                          "URL_Length", "age_of_domain", "having_IP_Address", "Result"]
    phishing_dataset = DataSet(name="phishing", examples=phishing, attrnames=phishing_attrnames, target="Result")
    phishing_accuracy = test(phishing_dataset)

    # NURSERY DATASET
    nursery = parse_file("datasets/nursery.txt")
    nursery_attrnames = ["parents", "has_nurs", "form", "children", "housing", "finance", "social", "health", "class"]
    nursery_dataset = DataSet(name="nursery", examples=nursery, attrnames=nursery_attrnames, target="class")
    nursery_accuracy = test(nursery_dataset)

    plot_accuracies([car_accuracy, phishing_accuracy, nursery_accuracy], ["car", "phishing", "nursery"])
