import matplotlib.pyplot as plt
import pandas as pd

from movie_globals import *

def make_epoch_chart(data, title, ylabel, figure_name, show=False):
    plt.figure(figsize=(6, 4))

    for d in data:
        plt.plot(data[d], label=d)

    plt.ylabel(ylabel)
    plt.xlabel('Epoch')
    plt.title(title)
    plt.legend()
    plt.subplots_adjust(left=0.15)
    plt.savefig(os.path.join(CHART_DIR, f'{figure_name}.png'), dpi=300, bbox_inches="tight")
    if show:
        plt.show()


# def make_charts(model_name=MODEL_NAME):
#     csv_name = f'{model_name}.csv'
#     df = pd.read_csv(os.path.join(CSV_DIR, csv_name))
#     loss = df[[' train_loss', ' valid_loss']]
#     accuracy = df[[' train_accuracy', ' valid_accuracy']]
#     make_epoch_chart(loss, 'Loss per Epoch', 'Loss', f'Loss_{model_name}', show=True)
#     make_epoch_chart(accuracy, 'Accuracy per Epoch', 'Accuracy', f'Accuracy_{model_name}', show=False)
