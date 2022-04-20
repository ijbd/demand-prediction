import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pandas as pd
import os

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(os.path.dirname(THIS_DIR), "output")
GALLERY_DIR = os.path.join(os.path.dirname(THIS_DIR), "gallery")

BALANCING_AUTHORITIES = ["CISO", "ERCO", "MISO", "PGE", "PJM", "SOCO", "SWPP"]

def plot_temp_demand(ax, temp_demand: pd.DataFrame):
    ax.scatter(temp_demand["Temperature (K)"], temp_demand["Demand (MW)"], s=1)
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel("Demand (MW)")

    return None

def plot_prediction_series(ax, pred: pd.DataFrame):
    ax.plot(pred["labels"].values)
    ax.plot(pred["predictions"].values)
    ax.set_xlabel("Day of Year")
    ax.set_ylabel("Demand (MW)")
    ax.set_xlim([0,365])
    ax.legend(["Observed", "Predicted"])
        
    return None

def plot_prediction_scatter(ax, pred: pd.DataFrame):
    ax.scatter(pred["labels"], pred["predictions"])
    ax.set_xlabel("Observed Demand (MW)")
    ax.set_ylabel("Predicted Demand (MW)")
    
    return None

def plot_history(ax, hist: pd.DataFrame):
    ax.plot(hist["loss"])
    ax.plot(hist["val_loss"])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("RMSE")
    ax.set_xlim([0,len(hist)-1])
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(["Train", "Validation"])

    return None

def make_plots():
    plt.style.use("ggplot") 
    plt.tight_layout()

    for bal_auth in BALANCING_AUTHORITIES:
        # plot temp-demand
        temp_demand_file = os.path.join(OUTPUT_DIR, f"{bal_auth}_cleaned_data.csv")
        temp_demand = pd.read_csv(temp_demand_file, index_col="Datetime")

        fig, ax = plt.subplots()
        
        plot_temp_demand(ax, temp_demand)
        ax.set_title(f"{bal_auth} Temperature versus Demand")
        
            # save and close
        temp_demand_plot_file = os.path.join(GALLERY_DIR, f"{bal_auth}_temp_demand.png")
        plt.savefig(temp_demand_plot_file)
        plt.close()

        # plot predictions
        pred_file = os.path.join(OUTPUT_DIR, f"{bal_auth}_ann_test_predictions.csv")
        pred = pd.read_csv(pred_file, index_col="Datetime")
        
        fig, axs = plt.subplots(2, 1, figsize=(7, 10))
        
        plot_prediction_series(axs[0], pred)
        plot_prediction_scatter(axs[1], pred)
        
        axs[0].set_title(f"{bal_auth} Test Predictions")

            # save and close
        prediction_plot_file = os.path.join(GALLERY_DIR, f"{bal_auth}_predictions.png")
        plt.savefig(prediction_plot_file)
        plt.close() 

        # plot history
        hist_file = os.path.join(OUTPUT_DIR, f"{bal_auth}_ann_history.csv")
        hist = pd.read_csv(hist_file, index_col=0)

        fig, ax = plt.subplots()
        plot_history(ax, hist)
        
        ax.set_title(f"{bal_auth} Training History")
        
            # save and close
        history_plot_file = os.path.join(GALLERY_DIR, f"{bal_auth}_history.png")
        plt.savefig(history_plot_file)
        plt.close()
    
    return None

if __name__ == "__main__":
    make_plots()
