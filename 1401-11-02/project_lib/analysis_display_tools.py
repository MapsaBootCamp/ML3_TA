import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay
from sklearn.calibration import CalibrationDisplay

class AnalysisCurvesDisplay:
    
    @staticmethod
    def create_frames():
        fig, ((axlu, axru), (axld, axrd)) = plt.subplots(
            nrows=2, ncols=2, sharex=True, sharey=False, figsize=(13, 11))
        axlu.set_title('Target Probability')
        axlu.set_title('Target Probability')
        axlu.set_xlabel('Predicted probability')
        axlu.set_ylabel('Actual label')
        axlu.set_yticks(ticks=[0,1], labels=['False', 'True'])
        axlu.set_ylim(-0.5, 2.1)

        axld.set_title('ROC Curve')
        axrd.set_title('Precision Recall Curve')
        axru.set_title('Calibration Curve')
        return fig, (axlu, axru, axld, axrd)
        
        
    def __init__(self,data:'pd.DataFrame', name='Classifier'):
        if not {'prob', 'target'} <= set(data.columns):
            raise ValueError("'prob' or 'target' columns does not exist")
            
        self.name = name
        self.data = data
            
    def plot(self, ax_probability, ax_calib, ax_roc, ax_per_recall,
             **kwargs):

        self._plot_probability(ax_probability, **kwargs)
        args = (self.data.target, self.data.prob)
        rcp = RocCurveDisplay.from_predictions(
            *args, ax=ax_roc, name=self.name, **kwargs
        )
        prp = PrecisionRecallDisplay.from_predictions(
            *args, ax=ax_per_recall, name=self.name, **kwargs
        )
        calp =  CalibrationDisplay.from_predictions(
            *args, strategy='quantile', n_bins=7, ax=ax_calib,
            name=self.name, **kwargs
        )
        
    def _plot_probability(self, ax, **kwargs):
        leg = ax.get_legend()
        n = len(leg.get_texts()) if leg else 0
        assert n < 9, 'There are too many plots'
        ax.scatter(
            x=self.data.prob,
            y=np.array(self.data.target) + 0.1*n,
            alpha=0.1,
            label=self.name,
            **kwargs
        )
        ax.legend()
        
    