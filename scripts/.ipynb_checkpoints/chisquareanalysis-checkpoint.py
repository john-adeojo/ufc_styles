import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

class ChiSquareAnalysis:
    def __init__(self, df, weight_class):
        self.df = df.copy()
        self.weight_class = weight_class

    def run_chisquare_analysis(self, var):
        
        data = self.df.loc[self.df['weight_class'] == self.weight_class].copy()
        
        contingency_table = pd.crosstab(data['style_matchup'], data[var])

        # Perform the Chi-square test
        chi2_stat, p_value, dof, ex = chi2_contingency(contingency_table)

        # Calculate the standardized residuals
        standardized_residuals = (contingency_table - ex) / np.sqrt(ex)
        
        #standardized_residuals.index.name = None
        standardized_residuals = standardized_residuals.reset_index()
        
        print(self.weight_class)
        print("Chi2 Stat:", chi2_stat)
        print("P Value:", p_value)
        print("Degrees of Freedom:", dof)
        # print(contingency_table)
        # print(standardized_residuals)
        # print("Expected Frequency Table:")
        # print(ex)
        if p_value < 0.05:
            return standardized_residuals, self.weight_class
        else:
            return 'Not significant', 'None'
