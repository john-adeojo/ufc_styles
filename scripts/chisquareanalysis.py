import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

class ChiSquareAnalysis:
    def __init__(self, df, category):
        self.df = df.copy()
        self.category = category

    def run_chisquare_analysis(self, var):
        
        self.df = self.df.loc[self.df['weight_class'] == self.category].copy()
        
        contingency_table = pd.crosstab(self.df['style_matchup'], self.df[var])

        # Perform the Chi-square test
        chi2_stat, p_value, dof, ex = chi2_contingency(contingency_table)

        # Calculate the standardized residuals
        standardized_residuals = (contingency_table - ex) / np.sqrt(ex)
        
        #standardized_residuals.index.name = None
        standardized_residuals = standardized_residuals.reset_index()

        print("Chi2 Stat:", chi2_stat)
        print("P Value:", p_value)
        print("Degrees of Freedom:", dof)
        # print("Expected Frequency Table:")
        # print(ex)

        return standardized_residuals
