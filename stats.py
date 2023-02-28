import numpy as np 
import pandas as pd
import collections 
import matplotlib.pyplot as plt

class formulas:
    # Dataframe management
    def __init__(self,df) -> None:
        self.df = df.dropna()
    
    def setCol(self,col):

        self.col = self.df[col]

    def createX(self, colName):
        self.x = self.df[colName]
        self.x_name = colName
        self.x_mu = np.mean(self.x)
    
    def createY(self, colName):
        self.y = self.df[colName]
        self.y_name = colName
        self.y_mu = np.mean(self.y)

    def set_x_y(self, x, y):
        self.createX(x)
        self.createY(y)
    
    def corr(self,x,y):
        self.set_x_y(x,y)
        top_term = 0
        btm_term_x = 0
        btm_term_y = 0
        

        n = len(self.x)
        m = len(self.y)

        if n!=m:
            print('woof, cols not equal length')
            return 
        
        for i in range(n):
            top_term += (self.x.iloc[i] - self.x_mu) * (self.y.iloc[i] - self.y_mu)
            btm_term_x += (self.x.iloc[i] - self.x_mu)**2
            btm_term_y += (self.y.iloc[i] - self.y_mu)**2
        
        self.corr = top_term/np.sqrt(btm_term_x * btm_term_y)
        print(self.corr)
        return self.corr

    def pdf(self,col):
        sorted_data = sorted(self.df[col],reverse=False)
        n = len(sorted_data)
        unique_sorted_data = np.unique(sorted_data)
        counter = collections.Counter(sorted_data)
        self.vals, self.cnt = zip(*counter.items())
        self.probVector = [x/n for x in self.cnt]
    
    def pdf_linearBinning(self,col):
        self.pdf(col)
        plt.xlabel('K')
        plt.ylabel('P(K)')
        plt.plot(self.vals, self.probVector,'o')
        plt.show()
    
    def pdf_log_binning(self,col):
        self.pdf(col)
        inMax, inMin = max(self.probVector), min(self.probVector)
        logBins = np.logspace(np.log10(inMin),np.log10(inMax))
        # degree, count
        self.vals, log_bin_edges = np.histogram(self.probVector,bins=logBins,
                                       density=True, range=(inMin, inMax))
        plt.title(f"Log Binning & Scaling")
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel('K')
        plt.ylabel('P(K)')
        plt.plot(self.vals, log_bin_edges[:-1], 'o')
        plt.show()

    def ctl(self, col, samples):
        res = []
        n = len(self.df[col])
        for dataPoint in range(samples):
            idxVector = [ np.random.randint(0,n),np.random.randint(0,n),np.random.randint(0,n)]

            randomSample = 0
            for idx in idxVector:
                randomSample +=  idx
            randomSample /= len(idxVector)
            res.append(randomSample)
        plt.hist(res)
        plt.show()

  
  
df = pd.read_csv('/Users/kjams/Desktop/pyVis/data/nyc.csv')
analysis = formulas(df)
analysis.set_x_y('Poverty','Income')
analysis.pdf_linearBinning('Poverty')
analysis.pdf_log_binning('Poverty')
analysis.ctl('Poverty',1000)
analysis.corr('Poverty','Income')
print(df['Poverty'].corr(df['Income']))