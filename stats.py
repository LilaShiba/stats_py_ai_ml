import numpy as np 
import pandas as pd
import collections 
import matplotlib.pyplot as plt

class formulas:
    # Dataframe management
    def __init__(self,df) -> None:
        self.df = df.dropna()
        self.n = len(self.df)
        self.entropy_res = collections.defaultdict(int)
        self.prob_res = collections.defaultdict(int)
        self.knnGraph = False
        self.cols = list(self.df.columns.values)
    
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
    
    def get_corr(self,x,y):
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

    def entropy(self, col):
    
        n = len(self.df[col])
        h = collections.defaultdict(int)
        
        for node in self.df[col]:
            h[node] += 1
        p = []
        for x in h.keys():
           p.append((h[x] / n))

        self.p = p
        entropy = round(-np.sum(p * np.log2(p)),2)
        print(entropy)
        self.entropy_res[col] = entropy
        self.prob_res[col] = p

    def dist(self,p1,p2):
        res = 0
        n = len(p1)
        for i in range(n):
            res += (p2[i][0] - p1[i][0])**2 + (p2[i][1] - p1[i][1])**2
        return np.sqrt(res)

    def create_node_list(self,vals):
        self.adjList = collections.defaultdict()
        x,y = vals[0], vals[1]
        names = [x,y]
        self.nodeList = []

        for idx in range(self.n):
            delta = node(   self.df.iloc[idx][x], 
                            self.df.iloc[idx][y],
                            idx,
                            self.df,
                            names         
                        )
            self.nodeList.append(delta)
            self.adjList[idx] = delta
        
    def init_knn(self,clusterSize,vals):
        '''
        vals must be x,y only
        TODO: create dynamic vals ds
        '''
        self.create_node_list(vals)
        self.create_graph()
    
    def create_graph(self):
        if not self.nodeList:
            return ('No node list. Complie first')
            
        graph = collections.defaultdict(list)

        for node in self.nodeList:
            x,y = node.x, node.y 
            graph[(x,y)].append(node)
        
        self.graph = graph
        delta = list(graph.keys())
        x = [i[0] for i in delta]
        y = [i[1] for i in delta]
        delta = plt.scatter(x,y)
        
        self.visualGraph = plt.show()
        return graph
        
class node(formulas):
    
    def __init__(self,x,y,idx,df,names) -> None:
        self.x = x 
        self.y = y 
        self.idx = idx
        self.names = names
        # invoking the __init__ of the parent class
        formulas.__init__(self, df)
        
        

        




# df = pd.read_csv('/Users/kjams/Desktop/pyVis/data/nyc.csv')
# analysis = formulas(df)
# analysis.init_knn(5,['Poverty','Income'])
# analysis.entropy('Income')
# analysis.set_x_y('Poverty','Income')
# analysis.pdf_linearBinning('Poverty')
# analysis.pdf_log_binning('Poverty')
# analysis.ctl('Poverty',1000)
# analysis.corr('Poverty','Income')
# print(df['Poverty'].corr(df['Income']))