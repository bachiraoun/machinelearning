# data imports
import numpy as np
import pandas as pd

# import matplotlib
import matplotlib
matplotlib.use('WXAgg')
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
from matplotlib.backends.backend_wx import NavigationToolbar2Wx
from matplotlib.figure import Figure

# import wx
import wx


class FeaturesFrame(wx.Frame):
    def __init__(self, features, Y, title="Features"):
        # set index
        self.__index = 0
        # get features
        assert isinstance(features, (pd.DataFrame, np.ndarray))
        if isinstance(features, np.ndarray):
            features = dict([(str(i), features[:,i]) for i in range(features.shape[1])])
            features = pd.DataFrame(features)
        self.__features = features._get_numeric_data()
        # get Y
        assert isinstance(Y, (pd.DataFrame, np.ndarray))
        assert Y.shape in ((self.__features.shape[0], ), (self.__features.shape[0], 1))
        self.__Y = Y
        # create Frame
        wx.Frame.__init__(self, None, title=title)
        self.SetBackgroundColour([255,255,255])
        # panel
        self.__panel = wx.Panel(self)
        # create figure
        self.__figure = Figure()
        self.__axes = self.__figure.add_subplot(111)
        self.__figure.patch.set_color('white')
        self.__canvas = FigureCanvas(self, -1, self.__figure)
        # create main vertical sizer
        self.__mainSizer = wx.BoxSizer(wx.VERTICAL)
        self.__mainSizer.Add(self.__canvas, 1, wx.LEFT | wx.TOP | wx.EXPAND)
        # add toolbar
        toolbar = NavigationToolbar2Wx(self.__canvas)
        toolbar.Realize()
        self.__mainSizer.Add(toolbar, proportion=0, flag=wx.BOTTOM|wx.TOP|wx.EXPAND, border=5)
        # create buttons size
        sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.__previous = wx.Button(self, wx.ID_ANY, 'Previous')
        self.__next     = wx.Button(self, wx.ID_ANY, 'Next')
        sizer.Add(self.__previous, 0, wx.CENTER)
        sizer.Add(self.__next, 0, wx.CENTER)
        self.__mainSizer.Add(sizer, proportion=0, flag=wx.BOTTOM|wx.TOP|wx.EXPAND, border=5)
        # set and fit sizer
        self.SetSizer(self.__mainSizer)
        self.Fit()
        # bind buttons
        self.__previous.Bind(wx.EVT_BUTTON, self.on_previous)
        self.__next.Bind(wx.EVT_BUTTON, self.on_next)

    def on_previous(self, event):
        self.__index = max(0, self.__index-1)
        self.draw(n=self.__index)
    
    def on_next(self, event):
        self.__index = min(self.__index+1, self.__features.shape[1]-1)
        self.draw(n=self.__index)
        
    def draw(self, n=0, marginPercent = 0.05,
                   color='red', markeredgecolor='black',
                   markersize=5, markeredgewidth=1):
        # remove all lines
        while self.__axes.lines:
            self.__axes.lines.pop(0)
        feature = self.__features.ix[:,n]
        self.__axes.plot(feature, self.__Y, 'o', 
                         color=color, markeredgecolor=markeredgecolor,
                         markersize=markersize, markeredgewidth=markeredgewidth)
        self.__axes.set_title(self.__features.columns[n])
        # set x limits
        min, max = np.min(feature), np.max(feature)
        margin   = marginPercent*(max-min)
        self.__axes.set_xlim([min-margin, max+margin])
        # set y limits
        min, max = np.min(self.__Y), np.max(self.__Y)
        margin   = marginPercent*(max-min)
        self.__axes.set_ylim([min-margin, max+margin])
        # draw canvas
        self.__canvas.draw()


def read_and_transform_csv(path):
    CSV = pd.read_csv(path)
    # get data 
    allDataAsNumerical = pd.DataFrame()
    numericalData      = pd.DataFrame()
    categoricalData    = pd.DataFrame()
    for s in CSV:
        if CSV[s].name == "target":
            Y = CSV[s]
        elif CSV[s].name == "id":
            id = CSV[s]
        elif CSV[s].dtype != object:
            numericalData[str(CSV[s].name)] = CSV[s]
            allDataAsNumerical[str(CSV[s].name)] =np.array(CSV[s]).astype(float)
        else:
            categoricalData[str(CSV[s].name)] = CSV[s] 
    ### ######################################################################### ###
    ### ########################### CATEGORICAL VALUES ########################## ###
    CAT_VAL_LUT = {}
    for name in categoricalData:
        lut  = {}
        data = []
        for v in categoricalData[name]:
            nval = lut.get(v, len(lut))
            lut[v] = nval
            data.append(nval)
        CAT_VAL_LUT[name] = lut
        allDataAsNumerical[name] = np.array(data).astype(float)
    # return data    
    return pd.DataFrame(allDataAsNumerical), Y
    
    
    
if __name__ == "__main__":
    # create data
    x    = np.linspace(-10,10,100)
    n = np.random.random((100,6))/10.
    d = {'sin':np.sin(x)+n[:,0], 
         'cos':np.cos(x)+n[:,1], 
         'tan':np.tan(x)+n[:,2], 
         'exp':np.exp(x)+n[:,3], 
         'sqr':x**2+n[:,4], 
         'sqrt':np.sqrt(x)+n[:,5] , 
         'categorical':[str(i) for i in x] }
    data = pd.DataFrame(d)
    Y    = x+5*x**2-(x**3)/10
    # launch 
    app = wx.App()
    fr = FeaturesFrame(features=data, Y=Y)
    fr.draw()
    fr.Show()
    app.MainLoop()
