# standard library imports 
import warnings
formatwarning_orig = warnings.formatwarning
warnings.formatwarning = lambda message, category, filename, lineno, line=None: \
formatwarning_orig(message, category, filename, lineno, line='')
    
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
    def __init__(self, features, Y, title="Features", hist=True, histBins=200):
        # set index
        self.__index = 0
        # get features
        assert isinstance(features, (pd.DataFrame, np.ndarray))
        if isinstance(features, np.ndarray):
            features = dict([(str(i), features[:,i]) for i in range(features.shape[1])])
            features = pd.DataFrame(features)
        self.__features = features._get_numeric_data()
        # get Y
        assert isinstance(Y, (pd.DataFrame, pd.Series, np.ndarray))
        assert Y.shape in ((self.__features.shape[0], ), (self.__features.shape[0], 1))
        self.__Y = Y
        self.__ymin, self.__ymax = np.min(self.__Y), np.max(self.__Y)
        self.__yrange = self.__ymax-self.__ymin
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
        self.__canvas.mpl_connect('button_press_event', self.on_mpl_mouse_press)
        self.__canvas.mpl_connect('button_release_event', self.on_mpl_mouse_release)
        self.__canvas.mpl_connect('pick_event', self.on_mpl_mouse_pick)
        self.__canvas.mpl_connect('motion_notify_event', self.on_mpl_mouse_motion)
        # create main vertical sizer
        self.__mainSizer = wx.BoxSizer(wx.VERTICAL)
        self.__mainSizer.Add(self.__canvas, 1, wx.LEFT | wx.TOP | wx.EXPAND)
        # add toolbar
        toolbar = NavigationToolbar2Wx(self.__canvas)
        toolbar.Realize()
        self.__mainSizer.Add(toolbar, proportion=0, flag=wx.BOTTOM|wx.TOP|wx.EXPAND, border=5)
        # create buttons size
        sizer = wx.BoxSizer(wx.HORIZONTAL)
        label = wx.StaticText(self, wx.ID_ANY, label="Index up to %s: "%(self.__features.shape[1]-1) )
        self.__indexTextCtrl     = wx.TextCtrl(self, wx.ID_ANY, str(self.__index))
        self.__previousButton    = wx.Button(self, wx.ID_ANY, 'Previous')
        self.__nextButton        = wx.Button(self, wx.ID_ANY, 'Next')
        sizer.Add(label, 0, wx.CENTER|wx.EXPAND, border=5)
        sizer.Add(self.__indexTextCtrl, 1, wx.CENTER|wx.EXPAND, border=5)
        sizer.Add(self.__previousButton, 0, wx.CENTER, border=5)
        sizer.Add(self.__nextButton, 0, wx.CENTER, border=5)
        self.__mainSizer.Add(sizer, proportion=0, flag=wx.BOTTOM|wx.TOP|wx.EXPAND, border=5)
        # set and fit sizer
        self.SetSizer(self.__mainSizer)
        self.Fit()
        # bind buttons
        self.__indexTextCtrl.Bind(wx.EVT_TEXT, self.on_index_text_control)
        self.__previousButton.Bind(wx.EVT_BUTTON, self.on_previous_button)
        self.__nextButton.Bind(wx.EVT_BUTTON, self.on_next_button)
        # hist
        self.__hist = hist
        self.__histBins = histBins
        # horizontal and vertical lines
        self.__hline1 = self.__hline2 = self.__vline1 = self.__vline2 = None
        self.__dragged = False 
        self.__hlinePoints = None
        self.__vlinePoints = {}

    def on_mpl_mouse_press(self, event):
        hitlist = self.__axes.hitlist(event)
        if self.__hline1 in hitlist:
            self.__dragged = self.__hline1
        elif self.__hline2 in hitlist:
            self.__dragged = self.__hline2
        elif self.__vline1 in hitlist:
            self.__dragged = self.__vline1            
        elif self.__vline2 in hitlist:
            self.__dragged = self.__vline2    
        else: 
            self.__dragged = False        

    def on_mpl_mouse_release(self, event):
        self.__dragged = False
        
    def on_mpl_mouse_pick(self, event):
        pass
            
    def on_mpl_mouse_motion(self, event):
        if self.__dragged:
            if self.__dragged in [self.__vline1, self.__vline2]:
                self.__dragged.set_xdata( [event.xdata, event.xdata] )
                if self.__dragged is self.__vline1:
                   self.__vlinePoints[self.__index] =[event.xdata, self.__vlinePoints[self.__index][1]]
                else:
                   self.__vlinePoints[self.__index] = [self.__vlinePoints[self.__index][0], event.xdata]
            elif self.__dragged in [self.__hline1, self.__hline2]:
                self.__dragged.set_ydata( [event.ydata, event.ydata] )
                if self.__dragged is self.__hline1:
                   self.__hlinePoints = [event.ydata, self.__hlinePoints[1]]
                else:
                   self.__hlinePoints = [self.__hlinePoints[0], event.ydata]
            # draw canvas
            self.__canvas.draw()
            
    def on_previous_button(self, event):
        self.__index = max(0, self.__index-1)
        self.__indexTextCtrl.ChangeValue( str(self.__index) )
        self.draw()
    
    def on_next_button(self, event):
        self.__index = min(self.__index+1, self.__features.shape[1]-1)
        self.__indexTextCtrl.ChangeValue( str(self.__index) )
        self.draw()
        
    def on_index_text_control(self, event):
        try:
            val = int(self.__indexTextCtrl.GetValue())
        except:
            val = self.__index
        val = max(0, val)
        val = min(val, self.__features.shape[1]-1)
        self.__index = val
        self.__indexTextCtrl.ChangeValue( str(self.__index) )
        self.draw()
        
    def draw(self, limMargin=0.05,
                   # feature scatter plot
                   fcolor='red', fmarkeredgecolor='black',
                   fmarkersize=5, fmarkeredgewidth=1,
                   # feature hist
                   histColor='blue', histalpha=0.5, histlinewidth=0, histborderlinewidth=2,
                   # Y vertical plot
                   ylinewidth=2, ycolor='black'):
        # clear axes
        self.__axes.cla()
        self.__axes.set_title("scatter plot of: %s "%self.__features.columns[self.__index])
        self.__axes.set_xlabel("feature data range")
        self.__axes.set_ylabel("Y")
        # plot feature scatter plot
        feature = self.__features.ix[:,self.__index]
        fmin, fmax = float(np.min(feature)), float(np.max(feature))
        frange     = fmax-fmin
        self.__axes.plot(feature, self.__Y, 'o', 
                         color=fcolor, markeredgecolor=fmarkeredgecolor,
                         markersize=fmarkersize, markeredgewidth=fmarkeredgewidth, zorder=1)
        # add horizontal and vertical lines
        x1, x2 = self.__vlinePoints.get(self.__index, [fmin+frange/3., fmax-frange/3.])
        self.__vlinePoints[self.__index] = [x1,x2]
        self.__vline1 = self.__axes.axvline(x1,linewidth=2, linestyle='--',color = 'black', zorder=100)
        self.__vline2 = self.__axes.axvline(x2,linewidth=2, linestyle='--',color = 'black', zorder=100)
        if self.__hlinePoints is None:
            self.__hlinePoints = [self.__ymin+self.__yrange/3., self.__ymax-self.__yrange/3.]
        y1,y2 = self.__hlinePoints
        self.__hline1 = self.__axes.axhline(y1,linewidth=2, linestyle='--',color = 'black', zorder=100)
        self.__hline2 = self.__axes.axhline(y2,linewidth=2, linestyle='--',color = 'black', zorder=100)
        
        # add histogram
        if self.__hist:
            hist , edges = np.histogram(feature, bins=self.__histBins)
            edges = edges.astype(float)
            edges = (edges[:-1]+edges[1:])/2.
            hist  = hist.astype(float)* float(np.max(self.__Y))/float(np.max(hist))
            self.__axes.bar(left=edges, height=hist, width=edges[1]-edges[0],
                            align='center',alpha=histalpha,
                            color=histColor, linewidth=histlinewidth, zorder=10)   
            # plot hist border line
            if histborderlinewidth:
                self.__axes.plot(edges,hist, color=histColor,
                                 linewidth=histborderlinewidth, zorder=10)                               
        # plot y histogram
        if ylinewidth:
            hist , edges = np.histogram(self.__Y, bins=self.__histBins)
            hist = hist.astype(float)
            edges = edges.astype(float)
            edges = (edges[:-1]+edges[1:])/2.
            hist *= 0.4*(fmax-fmin)/np.max(hist)
            self.__axes.plot(fmin+hist, edges, color=ycolor, linewidth=ylinewidth, zorder=20)     
        # set x limits
        margin   = limMargin*(frange)
        self.__axes.set_xlim([fmin-margin, fmax+margin])
        # set y limits
        margin   = limMargin*self.__yrange
        self.__axes.set_ylim([self.__ymin-margin, self.__ymax+margin])
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
        

def drop_stdv_outliers(numericalFeatures, Y, y_nstdv=3, f_nstdv=5, threshold=0.1, 
                       checkFeatures=True, checkY=True):
    assert isinstance(numericalFeatures, (pd.DataFrame, np.ndarray))
    assert isinstance(Y, (pd.DataFrame, pd.Series, np.ndarray))
    assert Y.shape in ((numericalFeatures.shape[0], ), (numericalFeatures.shape[0], 1))
    # initiate outliers
    outliers = []
    # get Y outliers
    if checkY:
        stdv = np.std(Y)
        min, max = -y_nstdv*stdv, y_nstdv*stdv
        outliers = list(np.where(Y<min)[0])
        outliers.extend( list(np.where(Y>max)[0]) )
        if len(outliers) > threshold*Y.shape[0]:
            warnings.warn("y_nstdv %i classifies more than %s %% of data as outliers. condition dropped"%(y_nstdv,threshold*100.))
        else:
            numericalFeatures.drop(numericalFeatures.index[outliers], inplace=True)
            Y.drop(Y.index[outliers], inplace=True)
    if checkFeatures:
        # get features outliers
        for column in numericalFeatures:
            f = numericalFeatures[column]
            stdv = np.std(f)
            min, max = -f_nstdv*stdv, f_nstdv*stdv
            outliers = list(np.where(f<min)[0])
            outliers.extend( list(np.where(f>max)[0]) )
            if len(outliers) > threshold*Y.shape[0]:
                warnings.warn("f_nstdv %i for %s classifies more than %s %% of data as outliers. condition dropped"%(y_nstdv,column, threshold*100.))
            else:
                numericalFeatures.drop(numericalFeatures.index[outliers], inplace=True)
                Y.drop(Y.index[outliers], inplace=True)
    # return 
    return numericalFeatures, Y
    
    
    
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
