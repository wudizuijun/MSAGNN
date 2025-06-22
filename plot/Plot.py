import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from plot.FontFormat import FontFormat
# from FontFormat import FontFormat
from scipy.stats import gaussian_kde



SINGLE_WIDTH = 17 / 2.54
DOUBLE_WIDTH = 8 / 2.54

class Plot():
    def __init__(self, pred, real) -> None:
        """ 对pred和real的结果进行绘制

        :param str data_path: plot data path
        """
        self.font = FontFormat()
        self._set_global_format()
        self.pred = pred
        self.real = real
        
        
    def _set_format(self, ax, border_width, fix_border=True):
        ''' set the format of the ax '''
        # set border line width
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(border_width)
            
        # y刻度封顶
        if fix_border:
            ax.set_yticks(np.linspace(ax.get_yticks()[0], round(ax.get_yticks()[-1],2), 5))
        
        # x,y轴字体
        for label in ax.get_xticklabels():
            label.set_fontproperties(self.font.ticks_font)
        for label in ax.get_yticklabels():
            label.set_fontproperties(self.font.ticks_font)
        
        # legend
        ax.legend(
            prop=self.font.legend_font, loc='best',
            edgecolor = '0', 
            framealpha = 0,
        )
        
    def _set_global_format(self):
        """ 对全局绘图参数进行设置
        """
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
        plt.rcParams['axes.unicode_minus'] = False

        plt.rcParams['xtick.direction']='in'
        plt.rcParams['ytick.direction']='in'
        
    
    def _draw_on_ax(self, ax, line_width=1, border_width=1.5):
        ''' plot on a given ax '''
        ax.plot(self.pred, label='Prediction Value', linewidth=1.5, color='crimson')
        ax.plot(self.real, label='Real Value', linewidth=1.5, color='dodgerblue')
        
        self._set_format(ax, border_width=border_width)

    def plot_loss(self, train_loss, valid_loss, figsize=None, line_width=1):
        ''' plot loss curve of traing and validation
        '''
        if figsize is None:
            figsize = (17/2.54, 5/2.54)
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        # set the format of the ax
        
        ax.plot(train_loss, label='Train Loss', linewidth=line_width, color='dodgerblue')
        ax.plot(valid_loss, label='Valid Loss', linewidth=line_width, color='crimson')
        
        self._set_format(ax, border_width=line_width)
        
        # plt.show()
        
        return fig
        
    def pred_line_plot(self, figsize=None, line_width=1):
        """ plot a single line in ax.

        :param tuple figsize: _description_, defaults to (7,3), 对应inch。 1 inch = 2.54cm
        论文作图图片宽度，单栏宽度(约8厘米),双栏宽度(约17厘米)和中间宽度(约15厘米)。图片的高度一般不超过20厘米。
        字体 8-12号
        """
        if figsize is None:
            figsize = (17/2.54, 5/2.54)
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        
        # set the format of the ax
        self._draw_on_ax(ax, line_width=line_width)
        # plt.show()
        return fig
    

    def r2_plot(self, figsize=None, kde=False):
        """plot the r2 diagran

        :param _type_ figsize: _description_
        """
        c = 'blue'
        
        if kde:
            xy = np.stack([self.real, self.pred])
            c = gaussian_kde(xy)(xy)*len(self.pred)
            fig, ax = plt.subplots(1, 1, figsize=(DOUBLE_WIDTH+1, DOUBLE_WIDTH))
            ax_plot = ax.scatter(self.real, self.pred, s=3, c=c, cmap=plt.get_cmap('jet')) 
            fig.colorbar(ax_plot, ax=ax)
        else:
            fig, ax = plt.subplots(1, 1, figsize=(DOUBLE_WIDTH, DOUBLE_WIDTH))
            ax.scatter(self.real, self.pred, s=3, c=c)
        
        # plt line
        min_x = min(min(self.pred), min(self.real))
        max_x = max(max(self.pred), max(self.real))
        offset = (max_x - min_x) * 0.01
        point_min, point_max = min_x - offset, max_x + offset
        ax.plot([point_min, point_max], [point_min, point_max], color='black', linewidth=1.2)
        
        self._set_format(ax, border_width=1.5, fix_border=False)
        # plt.show()   
        return fig
    
    def box_plot(self, data:np.ndarray=[], xticks:list=[], xlable:str='x label', y_label:str='y label', single_column=False):
        """plot the box plot

        :param np.ndarray data: _description_, defaults to [model_i pred data,..., real data]
        :param list xticks: _description_, defaults to []
        :param str xlable: _description_, defaults to 'x label'
        :param str y_label: _description_, defaults to 'y label'
        """
        # 后续完善，可以绘制不同类型的箱型图，例如absolute error, relative error, error distribution
        # plot_data = cal_fun(data)
        
        if single_column:
            fig, ax = plt.subplots(1, 1, figsize=(SINGLE_WIDTH, SINGLE_WIDTH/1.618))
        else:
            fig, ax = plt.subplots(1, 1, figsize=(DOUBLE_WIDTH, DOUBLE_WIDTH/1.618))
        
        if len(data) == 0:
            # plot self data
            error = self.pred - self.real
            ax.boxplot(error)
        else:
            pass
            
        self._set_format(ax, border_width=1.5, fix_border=True)
        # plt.show()
        return fig
        
    
    def _get_data(self, file_path:str):
        """ get the results data

        :param str file_path: _description_
        :raises ValueError: _description_
        :raises ValueError: _description_
        :return _type_: _description_
        """
        file_type = file_path.split('.')[-1]
        if file_type == 'csv':
            data = pd.read_csv(file_path)
        elif data == 'xlsx':
            data = pd.read_excel(file_path)
        else:
            raise ValueError('file type not supported')
        
        try:
            pred, real = data['prediction'].values, data['real'].values
        except:
            raise ValueError('the column name of the data is not correct')
        
        # check nan in the data    
        if np.any(np.isnan(pred)):
            raise ValueError('the pred data contains nan')
        if np.any(np.isnan(real)):
            raise ValueError('the real data contains nan')
        
        return pred, real
        
        

        
        
            

       

            
if __name__ == '__main__':
    myplot = Plot(r'G:\Code\FrameWork_version_1.0\plot\data\prediction.csv')
    # myplot.pred_line_plot()
    myplot.r2_plot(kde=True)
    plt.show()
    # myplot.box_plot()
    
    