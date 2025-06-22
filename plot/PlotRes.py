# 主要用于使用shell 脚本自动化运行程序然后对结果进行绘图
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from plot.FontFormat import FontFormat
from FontFormat import FontFormat # for debug
import os
from typing import Optional
import argparse
from datetime import datetime


SINGLE_WIDTH = 17 / 2.54
DOUBLE_WIDTH = 8 / 2.54
COLORS = ['dimgray', 'green', 'dodgerblue', 'red', 'black', 'orange']


class ResPlot():
    def __init__(self, root_path):
        """
        root_path: working space path
        """
        self.root_path = root_path
        self.save_folder = 'multi_folder_res'
        self.timeStamp = datetime.now().strftime("%Y%m%d%H%M%S")
        
        # init 
        self.font = FontFormat()
        self._set_global_format()
        self.WIGHT, self.HEIGHT = SINGLE_WIDTH, 5/2.54 
        self._create_save_foler()
        
    
    def _set_format(self, ax, border_width, fix_border=True):
        """set format for a ax
        input:
            border_witdh: ax 的线框
        """
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
        """set format for all ax"""
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
        plt.rcParams['axes.unicode_minus'] = False

        plt.rcParams['xtick.direction']='in'
        plt.rcParams['ytick.direction']='in'
    
    ''' TODO: 1. 按照plot type 的类型对绘图类型进行拓展,
    2. 对保存的figure进行美化
    '''
    def draw_all(self, folder_names:list, figsize: Optional[tuple]=None,
                     border_width:Optional[float]=None, plot_type='line',
                     param_name:Optional[str]=None) -> None:
        """绘制结果(main api)
        input:
            path: list, 模型预测结果文件夹名称 (要用要加上root path)
            plot_type: str, 绘图类型. if None, plot all type of figure.
            param_name: 绘制不同超参数值下结果对应的超参数值
        """
        all_model_res = self.load_all_preds(folder_names)
        
        # create a figure
        figsize = (self.WIGHT, self.HEIGHT) if figsize is None else figsize
        border_width = 1.5 if border_width is None else border_width
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        
        for idx, model_name in enumerate(all_model_res.keys()):
            if plot_type == 'line':
                # plot the ground true
                if idx == 0:
                    self._drawLine_per_model(all_model_res[model_name]['real'], ax, COLORS[idx], 'Ground truth')
                self._drawLine_per_model(all_model_res[model_name]['predition'], ax, COLORS[idx+1], model_name)
                
            elif plot_type == 'line_with_box':
                df = self._drawLine_all_withBox(res_paths, param_name, ax)
                self._save_dataframe(df, folder_names)
                
        self._set_format(ax, border_width=border_width)

        # save results
        self._save_fig(fig)
        self._save_recored_folders2Txt(folder_names)
    
    
    def _drawLine_per_model(self, data, ax, line_color, label, line_width=1, x=None, 
                           marker:Optional[str]=None, marker_size:Optional[float]=None):
        """ 在ax上绘制一个模型的预测结果
        data _dtype: np.array, prediction value
        label: legend value
        """
        if x is None:
            ax.plot(range(1, len(data)+1), data, label=label, linewidth=line_width, color=line_color,
                    marker=marker, markersize=marker_size)
        else:
            ax.plot(x, data, label=label, linewidth=line_width, color=line_color,
                    marker=marker, markersize=marker_size)
    
    ''' TODO: 增加对结果内容的保存
    需要对绘图的样式进行美化
    '''
    
    def _drawLine_all_withBox(self, folder_names:list, para_name, ax,  figsize: Optional[tuple]=None,
                     border_width:Optional[float]=None, plot_type='line') -> None:
        """绘制所有模型预测折线图结果
        input:
            folder_names: list, 模型预测结果文件夹名称 (要用要加上root path)
        """
        df = self.load_pred_with_diff_parameters(folder_names, para_name)
        y = 'mse'
        gpby = df.groupby('para_value')[y].mean()
        df.boxplot(
            # x = gpby.index.values,
            column=[y], by='para_value', ax=ax, grid=False,
            widths=0.2,
        )
        # line 
        self._drawLine_per_model(
            gpby.values,
            ax=ax, line_color=COLORS[2], label='mean', line_width=1,
            marker='o', marker_size=3
        )
        # add this because using datafame to plot boxplot, the xticklabels is not correct.
        ax.get_figure().suptitle("")
        ax.set_xlabel(para_name)
        return df
    
    def load_pred_with_diff_parameters(self, folder_names:list, para_name) -> pd.DataFrame:
        """ 加载一个模型在不同参属下的预测结果
        folder_names: list, 模型预测结果文件夹名称 (要用要加上root path)
        para_name: str, 模型参数名称
        return: pd.DataFrame
            | model_name | folder_name | para_value | seed |
            example: model_name, para_name,{para_value: {pred: numpy:array, real:...}}
            groupby p
        """
        columns = ['model_name', 'folder_name', 'para_value', 'seed',
                    "rmse", "mse", "mae", "mape", "r2"]
        res_df = pd.DataFrame(columns=columns)
        for folder_name in folder_names:
            folder_path = os.path.join(self.root_path, folder_name)
            model_name = self._get_model_name(folder_path)
            # 只能写的很臭了这块代码，呜呜呜
            txt_file_path = os.path.join(folder_path, model_name, 'record.txt')
            para_val, seed, rmse, mse, mae, mape, r2 = self._get_paramAndseed(txt_file_path, para_name)
            res_df = pd.concat([res_df, pd.DataFrame([[model_name, folder_name, para_val, seed,
                                                       rmse, mse, mae, mape, r2]], 
                                                     columns=columns)])
        return res_df
            
    def _get_paramAndseed(self, txt_file_path, parameter_name):
        with open(txt_file_path, 'r') as f:
            all_txt = f.readlines()
            lines = all_txt[-1]
            args = lines.strip().split(',')
            para_value = float([arg.split('=')[1] for arg in args if arg.split('=')[0].strip() == parameter_name][0])
            seed = int([arg.split('=')[1] for arg in args if arg.split('=')[0].strip() == 'seed'][0])
            rmse = float(all_txt[1].split(':')[1].strip())
            mse = float(all_txt[2].split(':')[1].strip())
            mae  = float(all_txt[3].split(':')[1].strip())
            mape = float(all_txt[4].split(':')[1].strip())
            r2 = float(all_txt[5].split(':')[1].strip())
            return para_value, seed, rmse, mse, mae, mape, r2
    
    def load_all_preds(self, paths:list) -> dict:
        """加载所有模型预测值, predition and real value
        input:
            path: list, 模型预测结果路径
        output:
            {model_1 name : {pred: numpy:array, real:...}, 
             model_2 name: ...}
        """
        # 对应一个模型只有一个结果的
        all_model_res = {}
        for path in paths:
            model_res_path = os.path.join(self.root_path, path)
            model_name, model_res = self._get_model_Info(model_res_path)
            all_model_res[model_name] = model_res
        return all_model_res
        
    def _get_model_Info(self, path):
        """get the model name from the path
        没有后缀的对应模型名称以及预测值(比较粗糙这个判断方法目前)
        input: 
            path: 一次训练保存的文件夹路径
        return:
            {model name : {pred: numpy:array, real:...}}
        """
        model_name = self._get_model_name(path)
        assert model_name != None, "can't find the proper results"
        res = pd.read_csv(os.path.join(path, 'prediction.csv'))
        pred, real = res['prediction'].values, res['real'].values
        return model_name, {'predition': pred, 'real': real}
        
    def _get_model_name(self, path):
        model_name = None
        files = os.listdir(path)
        for file in files:
            if '.' not in file:
                model_name = file
                break
        return model_name
    
    def _create_save_foler(self):
        if not os.path.exists(os.path.join(self.root_path, self.save_folder)):
            os.makedirs(os.path.join(self.root_path, self.save_folder))
        self.save_folder_path = os.path.join(self.root_path, self.save_folder, self.timeStamp)
        os.makedirs(self.save_folder_path)

    def _save_fig(self, fig, save_name='all_models.png'):
        fig.tight_layout()
        fig.savefig(os.path.join(self.save_folder_path, save_name), dpi=300)
    
    def _save_recored_folders2Txt(self, save_path, save_name='res_paths.txt'):
        with open(os.path.join(self.save_folder_path, save_name), 'w') as f:
            for path in res_paths:
                f.write(path + '\n')

    def _save_dataframe(self, df, save_path, save_name='df.csv'):
        df.to_csv(os.path.join(self.save_folder_path, save_name), index=False)
        
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='results')
    parser.add_argument('--res_paths', type=str, required=False, default='none')
    args = parser.parse_args()
    assert args.res_paths != None, "Value Error about 'res_path'"
    print(args.res_paths)
    
    root_path = r'G:\研究生\毕业论文\chapter_III\FrameWork_version_1.0_multiNode_twoConvTunnel\results'
    # used for line plot test
    res_paths = ["20240111202029", "20240111202039", "20240111202050"]
    # used for box plot test    
    path_str = '20250215171539,20250215172047,20250215172557,20250215173104,20250215173614,20250215174126,20250215174639,20250215175147,20250215175651,20250215180206'
    # path_str = '20240112173955,20240112174140,20240112174324,20240112174508,20240112174653,20240112174838,20240112175025,20240112175211,20240112175400,20240112175552,20240112175743,20240112175935,20240112180128,20240112180318,20240112180500,20240112180643,20240112180827,20240112181013,20240112181155,20240112181340,20240112173955,20240112181704,20240112181846,20240112182028,20240112182211,20240112182353,20240112182535,20240112182717,20240112182901,20240112183044,20240112183227,20240112183411,20240112183554,20240112183738,20240112183922,20240112184103,20240112184245,20240112184428,20240112184610,20240112184752,20240112173955,20240112185118,20240112185301,20240112185441,20240112185623,20240112185806,20240112185947,20240112190128,20240112190309,20240112190452'
    path_str = args.res_paths
    res_paths = [t for t in path_str.split(',')]
    resPlot = ResPlot(root_path)
    resPlot.draw_all(res_paths, plot_type='line_with_box', param_name='window_size')    